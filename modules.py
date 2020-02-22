#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np




class Modulated_Conv2d(nn.Conv2d):
    '''
    Modulated convolution layer. Implemented efficiently using grouped convolutions.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size,
                 demodulate=True, bias=True, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups=1,
                         bias=bias, padding_mode='zeros')
        self.demodulate = demodulate
        # style mapping
        self.style = nn.Linear(latent_size, in_channels)
        torch.nn.init.normal_(self.style.weight)
        # required shape might be different in transposed conv
        self.s_broadcast_view = (-1,1,self.in_channels,1,1)
        self.in_channels_dim = 2


    def convolve(self,x,w,groups):
        # bias would be added later
        return F.conv2d(x, w, None, self.stride, self.padding, self.dilation, groups=groups)


    def forward(self, x, v):
        N, in_channels, H, W = x.shape

        # new minibatch dim: (ch dims, K, K) -> (1, ch dims, K, K)
        w = self.weight.unsqueeze(0)

        # compute styles: (N, C_in)
        s = self.style(v) + 1

        # modulate: (N, ch dims, K, K)
        w = s.view(self.s_broadcast_view)*w

        # demodulate
        if self.demodulate:
            sigma = torch.sqrt((w**2).sum(dim=[self.in_channels_dim,3,4],keepdim=True) + 1e-8)
            w = w/sigma

        # reshape x: (N, C_in, H, W) -> (1, N*C_in, H, W)
        x = x.view(1, -1, H, W)

        # reshape w: (N, C_out, C_in, K, K) -> (N*C_out, C_in, K, K) for common conv
        #            (N, C_in, C_out, K, K) -> (N*C_in, C_out, K, K) for transposed conv
        w = w.view(-1, w.shape[2], w.shape[3], w.shape[4])

        # use groups so that each sample in minibatch has it's own conv,
        # conv weights are concatenated along dim=0
        out = self.convolve(x,w,N)

        # reshape back to minibatch.
        out = out.view(N,-1,out.shape[2],out.shape[3])

        # add bias
        if not self.bias is None:
            out += self.bias.view(1, self.bias.shape[0], 1, 1)

        return out





class Up_Mod_Conv(Modulated_Conv2d):
    '''
    Modulated convolution layer with upsampling by some factor, implemented with transposed conv.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size,
                 demodulate=True, bias=True, factor=2):
        assert (kernel_size % 2 == 1)
        padding = (max(kernel_size-factor,0)+1)//2
        super().__init__(in_channels, out_channels, kernel_size, latent_size, demodulate, bias,
                         stride=factor, padding=padding)
        self.output_padding = torch.nn.modules.utils._pair(2*padding - kernel_size + factor)
        # transpose as expected in F.conv_transpose2d
        self.weight = nn.Parameter(self.weight.transpose(0,1).contiguous())
        self.transposed = True
        # taking into account transposition
        self.s_broadcast_view = (-1,self.in_channels,1,1,1)
        self.in_channels_dim = 1

    def convolve(self, x, w, groups):
        return F.conv_transpose2d(x, w, None, self.stride, self.padding, self.output_padding, groups, self.dilation)






class Down_Mod_Conv(Modulated_Conv2d):
    '''
    Modulated convolution layer with downsampling by some factor
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size,
                 demodulate=True, bias=True, factor=2):
        assert (kernel_size % 2 == 1)
        padding = kernel_size//2
        super().__init__(in_channels, out_channels, kernel_size, latent_size, demodulate, bias,
                         stride=factor, padding=padding)

    def convolve(self, x, w, groups):
        return F.conv2d(x, w, None, self.stride, self.padding, self.dilation, groups=groups)




class Down_Conv2d(nn.Conv2d):
    '''
    Convolution layer with downsampling by some factor
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, factor=2):
        assert (kernel_size % 2 == 1)
        padding = kernel_size//2
        super().__init__(in_channels, out_channels, kernel_size, factor, padding, bias=True)

    def convolve(self, x):
        return F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)





class Noise(nn.Module):
    '''
    Add normal noise with learnable magnitude.
    '''
    def __init__(self):
        super().__init__()
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, input_noise=None):
        input_noise = input_noise if input_noise else torch.randn(x.shape[0],1,x.shape[2],x.shape[3])
        noise = self.noise_strength*input_noise
        return x + noise





class Mapping(nn.Module):
    '''
    Mapping network. Transforms the input latent code to the disentangled latent representation.
    '''
    def __init__(self, n_layers, latent_size, nonlinearity, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.layers = []
        for idx in range(n_layers):
            layer = nn.Linear(latent_size, latent_size)
            self.add_module(str(idx), layer)
            self.layers.append(layer)
            self.layers.append(nonlinearity)

    def forward(self, input):
        if self.normalize:
            input = input/torch.sqrt((input**2).mean(dim=1, keepdim=True) + 1e-8)
        for module in self.layers:
            input = module(input)
        return input





class G_Block(nn.Module):
    '''
    Basic block for generator. Increases spatial dimensions by predefined factor.
    '''
    def __init__(self, in_fmaps, out_fmaps, kernel_size, latent_size, nonlinearity, factor=2, img_channels=3):
        super().__init__()
        inter_fmaps = (in_fmaps + out_fmaps)//2
        self.upconv = Up_Mod_Conv(in_fmaps, inter_fmaps, kernel_size, latent_size,
                                      factor=factor)
        self.conv = Modulated_Conv2d(inter_fmaps, out_fmaps, kernel_size, latent_size,
                                     padding=kernel_size//2)
        self.noise = Noise()
        self.to_channels = Modulated_Conv2d(out_fmaps, img_channels, kernel_size=1,
                                      latent_size=latent_size, demodulate = False)
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)
        self.act = nonlinearity

    def forward(self, x, v, y=None, input_noises=[None,None]):
        x = self.noise(self.upconv(x,v), input_noises[0])
        x = self.act(x)
        x = self.noise(self.conv(x,v), input_noises[1])
        x = self.act(x)
        if not y is None:
            y = self.upsample(y)
        else:
            y = 0
        y = y + self.act(self.to_channels(x,v))
        return x, y




class D_Block(nn.Module):
    '''
    Basic block for discriminator. Decreases spatial dimensions by predefined factor.
    '''
    def __init__(self, in_fmaps, out_fmaps, kernel_size, nonlinearity, factor=2):
        super().__init__()
        inter_fmaps = (in_fmaps + out_fmaps)//2
        self.conv = nn.Conv2d(in_fmaps, inter_fmaps, kernel_size, padding=kernel_size//2)
        self.downconv = Down_Conv2d(inter_fmaps, out_fmaps, kernel_size, factor=factor)
        self.down = Down_Conv2d(in_fmaps, out_fmaps, kernel_size=1, factor=factor)
        self.act = nonlinearity

    def forward(self, x):
        t = x
        x = self.conv(x)
        x = self.act(x)
        x = self.downconv(x)
        x = self.act(x)
        t = self.down(t)
        return (x + t)/ np.sqrt(2)




class Minibatch_Stddev(nn.Module):
    '''
    Minibatch standard deviation layer.
    '''
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        s = x.shape
        t = x.view(self.group_size, -1, s[1], s[2], s[3])
        t = t - t.mean(dim=0, keepdim=True)
        t = torch.sqrt((t**2).mean(dim=0) + 1e-8)
        t = t.mean(dim=[1,2,3], keepdim=True) # [N/G,1,1,1]
        t = t.repeat(self.group_size,1,1,1).expand(x.shape[0],1,*x.shape[2:])
        return torch.cat((x,t),dim=1)
