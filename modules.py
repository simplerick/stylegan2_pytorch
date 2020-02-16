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
                 demodulate=True, bias=True, stride=1, padding=0, dilation=1):
        assert (kernel_size % 2 == 1)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups=1,
                         bias=bias, padding_mode='zeros')
        self.demodulate = demodulate
        # style mapping
        self.style = nn.Linear(latent_size, in_channels)
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
        s = self.style(v)

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
    Convolution layer with upsampling by some factor, implemented with transposed conv.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size,
                 demodulate=True, bias=True, factor=2):
        padding = (max(kernel_size-factor,0)+1)//2
        super().__init__(in_channels, out_channels, kernel_size, latent_size, demodulate, bias,
                         stride=factor, padding=padding)
        self.output_padding = 2*padding - kernel_size + factor
        # transpose as expected in F.conv_transpose2d
        self.weight = nn.Parameter(self.weight.transpose(0,1).contiguous())
        # taking into account transposition
        self.s_broadcast_view = (-1,self.in_channels,1,1,1)
        self.in_channels_dim = 1

    def convolve(self, x, w, groups):
        return F.conv_transpose2d(x, w, None, self.stride, self.padding, self.output_padding, groups, self.dilation)






class Down_Mod_Conv(Modulated_Conv2d):
    '''
    Convolution layer with downsampling by some factor
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size,
                 demodulate=True, bias=True, factor=2):
        if kernel_size > factor:
            padding, dilation = kernel_size//2, 1
        else:
            dilation = int((factor+kernel_size-3)/(kernel_size-1))
            k = dilation*(kernel_size-1) + 1
            padding = k//2
        super().__init__(in_channels, out_channels, kernel_size, latent_size, demodulate, bias,
                         stride=factor, padding=padding, dilation=dilation)

    def convolve(self, x, w, groups):
        return F.conv2d(x, w, None, self.stride, self.padding, self.dilation, groups=groups)





class Noise(nn.Module):
    '''
    Add normal noise with learnable magnitude.
    '''
    def __init__(self):
        super().__init__()
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        noise = self.noise_strength*torch.randn(x.shape[0],1,x.shape[2],x.shape[3])
        return x + noise





class Mapping(nn.Module):
    '''
    Mapping network. Transforms the input latent code to the disentangled latent representation.
    '''
    def __init__(self, n_layers, n_features, nonlinearity, normalize=True):
        super().__init__()
        self.normalize = normalize
        for idx in range(n_layers):
            self.add_module(str(2*idx), nn.Linear(n_features, n_features))
            self.add_module(str(2*idx+1), nonlinearity)

    def forward(self, input):
        if normalize:
            input = input/torch.sqrt((input**2).mean(dim=1, keepdim=True) + 1e-8)
        for module in self:
            input = module(input)
        return input





class G_Block(nn.Module):
    '''
    Basic block for generator.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, latent_size, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        inter_channels = (in_channels + out_channels)//2
        self.upconv = Up_Mod_Conv(in_channels, inter_channels, kernel_size, latent_size,
                                      factor=2)
        self.conv = Modulated_Conv2d(inter_channels, out_channels, kernel_size, latent_size,
                                     padding=kernel_size//2)
        self.noise = Noise()
        self.toRGB = Modulated_Conv2d(out_channels, 3, kernel_size=1,
                                      latent_size=latent_size, demodulate = False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.act = nonlinearity

    def forward(self, x,v,y=0):
        x = self.noise(self.upconv(x,v))
        x = self.act(x)
        x = self.noise(self.conv(x,v))
        x = self.act(x)
        if y != 0:
            y = self.upsample(y)
        y = y + self.act(self.toRGB(x,v))
        return x, y
