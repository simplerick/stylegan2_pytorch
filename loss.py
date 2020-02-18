import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def G_logistic_ns(fake_logits):
    return -F.logsigmoid(fake_logits).mean() # -log(D(G(z)))


def D_logistic(real_logits, fake_logits):
    return torch.mean(-F.logsigmoid(real_logits) + F.softplus(fake_logits)) # -log(D(x)) - log(1-D(G(z)))



def R1_reg(real_imgs, real_logits):
    grads = torch.autograd.grad(real_logits.sum(), real_imgs, create_graph=True)[0]
    # Disable gradients for imgs
    real_imgs.reqires_grad = False
    return torch.mean(grads**2)



class Path_length_reg(nn.Module):
    def __init__(self, decay=0.01):
        super().init()
        self.decay = decay
        self.avg = 0

    def forward(self, latent, gen_out):
        # Compute |J*y|.
        noise = torch.randn(gen_out.shape) #[N,Channels,H,W]
        grads = torch.autograd.grad((gen_out * noise).sum(), latent, create_graph=True)[0]  #[N,latent_size]
        # Disable gradients for latent
        latent.reqires_grad = False
        lengths = torch.sqrt((grads**2).sum(1)) #[N]
        # Update exp average. Lengths are detached
        self.avg = self.decay*torch.mean(lengths.detach()) + (1-self.decay)*self.avg
        return torch.mean((lengths - self.avg)**2)
