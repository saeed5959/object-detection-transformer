import torch
from torch import nn
from torch.nn.functional import layer_norm
from einops import rearrange

from core.settings import model_config


class LinearProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_divide =  model_config.num_divide
        self.source = model_config.source
        self.patch_dim = 3*self.num_divide^2
        self.linear = nn.Linear(self.patch_dim, self.patch_dim)
        self.resnet = nn.Sequential(
            nn.Conv2d(3,5,kernel_size=3),
            nn.Conv2d(5,7),
            nn.Conv2d(7,3),
            nn.Linear(self.patch_dim,self.patch_dim)
        )
        

    def forward(self, x):
        x = self.divide_patch(x)
        x = self.resnet(x)
        x = self.position_embedding(x)

        return x
    
    def divide_patch(self, x):
        #x : [B, h, w, 3]
        if self.source:
            out = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.divide_patch, pw = self.divide_patch)
        else:
            out = rearrange(x, 'b c (h ph) (w pw) -> b (h w) c ph pw ', ph = self.divide_patch, pw=self.divide_patch)
        
        return out
    
    def baseline(self, x):
        if self.source:
            out = self.linear(x)
            out = layer_norm(x)
        else:
            out = self.resnet(x)

        return out
    
    def position_embedding(self):

        return