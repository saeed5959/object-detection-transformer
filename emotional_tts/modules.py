import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import functional as F
        
        
        
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 192
        self.kernel_size:int = 3
        
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, self.kernel_size, 1)),
            weight_norm(nn.Conv1d(channels, channels, self.kernel_size, 1)),
            weight_norm(nn.Conv1d(channels, channels, self.kernel_size, 1))
        ])
        
    def forward(self, x):
        
        xt = F.leaky_relu(x)
        x = self.convs(xt)
        return x
    
    
#not completed
class Attention(nn.Module):
    def __init__(self):
        super().__init__()