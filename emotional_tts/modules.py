import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

from core.settings import model_config
        
        
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        channels = model_config.hidden_channels
        self.kernel_size:int = model_config.kernel_size
        
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