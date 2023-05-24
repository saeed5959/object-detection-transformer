import torch
from torch import nn
from torch.nn.functional import layer_norm, relu
from einops import rearrange

from core.settings import model_config, train_config

device = train_config.device

class LinearProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim =  model_config.dim
        self.source = model_config.source
        self.patch_num = model_config.patch_num
        self.patch_size = model_config.patch_size
        self.linear = nn.Linear(self.dim, self.dim)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
        )
        self.conv2d_2 =nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, padding='same'),
            nn.BatchNorm2d(48),
        )
        self.conv1d_1 = nn.Conv2d(3, 32, kernel_size=1, padding='same')
        self.conv1d_2 = nn.Conv2d(32, 48, kernel_size=1, padding='same')
        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.pos_embed = nn.Embedding(self.patch_num,self.dim)
        

    def forward(self, x):
        if self.source:
            x = self.divide_patch(x,downsample=1)
            x = layer_norm(x, [x.size()[-2], x.size()[-1]])
            x = self.linear(x)
            
        else:
            x = self.resnet(x)
            x = layer_norm(x, [x.size()[-2], x.size()[-1]])
            x = self.divide_patch(x, downsample=4)
            
        pos = self.position_embedding(x)
        out = x+pos

        return out
    
    def divide_patch(self, x, downsample):
        out = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.patch_size // downsample, pw = self.patch_size // downsample)
        
        return out
    
    def resnet(self, x):
        x_conv2d_1 = self.conv2d_1(x)
        x_maxpool_1 = self.maxpool_1(relu(x_conv2d_1 + self.conv1d_1(x)))

        x_conv2d_2 = self.conv2d_2(x_maxpool_1)
        x_maxpool_2 = self.maxpool_2(relu(x_conv2d_2 + self.conv1d_2(x_maxpool_1)))

        return x_maxpool_2
    
    def position_embedding(self, x):
        #using a learnable 1D-embedding in a raster order
        batch_number, patch_number, dim_size = x.size()
        pos = torch.arange(patch_number).repeat(batch_number,1).to(device)
        out = self.pos_embed(pos)

        return out
    
