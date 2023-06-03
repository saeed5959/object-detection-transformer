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
        self.patch_num = model_config.patch_num
        self.patch_size = model_config.patch_size
        self.linear = nn.Linear(self.dim, self.dim)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
        )
        self.conv2d_2 =nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
        )
        self.conv2d_3 =nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
        )
        self.conv2d_4 =nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=7, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=7, padding='same'),
            nn.BatchNorm2d(512),
        )
        self.conv1d_1 = nn.Conv2d(3, 64, kernel_size=1, padding='same')
        self.conv1d_2 = nn.Conv2d(64, 128, kernel_size=1, padding='same')
        self.conv1d_3 = nn.Conv2d(128, 256, kernel_size=1, padding='same')
        self.conv1d_4 = nn.Conv2d(256, 512, kernel_size=1, padding='same')
        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.maxpool_3 = nn.MaxPool2d(2)
        self.maxpool_4 = nn.MaxPool2d(2)
        self.pos_embed = nn.Embedding(self.patch_num,self.dim)
        

    def forward(self, x):

        x = self.resnet(x)
        x = layer_norm(x, [x.size()[-3], x.size()[-2], x.size()[-1]])
        x = self.divide_patch(x, downsample=16)
            
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

        x_conv2d_3 = self.conv2d_3(x_maxpool_2)
        x_maxpool_3 = self.maxpool_3(relu(x_conv2d_3 + self.conv1d_3(x_maxpool_2)))

        x_conv2d_4 = self.conv2d_4(x_maxpool_3)
        x_maxpool_4 = self.maxpool_4(relu(x_conv2d_4 + self.conv1d_4(x_maxpool_3)))

        return x_maxpool_4
    
    def position_embedding(self, x):
        #using a learnable 1D-embedding in a raster order
        batch_number, patch_number, dim_size = x.size()
        pos = torch.arange(patch_number).repeat(batch_number,1).to(device)
        out = self.pos_embed(pos)

        return out
    
