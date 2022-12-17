import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from emotional_tts.modules import Attention, ResBlock
from emotional_tts.utils import make_mask

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_phon: int = 178
        self.embed_size: int = 128
        self.max_input: int = 192
        self.kernel_size:int = 3
        self.p_dropout: int = 0.5
        
        self.phon_embed = nn.Embedding(self.n_phon, self.embed_size)
        
        self.trans_attention = Attention(self.kernel_size, self.p_dropout)
        
        self.proj = nn.Conv1d(self.max_input, self.max_input * 2, 1)
        
        
        
    def forward(self, x, x_length):
        
        x_embed = self.phon_embed(x)
        
        x_mask = make_mask(x_length, self.max_input)
        
        x_attn = self.trans_attention(x_embed, x_mask)
        
        x_proj = self.proj(x_attn) * x_mask
        
        u, sigma = torch.split(x_proj, self.n_phon, dim=1)
        
        return u, sigma, x_attn


class DurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_size: int = 128
        self.max_input: int = 192
        self.kernel_size:int = 3
        self.p_dropout: int = 0.5
        
        
        self.conv_1 = nn.Conv1d(self.max_input, self.max_input,self.kernel_size)
        self.norm_1 = nn.LayerNorm(self.max_input)
        self.drop_out = nn.Dropout(self.p_dropout)
        
        self.conv_2 = nn.Conv1d(self.max_input, self.max_input, self.kernel_size)
        self.norm_2 = nn.LayerNorm(self.max_input)
        
        self.linear_layer = nn.Linear(self.embed_size,1)
        
    def forward(self, x, x_mask):
        
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop_out(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop_out(x)
        
        x = self.linear_layer(x * x_mask)
        
        return x * x_mask
        

#not completed
class FlowBaseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x, x_mask):
        
        return
    
    
    
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock()

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False) 

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
        
        
        
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        norm_f = weight_norm 
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap     
        
        
        
        
        
        
        
        
        
        
        
        
