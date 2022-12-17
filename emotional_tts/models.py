import torch
from torch import nn

from emotional_tts.modules import Attention
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
        
        return u, sigma



