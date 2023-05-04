import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from object_detection.transformer import Attention, ResBlock
from object_detection.utils import make_mask, mono_alighn
from core.settings import model_config

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_phon: int = model_config.n_phon
        self.embed_size: int = model_config.embed_size
        self.max_input: int = model_config.max_input
        self.kernel_size:int = model_config.kernel_size
        self.p_dropout: int = model_config.p_dropout
        
        self.phon_embed = nn.Embedding(self.n_phon, self.embed_size)
        
        self.trans_attention = Attention(self.kernel_size, self.p_dropout)
        
        self.proj = nn.Conv1d(self.max_input, self.max_input * 2, 1)
        
        
        
    def forward(self, x, x_length, speaker_embed, emotion_embed):
        
        x_embed = self.phon_embed(x)
        
        x_mask = make_mask(x_length, self.max_input)
        
        x_attn = self.trans_attention(x_embed, x_mask)
        
        x_attn = x_attn + speaker_embed + emotion_embed
        
        x_proj = self.proj(x_attn) * x_mask
        
        u, sigma = torch.split(x_proj, self.n_phon, dim=1)
        
        return u, sigma, x_attn


class DurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_phon: int = model_config.n_phon
        self.embed_size: int = model_config.embed_size
        self.max_input: int = model_config.max_input
        self.kernel_size:int = model_config.kernel_size
        self.p_dropout: int = model_config.p_dropout
        
        
        
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
    def __init__(self):
        super(Generator, self).__init__()
        self.num_kernels = len(model_config.resblock_kernel_sizes)
        self.num_upsamples = len(model_config.upsample_rates)
        self.conv_pre = nn.Conv1d(model_config.initial_channel, model_config.upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock()

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = model_config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(model_config.resblock_kernel_sizes, model_config.resblock_dilation_sizes)):
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
        
        
        
class TextToSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.text_encoder = TextEncoder()
        self.duration_pred = DurationPredictor()
        self.flow = FlowBaseBlock()
        self.generator = Generator()
        
        self.speaker_embed_layer = nn.Embedding(10,128)
        self.emotion_embed_layer = nn.Embedding(5,128)
        
        
    def forward(self, x, x_length, y, y_length, speaker_id, emotion_id):
        
        speaker_embed = self.speaker_embed_layer(speaker_id)
        emotion_embed = self.emotion_embed_layer(emotion_id)
        
        u, sigma, x_attn = self.text_encoder(x, x_length, speaker_embed, emotion_embed)
        
        x_dur_pred = self.duration_pred(x_attn)
                
        y_flow = self.flow(y, y_length)
        
        x_dur_mono = mono_alighn(u, sigma, y_flow)
        
        out = self.generator(y, y_length)
        
        return out, x_dur_mono, x_dur_pred
        
        
        
        
        
        
