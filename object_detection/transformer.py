import torch
from torch import nn
from torch.nn.functional import softmax, relu, layer_norm
from einops import rearrange

from core.settings import model_config
        
        
class SelfAttention_me(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.linear_1 = nn.Linear(self.dim, self.dim)
        self.linear_2 = nn.Linear(self.dim, self.dim)
        self.linear_3 = nn.Linear(self.dim, self.dim)

    def forward(self, x : torch.Tensor):

        ### Scaled Dot-Product Attention
        V = x  # x : [batch_size, patch_size, vector_dim]
        V_linear = self.linear_1(V)
        K = x
        K_linear = self.linear_2(K)
        Q = x
        Q_linear = self.linear_1(Q)

        out = torch.matmul(softmax(torch.matmul(Q_linear, K_linear.transpose(1,2))/torch.sqrt(torch.tensor(self.dim)), dim=-1), V_linear)

        return out


class MultiHeadAttention_me(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.head_num = model_config.head_num
        self.heads = nn.ModuleList([SelfAttention_me() for _ in range(self.head_num)]) 
        self.linear = nn.Linear(self.dim*self.head_num, self.dim)

    def forward(self, x: torch.Tensor):
        out = [attention_head(x) for attention_head in self.heads]
        out = torch.cat(tuple(out),dim=-1)
        out = self.linear(out)
        
        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_num = model_config.head_num
        self.dim = model_config.dim
        self.head_dim = self.dim // self.head_num
        self.linear_v = nn.Linear(self.dim, self.dim)
        self.linear_k = nn.Linear(self.dim, self.dim)
        self.linear_q = nn.Linear(self.dim, self.dim)
        self.linear_out = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_q.weight)

    def forward(self, x : torch.Tensor):

        ### Scaled Dot-Product Attention
        V = x  
        V_linear = self.linear_v(V)
        K = x
        K_linear = self.linear_k(K)
        Q = x
        Q_linear = self.linear_q(Q)

        #reshaping x for multihead
        V_linear = rearrange(V_linear, 'b n (h d) -> b h n d', h = self.head_num)
        K_linear = rearrange(K_linear, 'b n (h d) -> b h n d', h = self.head_num)
        Q_linear = rearrange(Q_linear, 'b n (h d) -> b h n d', h = self.head_num)
        
        out = torch.matmul(softmax(torch.matmul(Q_linear, V_linear.transpose(2,3))/torch.sqrt(torch.tensor(self.dim)), dim=-1), V_linear)

        #back reshaping from multihead
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.linear_out(out)
        
        return out
 

    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.multihead = MultiHeadAttention() 
        self.linear_1 = nn.Linear(self.dim, 2048)
        self.linear_2 = nn.Linear(2048, self.dim)

    def forward(self, x: torch.Tensor):
        out_multihead = self.multihead(x)
        out_multihead_add_norm = layer_norm(out_multihead + x, [x.size()[-1]])

        out_FFN = self.linear_1(out_multihead_add_norm)
        out_FFN = relu(out_FFN)
        out_FFN = self.linear_2(out_FFN)

        out = layer_norm(out_FFN + out_multihead_add_norm, [x.size()[-1]])

        return out
    
