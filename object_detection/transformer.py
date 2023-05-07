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

        out = torch.matmul(softmax(torch.matmul(K_linear, Q_linear.transpose(1,2))/torch.sqrt(self.dim), dim=-1), V_linear)

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

    def forward(self, x : torch.Tensor):
        ### Scaled Dot-Product Attention
        V = x  
        V_linear = self.linear_v(V)
        K = x
        K_linear = self.linear_k(K)
        Q = x
        Q_linear = self.linear_q(Q)

        #reshaping x for multihead
        ####
        #vits solution that I think it is not true
        #x = x.view(b, self.num_head, self.head_dim, m).transpose(2, 3)
        
        #git solution that are same with my solution
        #x = rearrange(x, 'b n (h d) -> b h n d', h = self.num_head)
        ###
        #my solution : you god damn right :)
        V_linear = torch.stack(list(torch.split(V_linear, self.head_dim, -1)), dim=0)
        K_linear = torch.stack(list(torch.split(K_linear, self.head_dim, -1)), dim=0)
        Q_linear = torch.stack(list(torch.split(Q_linear, self.head_dim, -1)), dim=0)
    
        out = torch.matmul(softmax(torch.matmul(K_linear, Q_linear.transpose(2,3))/torch.sqrt(self.dim), dim=-1), V_linear)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.linear_out(out)
        
        return out
 

    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.multihead = MultiHeadAttention() 
        #self.conv1 = nn.Conv1d(self.dim_input, self.dim_input, kernel_size=1)
        #self.conv2 = nn.Conv1d(self.dim_input, self.dim_input, kernel_size=1)
        self.linear_1 = nn.Linear(self.dim, 2048)
        self.linear_2 = nn.Linear(2048, self.dim)

    def forward(self, x: torch.Tensor):
        out_multihead = self.multihead(x)
        out_multihead_add_norm = layer_norm(out_multihead + x)

        out_FFN = self.linear_1(out_multihead_add_norm)
        out_FFN = relu(out_FFN)
        out_FFN = self.linear_2(out_FFN)

        out = layer_norm(self.out_FFN + out_multihead_add_norm)

        return out
    
