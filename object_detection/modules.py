import torch
from torch import nn
from torch.nn.functional import softmax

from core.settings import model_config
        
        
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_input = model_config.dim_iput
        self.linear_1 = nn.Linear(self.dim_input, self.dim_input)
        self.linear_2 = nn.Linear(self.dim_input, self.dim_input)
        self.linear_3 = nn.Linear(self.dim_input, self.dim_input)


    def forward(self, img_vector_in : torch.Tensor):

        ### Scaled Dot-Product Attention
        V = img_vector_in  #img_vector_in : [batch_size, patch_size, vector_dim]
        V_linear = self.linear_1(V)
        K = img_vector_in
        K_linear = self.linear_2(K)
        Q = img_vector_in
        Q_linear = self.linear_1(Q)

        img_vector_out = V_linear * softmax(((K_linear * Q_linear.transpose)/torch.sqrt(self.dim_input)), dim=2).transpose

        return img_vector_out


class MultiHeadAttention(nn.Module);
    def __init__(self):
        super().__init__()


    def forward(self, img_vector_in: torch.Tensor):
        return 

    
