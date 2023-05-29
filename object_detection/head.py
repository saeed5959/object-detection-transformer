import torch
from torch import nn
from torch.nn.functional import softmax

from core.settings import model_config

class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.patch_num = model_config.patch_num
        self.class_num = model_config.class_num
        self.linear_class_1 = nn.Linear(self.dim+self.patch_num, 256)
        self.linear_class_2 = nn.Linear(256, 1 + self.class_num + 4) 

    def forward(self, x):
        similarity_matrix = softmax(torch.matmul(x, x.transpose(1,2)), dim=-1)
        x_and_similarity = torch.cat((x, similarity_matrix), dim=-1)

        out = self.linear_class_1(x_and_similarity)
        out = self.linear_class_2(out)

        return out
