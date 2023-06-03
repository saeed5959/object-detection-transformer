import torch
from torch import nn
from torch.nn.functional import softmax, relu

from core.settings import model_config, train_config

device = train_config.device

class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.patch_num = model_config.patch_num
        self.class_num = model_config.class_num
        self.linear_class_1 = nn.Linear(self.dim+self.patch_num, 256)
        self.linear_class_2 = nn.Linear(256, 1 + self.class_num + 4) 

    def forward(self, x):
        #similarity_matrix = softmax(torch.matmul(x, x.transpose(1,2)), dim=-1)
        similarity_matrix = torch.matmul(x, x.transpose(1,2)) / self.patch_num
        #putting between (0,1)
        similarity_matrix = torch.minimum(torch.Tensor([1]).to(device), torch.maximum(torch.Tensor([0]).to(device), similarity_matrix))
        x_and_similarity = torch.cat((x, similarity_matrix), dim=-1)

        out = self.linear_class_1(x_and_similarity)
        out = self.linear_class_2(out)

        return out, similarity_matrix
