import torch
from torch import nn
from torch.nn.functional import softmax, relu

from core.settings import model_config

class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.patch_num = model_config.patch_num
        self.class_num = model_config.class_num
        self.linear_class = nn.Linear(self.dim+self.patch_num, 1 + self.class_num + 4) 
        #self.linear_box = nn.Linear(self.dim+self.patch_num, 4)

    def forward(self, x):
        similarity_matrix = softmax(torch.matmul(x, x.transpose(1,2)), dim=-1)
        x_and_similarity = torch.cat((x, similarity_matrix), dim=-1)

        # class_out = softmax(self.linear_class(x_and_similarity), dim=-1)
        class_out = self.linear_class(x_and_similarity)
        #box_out = relu(self.linear_box(x_and_similarity))

        return class_out
