import torch
from torch import nn

from core.settings import model_config, train_config

device = train_config.device

class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.patch_num = model_config.patch_num
        self.class_num = model_config.class_num
        self.linear_class_1 = nn.Linear(self.dim + self.dim + self.patch_num, 256)
        self.linear_class_2 = nn.Linear(256, 1 + self.class_num + 4) 

    def forward(self, x):
        #similarity_matrix = softmax(torch.matmul(x, x.transpose(1,2)), dim=-1)
        similarity_matrix = torch.matmul(x, x.transpose(1,2)) / self.patch_num
        #putting between (0,1)
        similarity_matrix = torch.minimum(torch.tensor([1]), torch.maximum(torch.tensor([0]), similarity_matrix))

        #you can make 0 for the values between (0,0.5)

        b, m, n = similarity_matrix.size()
        
        average_patch_select = torch.matmul(similarity_matrix, x) / torch.sum(similarity_matrix, dim=-1).reshape(b,m,1).repeat(1,1,self.dim)

        x_and_similarity = torch.cat((x, average_patch_select, similarity_matrix), dim=-1)

        out = self.linear_class_1(x_and_similarity)
        out = self.linear_class_2(out)

        return out, similarity_matrix
