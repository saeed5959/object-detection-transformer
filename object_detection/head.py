import torch
from torch import nn
from torch.nn.functional import relu
from core.settings import model_config, train_config

device = train_config.device

class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.patch_num = model_config.patch_num
        self.class_num = model_config.class_num
        self.poa_epoch = model_config.poa_epoch
        self.linear_similarity = nn.Linear(self.dim,self.dim)
        self.linear_class_1 = nn.Linear(self.dim + self.dim + self.patch_num, 512)
        self.linear_class_2 = nn.Linear(512, 1 + self.class_num + 4) 

    def forward(self, x, poa, epoch):
        x_linear = self.linear_similarity(x)
        #bound between (0,1)
        similarity_matrix = torch.min(torch.tensor([1]), relu(torch.matmul(x_linear, x_linear.transpose(1,2)) / self.patch_num))
        
        if epoch > self.poa_epoch:
            similarity_matrix_main = similarity_matrix
        else:
            similarity_matrix_main = poa

        b, m, n = similarity_matrix_main.size()
        
        average_patch_select = torch.matmul(similarity_matrix_main, x) / torch.sum(similarity_matrix_main, dim=-1).reshape(b,m,1).repeat(1,1,self.dim)

        x_and_similarity = torch.cat((x, average_patch_select, similarity_matrix_main), dim=-1)

        out = self.linear_class_1(x_and_similarity)
        out = self.linear_class_2(out)

        return out, similarity_matrix
