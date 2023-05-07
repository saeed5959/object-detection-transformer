import torch
from torch import nn

from core.settings import model_config


class HeadDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = model_config.dim
        self.class_num = model_config.class_num
        self.linear_class = nn.Linear(self.dim, self.class_num) 
        self.linear_box = nn.Linear(self.dim, 4)

    def forward(self, x):
        class_out = self.linear_class(x)
        box_out = self.linear_box(x)

        return class_out, box_out
