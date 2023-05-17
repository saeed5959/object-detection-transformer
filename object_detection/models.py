import torch
from torch import nn

from object_detection import linear_patch, transformer, head

class VitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = linear_patch.LinearProjection()
        self.transformer_block = transformer.Transformer()
        self.head_block = head.HeadDetect()

    def forward(self, x):
        linear_out = self.linear_block(x)
        transformer_out = self.transformer_block(linear_out)
        class_out, box_out = self.head_block(transformer_out)

        return class_out, box_out

