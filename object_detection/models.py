import torch
from torch import nn
from torch.nn.functional import sigmoid, softmax

from object_detection import linear_patch, transformer, head
from core.settings import model_config, train_config

device = train_config.device

class VitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = linear_patch.EfficientCNN()
        self.transformer_block = transformer.Transformer()
        self.head_block = head.HeadDetect()

    def forward(self, x, poa, epoch):
        linear_out = self.linear_block(x)
        transformer_out = self.transformer_block(linear_out)
        out, similarity_matrix = self.head_block(transformer_out, poa, epoch)

        return out, similarity_matrix
    
    def inference(self, x, poa, epoch):
        #model output
        linear_out = self.linear_block(x)
        transformer_out = self.transformer_block(linear_out)
        out, similarity_matrix = self.head_block(transformer_out, poa, epoch)

        #sigmoid for object
        obj_out = sigmoid(out[:,:,0])
        #softmax for class
        class_out = softmax(out[:,:,1:model_config.class_num+1], dim=-1)
        #bound [0,1] for bbox
        box_out = out[:,:,model_config.class_num+1:]
        box_out = torch.minimum(torch.tensor([1]).to(device), torch.maximum(torch.tensor([0]).to(device), box_out))

        return obj_out, class_out, box_out

