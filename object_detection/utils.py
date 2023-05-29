import torch
import cv2
from einops import rearrange


def load_pretrained(model: object, pretrained: str, device: str):

    pretrained_model = torch.load(pretrained, map_location=device)
    model.load_state_dict(pretrained_model)

    return model

def img_preprocess_inference(img_path : str):
    img = cv2.imread(img_path) / 255
    img = cv2.resize(img,(256,256))

    img = rearrange(img, 'h w c -> c h w')
    img = torch.Tensor(img)
    img = img.unsqueeze(dim=0)

    return img

def nms_img():
    #non maxima supression 

    return