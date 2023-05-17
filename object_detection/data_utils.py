import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from core.settings import model_config

class DatasetObjectDetection(Dataset):
    def __init__(self, dataset_file_path):
        super().__init__()
        self.model_config = model_config
        with open(dataset_file_path) as file:
            self.dataset_file = file.readlines()

        
    def get_image(self, data):
        #split data with |
        data_list = data.split("|")

        #read image and convert to tensor
        img_path = data_list[0]
        img = cv2.imread(img_path)
        img = torch.Tensor(img)

        #read bounding box
        data_list = data_list[1:]
        bbox = []
        class_id = []
        class_mask = []
        for patch in data_list:
            c, x, y, w, h = patch.split(",")
            c, x, y, w, h = int(c), int(x), int(y), int(w), int(h)            
            bbox.append([x, y, w, h])
            class_id.append(np.eye(model_config.class_num)[c])
            class_mask.append([c])

        bbox = torch.Tensor(bbox)
        class_id = torch.Tensor(class_id)
        class_mask = torch.Tensor(class_mask)

        return img, class_id, bbox, class_mask
    
    def __getitem__(self,index):
        return self.get_image(self.dataset_file[index])
    
    def __len__(self):
        return len(self.dataset_file)