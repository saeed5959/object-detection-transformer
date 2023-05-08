import torch
from torch.utils.data import Dataset
import cv2

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
        bounding_box = []
        for patch in data_list:
            c, x, y, w, h = patch.split(",")
            c, x, y, w, h = int(c), int(x), int(y), int(w), int(h)            
            bounding_box.append([c, x, y, w, h])

        bounding_box = torch.Tensor(bounding_box)
        
        return img, bounding_box
    
    def __getitem__(self,index):
        return self.get_image(self.dataset_file[index])
    
    def __len__(self):
        return len(self.dataset_file)