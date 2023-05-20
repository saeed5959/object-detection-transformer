import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from einops import rearrange

from core.settings import model_config

class DatasetObjectDetection(Dataset):
    def __init__(self, dataset_file_path, transform):
        super().__init__()
        self.model_config = model_config
        with open(dataset_file_path) as file:
            self.dataset_file = file.readlines()
        self.transform = transform
        
    def get_image(self, data: str, augment: bool):
        #split data with |
        data_list = data.split("|")

        #read image and convert to tensor
        img_path = data_list[0]
        img = cv2.imread(img_path) / 255
        img = rearrange(img, 'h w c -> c h w')
        img = torch.Tensor(img)

        #read bounding box
        data_list = data_list[1:]
        bbox = []
        class_id = []
        class_mask = []
        for patch in data_list:
            c, x, y, w, h = patch.split(",")
            c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)            
            bbox.append([x, y, w, h])
            #label smoothing
            label_smoothing = np.abs(np.random.normal(0,0.05))
            patch_class = np.eye(model_config.class_num)[c]
            patch_class[c] = patch_class[c] - label_smoothing 
            class_id.append(patch_class)
            class_mask.append([c])

        bbox = torch.Tensor(np.array(bbox))
        class_id = torch.Tensor(np.array(class_id))
        class_mask = torch.Tensor(np.array(class_mask))

        #augmentation
        if augment:
            img, class_id, bbox, class_mask = self.transform(img, class_id, bbox, class_mask)

        return img, class_id, bbox, class_mask
    
    def __getitem__(self,index):
        if self.model_config.augmentation:
            if index//self.model_config.augment_num == 0:
                return self.get_image(self.dataset_file[index], augment=False)
            else:    
                return self.get_image(self.dataset_file[index], augment=True)
        else:
            return self.get_image(self.dataset_file[index], augment=False)
    
    def __len__(self):
        return len(self.dataset_file)*self.model_config.augment_num
    

def augmentation(img, class_id, bbox, class_mask):

    #shift : yes

    #scale : yes

    #rotation : yes

    #occlusion : cutout : yes

    #label smoothing : yes : done

    #normalize value of image form 255 to 1 : yes : done

    #dropout : no

    #mosaic : no

    #cutmix : no

    #patch can contain more than one object then the label can be [0.6 0.4] instead of [1 0]

    #object can be so tiny that is inside a patch but IOU is less than 0.5

    #I am not sure mask loss for background bbox

    #write inference mode

    return img, class_id, bbox, class_mask