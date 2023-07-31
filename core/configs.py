"""
    All config is in here
"""
import torch

class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.panoptic_file_path: str = "./dataset/dataset_file_out.txt"
        self.json_file_path_1: str = "./dataset/annotations/panoptic_train2017_main.json"
        self.json_file_path_2: str = "./dataset/annotations/panoptic_train2017.json"
        self.height: int = 480
        self.width: int = 480
        self.dim: int = 1280 # efficientnet =1280 and resnet = 2048
        self.head_num: int = 16
        self.patch_size: int = 32
        self.patch_num: int = 225
        self.patch_num_h: int = 15
        self.class_num: int = 90 
        self.augmentation: bool = False
        self.augment_num: int = 1
        self.iou_thresh: float = 0.1
        self.obj_thresh: float = 0.9
        self.class_thresh: float = 0.7
        self.poa_epoch: int = 15
        self.iou_thresh_dataset: float = 0.1
        self.transformer_num: int = 4
        self.droupout_rate: float = 0.1
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = 50
        self.epochs: int = 100
        self.batch_size: int = 8
        self.learning_rate: float = 0.0001
        self.lr_end: float = 0.00001
        self.step_show: int = 100
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
