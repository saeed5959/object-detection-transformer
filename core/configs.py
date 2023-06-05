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
        self.height: int = 256
        self.width: int = 256
        self.dim: int = 384
        self.head_num: int = 2
        self.patch_size: int = 16
        self.patch_num: int = 256
        self.patch_num_h: int = 16
        self.class_num: int = 90 
        self.augmentation: bool = False
        self.augment_num: int = 1
        self.iou_thresh: float = 0.5
        self.poa_epoch: int = 15
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = 10
        self.epochs: int = 60
        self.batch_size: int = 32
        self.learning_rate: float = 0.0001
        self.step_show: int = 100
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
