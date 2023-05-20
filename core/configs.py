"""
    All config is in here
"""


class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.height: int = 256
        self.width: int = 256
        self.dim: int = 768
        self.head_num: int = 2
        self.patch_size: int = 16
        self.patch_num: int = 256
        self.patch_num_h: int = 16
        self.source: bool = False
        self.class_num: int = 90 + 1 # 1=background
        self.augmentation: bool = False
        self.augment_num: int = 1
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = 100
        self.epochs: int = 1000
        self.batch_size: int = 8
        self.learning_rate: float = 0.0001
