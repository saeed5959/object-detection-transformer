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
        self.patch_num: int = (self.height * self.width) / (self.patch_size^2) 
        self.source: bool = False
        self.class_num: int = 5
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = self.train_dict.get("save_model", 100)
        self.epochs: int = self.train_dict.get("epochs", 500)
        self.batch_size: int = self.train_dict.get("batch_size",8)
        self.learning_rate: float = self.train_dict.get("learning_rate", 0.0002)
