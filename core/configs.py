"""
    All config is in here
"""


class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.dim_iput: int = 512
        self.num_head: int = 2
        self.num_divide: int = 15
        self.source: bool = False
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = self.train_dict.get("save_model", 100)
        self.epochs: int = self.train_dict.get("epochs", 500)
        self.batch_size: int = self.train_dict.get("batch_size",8)
        self.learning_rate: float = self.train_dict.get("learning_rate", 0.0002)
