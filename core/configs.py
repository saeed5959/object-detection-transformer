"""
    All config is in here
"""


class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.resblock: int = 1
        self.filter_channels: int = 768
        self.hidden_channels: int = 192
        self.upsample_rates: list = [8, 8, 2, 2]
        self.n_phon: int = 178
        self.embed_size: int = 128
        self.max_input: int = 192
        self.kernel_size:int = 3
        self.p_dropout: int = 0.1
        self.sample_ratye: int = 22050
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = self.train_dict.get("save_model", 100)
        self.epochs: int = self.train_dict.get("epochs", 500)
        self.batch_size: int = self.train_dict.get("batch_size",8)
        self.learning_rate: float = self.train_dict.get("learning_rate", 0.0002)
