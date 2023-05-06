import torch
from torch.utils.data import Dataset

from core.settings import model_config

class DatasetTTS(Dataset):
    def __init__(self):
        super().__init__()
        self.config_model = model_config
        
        
    def __getitem__(self,index):
        return self.get_voice_text(self.data_file[index])
    
    def __len__(self):
        return len(self.data_file)