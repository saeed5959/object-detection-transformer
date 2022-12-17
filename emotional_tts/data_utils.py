import torch
import librosa
from torch.utils.data import Dataset

from core.settings import model_config

class DatasetTTS(Dataset):
    def __init__(self,training_file):
        super().__init__()
        self.config_model = model_config
        
        with open(training_file) as file:
            train_f = file.readlines()
        
        self.data_file = []
        for line in train_f:
            voice_path, text, speaker_id, emotion_id = line.split("|")
            self.data_file.append([voice_path, text, speaker_id, emotion_id])
            
    def get_voice_text(self,data:list):
        voice_path, text, speaker_id, emotion_id = data[0], data[1], data[2], data[3]
        voice = librosa.load(voice_path,self.config_model.sample_rate)
        
        voice = torch.Tensor(voice)
        text = torch.Tensor(text)
        speaker_id = torch.Tensor(speaker_id)
        emotion_id = torch.Tensor(emotion_id)
        
        return voice, text, speaker_id, emotion_id
        
    def __getitem__(self,index):
        return self.get_voice_text(self.data_file[index])
    
    def __len__(self):
        return len(self.data_file)