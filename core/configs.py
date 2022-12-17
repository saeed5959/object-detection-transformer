"""
    All config is in here
"""
import torch
from typing import Dict, List
import json
from fastapi import HTTPException
from datetime import datetime
import boto3
import os
import redis
import requests 
import hashlib
from vosk import Model
from denoiser.demucs import Demucs
import torch 
import hashlib
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


redis_db = redis.Redis(host="localhost", port=6379)

def read_file (address):
    if os.path.exists(address):
        f = open(address)
        data = json.loads(f.read())
        return data
    else:
        pass
    

class ValidateVoiceConfig:
    """"
        All validate voice config 
    """

    def __init__(self, config_file_address):
        self.file_data: dict = read_file(address = config_file_address) or {}
        self.validate_voice_dict: dict = self.file_data.get("validate_voice", {})
        self.validate_voice_thrsh: int = self.validate_voice_dict.get("validate_voice_thrsh", 0.6)
        
        
class DataConfig:
    """
        All data config
    """

    def __init__(self, config_file_address):
        self.file_data: dict = read_file(address = config_file_address) or {}
        self.data_dict: dict = self.file_data.get("data", {})
        self.mel_fmax = self.data_dict.get("mel_fmax", None)
        self.filter_length:int = self.data_dict.get("filter_length", 1024)
        self.mel_fmin: float = self.data_dict.get("mel_fmin", 0.0)
        self.add_blank: int = self.data_dict.get("add_blank", True)
        self.hop_length: int = self.data_dict.get("hop_length", 256)
        self.n_speakers: int = self.data_dict.get("n_speakers", 109)
        self.win_length: int = self.data_dict.get("win_length", 1024)
        self.max_wav_value: float = self.data_dict.get("max_wav_value", 32768.0)
        self.sampling_rate: int = self.data_dict.get("sampling_rate", 22050)
        self.n_mel_channels: int = self.data_dict.get("n_mel_channels", 80)



class ModelConfig:
    """
        All model config
    """

    def __init__(self, config_file_address):
        file_data: dict = read_file(address = config_file_address) or {}
        model_dict: dict = file_data.get("model", {})
        self.n_heads: int = model_dict.get("n_heads", 2)
        self.n_layers: int = model_dict.get("n_layers", 6)
        self.resblock: str = model_dict.get("resblock", "1")
        self.p_dropout: float = model_dict.get("p_dropout", 0.1)
        self.n_layers_q: int = model_dict.get("n_layers_q", 3)
        self.kernel_size: int = model_dict.get("kernel_size", 3)
        self.gin_channels: int = model_dict.get("gin_channels", 256)
        self.inter_channels: int = model_dict.get("inter_channels", 192)
        self.upsample_rates: list = model_dict.get("upsample_rates", [8, 8, 2, 2])
        self.filter_channels: int = model_dict.get("filter_channels", 768)
        self.hidden_channels: int = model_dict.get("hidden_channels", 192)
        self.use_spectral_norm: bool = model_dict.get("use_spectral_norm", False)
        self.resblock_kernel_sizes: list = model_dict.get("resblock_kernel_sizes", [3, 7, 11])
        self.upsample_kernel_sizes: list = model_dict.get("upsample_kernel_sizes", [16, 16, 4, 4])
        self.upsample_initial_channel: int = model_dict.get("upsample_initial_channel", 512)
        self.resblock_dilation_sizes: List[list] = model_dict.get("resblock_dilation_sizes",
                                                                      [
                                                                          [1, 3, 5],
                                                                          [1, 3, 5],
                                                                          [1, 3, 5]
                                                                      ])
        

                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self, config_file_address):
        self.file_data: dict = read_file(address = config_file_address) or {}
        self.train_dict: dict = self.file_data.get("train", {})
        self.save_model: int = self.train_dict.get("save_model", 500)
        self.eps: float = self.train_dict.get("eps", 1e-09)
        self.c_kl: float = self.train_dict.get("c_kl", 1.0)
        self.seed: int = self.train_dict.get("seed", 1234)
        self.betas: list = self.train_dict.get("betas", [0.8, 0.99])
        self.c_mel: int = self.train_dict.get("c_mel", 45)
        self.epochs: int = self.train_dict.get("epochs", 500)
        self.fp16_run: bool = self.train_dict.get("fp16_run", True)
        self.lr_decay: float = self.train_dict.get("lr_decay", 0.999875)
        self.batch_size: int = self.train_dict.get("batch_size", 4)
        self.segment_size: int = self.train_dict.get("segment_size", 8192)
        self.init_lr_ratio: int = self.train_dict.get("init_lr_ratio", 1)
        self.learning_rate: float = self.train_dict.get("learning_rate", 0.0002)
        self.warmup_epochs: int = self.train_dict.get("warmup_epochs", 0)

class IdSpeaker:
    
    def __init__(self,config_file_address):
        self.file_data = read_file(address = config_file_address) or {}
        self.id_speaker_dict: dict = self.file_data.get("id_speaker", {})
        self.id_speaker_mix_male = self.id_speaker_dict.get("id_speaker_mix_male", 71)
        self.id_speaker_mix_female = self.id_speaker_dict.get("id_speaker_mix_female", 67)
        self.id_speaker_male = self.id_speaker_dict.get("id_speaker_male", 29)
        self.id_speaker_female = self.id_speaker_dict.get("id_speaker_female", 57)
        
        
class BaseConfig:
    """
        All base config
    """

    def __init__(self, config_file_address):
        self.data = DataConfig(config_file_address)
        self.train = TrainConfig(config_file_address)
        self.model = ModelConfig(config_file_address)
        self.validate_voice  = ValidateVoiceConfig(config_file_address)
        self.id_speaker = IdSpeaker(config_file_address)
        
    
    
class LanguageModel:
    """
        Model of language
    """

    def __init__(self, language_data: dict):
        self.id = language_data.get("id")
        self.name = language_data.get("name")
        self.pre_trained_path = language_data.get("model_file")
        self.pre_trained_link = language_data.get("model_link")
        self.voice_cloning = language_data.get("voice_cloning")
        self.speech_to_text = language_data.get("speech_to_text")
        


class LanguageData:
    """
        List of all language will be here
    """

    def __init__(self, data_file_path, language_ids):
        self.file_data:list = read_file(address = data_file_path)
        self.languages: "Dict[int, LanguageModel]" = {}
        self.models = {}
        for language in self.file_data:
            self.languages[language["id"]] = LanguageModel(language)
        
        for language_id in language_ids:
            self.models[language_id] = Model(self.languages[language_id].pre_trained_path)
       

    def get_speech_to_text(self, language_id) -> "LanguageModel":
        """
            Get language for speech to text
        """
        language = self.languages.get(language_id)
        if language is None or not language.speech_to_text:
            raise HTTPException(status_code = 404, detail = f"Language id {language_id} not found")
        return language

    def get_voice_cloning(self, language_id) -> "LanguageModel":
        """
            Get language for voice cloning
        """
        language = self.languages.get(language_id)
        if language is None or not language.voice_cloning:
            raise HTTPException(status_code = 404, detail = f"Language id {language_id} not found")
        return language
   
    def get_language_model(self, language_id):
        model = self.models[language_id]
        return model


class TTSIdSpeaker:

    def __init__(self,config_file_address):
        self.file_data:dict = read_file(address = config_file_address) or {}
        self.tts_id_speaker:list = self.file_data.get("tts_id_speaker",[102, 99, 96, 93, 91, 83, 82, 77,
                                                                        68, 67, 65, 63, 60, 57, 49, 37, 32,
                                                                        22, 19, 18, 16, 11, 10, 9, 6, 4])



class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path
        

    def info(self, message):
        with open(self.logger_path, "a") as file:
            now = datetime.now().isoformat(timespec ='seconds', sep =' ')
            file.write(f"{now} : INFO : {message}\n")
            
    
    def error(self, message):
        with open(self.logger_path, "a") as file:
            now = datetime.now().isoformat(timespec = 'seconds', sep = ' ')
            file.write(f"{now} : ERROR : {message}\n")



class DataServer:
    """
        Connections data server upload, download, delete and get list objects
    """
    def __init__(self, endpoint, access_id, access_key, bucket):    
        self.client = boto3.client("s3", endpoint_url = endpoint, aws_access_key_id = access_id, aws_secret_access_key = access_key)
        self.bucket = bucket
        
    def upload(self, path_local, path_dataserver):
        self.client.upload_file(path_local, self.bucket, path_dataserver)
    
    def download(self, path_local, path_dataserver):
        folder_local = os.path.dirname(path_local)
        os.makedirs(folder_local, exist_ok = True)
        self.client.download_file(self.bucket, path_dataserver, path_local)
        
    def delete_file(self, dataserver):
        self.client.delete_object(Bucket = self.bucket, Key = dataserver)
    
    def list_objects(self, path_folder):
        list_path_folder = []
        for obj in self.client.list_objects_v2(Bucket = self.bucket, Prefix = path_folder)['Contents']:
            list_path_folder.append(obj['Key'])
            
        return list_path_folder

    def check_path(self, file_path):
        file_directory = os.path.dirname(file_path)
        files = self.list_objects(file_directory)
        if file_path in files:
            return True
        return False
            

  
def check_connection(url):
    try:
        requests.get(url, verify=True)
        return True
    except:
        return False

    
class LoadDenoiser:
    def __init__(self, pretrained_path):
        self.model = Demucs(hidden=64)
        state_dict = torch.load(pretrained_path,map_location='cpu')
        self.model.load_state_dict(state_dict)

    def load_pretrained(self):
        model = self.model
        return model


class LoadSTTFacebook:
    def __init__(self,pretrained_path):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_path)
        
    def load_pretrained(self):
        return self.model,self.tokenizer

def TTS_VC_key():
    key_TTS_VC_service = hashlib.md5(os.urandom(32)).hexdigest()
    n_TTS_VC_service = 0
    redis_db.set(key_TTS_VC_service, json.dumps(n_TTS_VC_service))



# multi gpu selection
def gpu_states_list():
    device_count = torch.cuda.device_count()
    
    gpu_states = [0] * device_count
    redis_key_gpu_states = hashlib.md5(os.urandom(32)).hexdigest()
    redis_db.set(redis_key_gpu_states, json.dumps(gpu_states))
        
    return redis_key_gpu_states

def gpu_selection(redis_key_gpu_states:str):
    
    gpu_states = json.loads(redis_db.get(redis_key_gpu_states))
    gpu_states_min = min(gpu_states)
    
    for gpu_num in range(len(gpu_states)):
        if gpu_states[gpu_num]==gpu_states_min:
            gpu_states[gpu_num] += 1
            redis_db.set(redis_key_gpu_states,json.dumps(gpu_states))
            gpu_num_select = gpu_num
            break
        
    return gpu_num_select

def gpu_state_reduction(redis_key_gpu_states:str, gpu_num: int):
    gpu_states = json.loads(redis_db.get(redis_key_gpu_states))
    gpu_states[gpu_num] -= 1
    redis_db.set(redis_key_gpu_states,json.dumps(gpu_states))