"""
    List of all models for fastapi
"""
from typing import List, Dict
from pydantic.main import BaseModel, Extra


class DeletefileRequest(BaseModel):
    """
       Request model for delete voice
    """
    file_path: str

class ValidateVoiceResponse(BaseModel):
    """
        Response model for validate voice
    """
    result: bool

class ValidateVoiceWithAddressRequest(BaseModel):
    """
        Request model for validate voice
    """
    voice_address: str 
    text: str 
    language_id: int 

class ValidateVoiceWithAddressResponse(BaseModel):
    """
        Response model for validate voice
    """
    result: bool

class DenoiserResponse(BaseModel):
    """
        Response model for denoiser
    """
    output_path: str

class DenoiserWithAddressRequest(BaseModel):
    """
        Response model for denoiser
    """
    voice_address: str 

class DenoiserWithAddressResponse(BaseModel):
    """
        Response model for denoiser
    """
    output_path: str

class MakeModelRequest(BaseModel):
    """
        Request model for make model
    """
    gender: int
    voices: List[Dict]
    model_path: str

class MakeModelResponse(BaseModel):
    """
        Response model for speech to text
    """
    key: str
    
class VoiceCloningRequest(BaseModel):
    """
        Request model for voice cloning
    """
    model_path: str 
    text: str 
    gender: int 
    speed: float

class VoiceCloningLongRequest(BaseModel):
    """
        Request model for voice cloning
    """
    model_path: str 
    text: str 
    gender: int 
    speed: float
    voice_path_idrive: str

class TextToSpeechRequest(BaseModel):
    """
        Request model for text to speech
    """
    text: str 
    id_speaker: int
    speed: float
    
class TextToSpeechLongRequest(BaseModel):
    """
        Request model for text to speech
    """
    text: str 
    id_speaker: int
    speed: float
    voice_path_idrive: str

class SpeechToTextResponse(BaseModel):
    """
        Response model for speech to text
    """
    key: str

class RedisResponse(BaseModel):
    """
        Response model for redis  
    """
    value: dict

class RedisRequest(BaseModel):
    """
        Request model for redis
    """
    key: str
    
class TXTToTextResponse(BaseModel):
    """
        Request model for pdf to text
    """
    text: str
    n_character: int


class WordToTextResponse(BaseModel):
    """
        Request model for pdf to text
    """
    text: str
    n_character: int
    
class PDFToTextResponse(BaseModel):
    """
        Request model for pdf to text ai
    """
    key: str

class VoiceCloningServiceRequest(BaseModel):
    """
        Request model for voice cloning
    """
    model_local: str 
    text: str 
    gender: int 
    speed: float
    output_voice: str
    cuda_num: str

class TextToSpeechServiceRequest(BaseModel):
    """
        Request model for text to speech
    """
    text: str 
    id_speaker: int
    speed: float
    output_path: str
    cuda_num: int
