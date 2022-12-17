"""
    All settings
"""
from pathlib import Path
from decouple import AutoConfig
import json

from core import configs


BASE_DIR = Path(__file__).resolve().parent.parent
config = AutoConfig(search_path = BASE_DIR)

APP_VERSION = "1.0.0"
DEBUG = config("DEBUG", cast = bool, default = False)

DEFAULT_PRE_TRAINED_PATH_G = config("DEFAULT_PRE_TRAINED_PATH_G")
DEFAULT_PRE_TRAINED_PATH_D = config("DEFAULT_PRE_TRAINED_PATH_D")

DENOISER_CODE_PATH = config("DENOISER_CODE_PATH")
TEXT_TO_SPEECH_PRE_TRAINED_PATH = config("TEXT_TO_SPEECH_PRE_TRAINED_PATH")
PUNC_PRE_TRAINED_PATH = config("PUNC_PRE_TRAINED_PATH")
AUTO_PUNC = eval(config("AUTO_PUNC"))


MODEL_LOCAL = config("MODEL_LOCAL")
VOICE_LOCAL = config("VOICE_LOCAL")
MODEL_UPLOAD = config("MODEL_UPLOAD")
VOICE_UPLOAD = config("VOICE_UPLOAD")
VOICES_MIX_PATH = json.loads(config("VOICES_MIX_PATH"))

N_STT_CORE = config("STT_N_CORE", default = 3)

BASE_CONFIG_PATH = config("BASE_CONFIG_PATH")
CONFIG_BASE = configs.BaseConfig(BASE_CONFIG_PATH)
TTS_ID_SPEAKER= configs.TTSIdSpeaker(BASE_CONFIG_PATH)

LOGGER_PATH = config("LOGGER_PATH", default = "logger.txt")
LOGGER = configs.Logger(LOGGER_PATH)

HASH=config("HASH")
ENDPOINT = config("ENDPOINT")
ACCESS_ID = config("ACCESS_ID")
ACCESS_KEY = config("ACCESS_KEY")
BUCKET = config("BUCKET")
DATASERVER = configs.DataServer(ENDPOINT, ACCESS_ID, ACCESS_KEY, BUCKET)

PHONEMIZER_API = config("PHONEMIZER_API")

VC_API = config("VC_API")
TTS_API = config("TTS_API")
N_TTS_VC_SERVICE = config("N_TTS_VC_SERVICE", default = 3)

KEY_TTS_VC_SERVICE = "aef3e989f18fc62396f00859db788836"

MAX_NUM_WORDS = config("MAX_NUM_WORDS", default = 20)
MIN_NUM_WORDS = config("MAX_NUM_WORDS", default = 5)
LEXICAN_PATH = config("LEXICAN_PATH", default = "/home/fluid/base/mfa_pretrained/dictionary/english_mfa.dict")
ACOUSTIC_PATH = config("ACOUSTIC_PATH", default ="/home/fluid/base/mfa_pretrained/english_mfa_acoustic")
CONDA_PATH = config("CONDA_PATH", default ="/home/fluid/anaconda3/etc/profile.d/conda.sh")
