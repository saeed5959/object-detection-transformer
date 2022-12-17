from core import settings
import torch
from voice_cloner.text.symbols import symbols
from voice_cloner import models
from voice_cloner import utils


class LoadModel:
    '''
      to  keep the model and pretrained weights loaded on the cpu
    '''
    def __init__(self):
        hps = settings.CONFIG_BASE
        torch.manual_seed(hps.train.seed)
        try: 
            self.net_g = models.SynthesizerTrn(
                len(symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model.__dict__)

        except Exception as e: 
            settings.LOGGER.info('EXCEPTION ERROR LOAD_MODEL: '+ str(e))
       
        
        try: 
            self.net_g_no_weight = models.SynthesizerTrn(
                len(symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model.__dict__)

        except Exception as e: 
            settings.LOGGER.info('EXCEPTION ERROR LOAD_MODEL_NO_WEIGHT: '+ str(e))       
        
        self.net_d = models.MultiPeriodDiscriminator(hps.model.use_spectral_norm)
        self.optim_g = torch.optim.AdamW(
        self.net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
        self.optim_d = torch.optim.AdamW(
        self.net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
        model_G_path = settings.DEFAULT_PRE_TRAINED_PATH_G
        _, _, _, _ = utils.load_checkpoint(model_G_path, self.net_g)
        model_D_path = settings.DEFAULT_PRE_TRAINED_PATH_D
        _, _, _, _ = utils.load_checkpoint(model_D_path, self.net_d, self.optim_d)
    
    def return_net_g(self):
        net_g = self.net_g
        return net_g
    def return_net_d(self):
        net_d = self.net_d
        return net_d

    def return_optim_g(self):
        optim_g = self.optim_g
        return optim_g
    def return_optim_d(self):
        optim_d = self.optim_d
        return optim_d

    def return_net_g_no_weight(self):
        net_g = self.net_g_no_weight
        return net_g





