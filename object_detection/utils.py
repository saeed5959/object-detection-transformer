import torch
import numpy as np

def make_mask(length, max_length):
    
    m_zero = torch.zeros(max_length)
    mask = length > m_zero
    return mask
    
#not complete
def mono_alighn(u, sigma, y_flow):
    
    x_guas = np.random.normal(loc=u, scale=sigma, size=np.shape(u))
    
    #finding best alighnment between x_guas and y_flow
    x_dur_mono = []
    
    return x_dur_mono