import torch


def make_mask(length, max_length):
    
    m_zero = torch.zeros(max_length)
    mask = length > m_zero
    return mask
    
    