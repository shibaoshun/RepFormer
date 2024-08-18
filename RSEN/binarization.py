import torch
import numpy as np
import torch.nn as nn

def apply_threshold(x):

    tau = torch.mean(x)
    y = torch.where(x > tau, torch.ones_like(x), torch.zeros_like(x))
    return y

class Binarization(nn.Module):
    def __init__(self):
        super(Binarization, self).__init__()
       
    def forward(self, x):

        ch_mask = apply_threshold(x)

        return ch_mask




