import glob
import os

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def resize_image(img, new_size):

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(new_size, resample=Image.NEAREST)
    img = np.array(pil_img)

    return img