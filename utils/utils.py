from PIL import Image

import glob
import os
import torch
import torch.nn as nn
import numpy as np
import argparse

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


def transform_frames(frames, device, stack=False):
    if stack:
        return transform_grayscale(frames, device)
    else:
        return transform_colored(frames, device)


def transform_colored(frames, device):
    # only get the first frame
    frame = frames[-1]
    frame = frame[20:210, 9:312]  # crop
    frame = resize_image(frame.astype(np.uint8), (125, 75))
    frame = frame / 255
    frame = frame - frame.mean()
    frame = frame.transpose(2, 0, 1)
    return torch.FloatTensor(frame).to(device).unsqueeze(0)

def transform_grayscale(frames, device):
    # get all frame and make gray scale
    x = []
    for frame in frames:
        frame = frame[20:210, 9:312]  # crop
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
        frame = frame[::3, ::3]  # downsample
        frame = frame / 255
        frame = frame - frame.mean()
        x.append(torch.FloatTensor(frame.reshape(1, 61, 120)).to(device))
    return torch.stack(x, dim=1)