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

def resize_image(img, new_size, resample=Image.NEAREST):

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(new_size, resample=resample)
    img = np.array(pil_img)

    return img


def transform_frames(frames, device, stack=False):
    if stack:
        return transform_grayscale(frames, device)
    else:
        return transform_colored(frames, device)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def transform_colored(frames, device):
    # only get the first frame
    frame = frames[-1]
    frame = frame[20:210, 12:312]  # crop
    # frame = resize_image(frame.astype(np.uint8), (125, 75))
    frame = frame / 255
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    return torch.FloatTensor(frame).to(device).unsqueeze(0)

def transform_grayscale(frames, device):
    # get all frame and make gray scale
    x = []
    for frame in frames:
        frame = frame[20:210, 12:312]  # crop
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
        # frame = resize_image(frame.astype(np.uint8), (125, 75))  # downsample
        # import pdb; pdb.set_trace()
        frame = frame / 255
        frame = frame - 0.458
        x.append(torch.FloatTensor(frame).to(device))
    return torch.stack(x, dim=0).unsqueeze(0)

def add_noise(frames, power, position, scale=0.2):
    frames_noised = torch.Tensor(np.random.normal(0, scale, frames.shape)).to(frames.device) + frames
    frames = torch.max(torch.min(frames_noised, frames + scale), frames - scale)
    power_noised = torch.Tensor(np.random.normal(0, scale, power.shape)).to(power.device) + power
    power = torch.max(torch.min(power_noised, power + scale), power - scale)
    position_noised = torch.Tensor(np.random.normal(0, scale, position.shape)).to(position.device) + position
    position = torch.max(torch.min(position_noised, position + scale), position - scale)
    return frames, power, position

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr