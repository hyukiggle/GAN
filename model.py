import argparse
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def block(in_channels, out_channels, normalize = True):
        layers = [nn.Linear(in_channels, out_channels)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_channels,0.8))
        layers.append(nn.LeakyReLU(0.2, inplace= True))
        
    self.model = nn.Sequential(
        *block(opt.latent_dim,128, normalize= True)
    )