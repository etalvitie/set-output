from src.dspn.DSPN import hungarian_loss
import scipy.optimize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from src.dspn.FSPool import FSPool
import pytorch_lightning as pl

class FSEncoder(pl.LightningModule):
    ##Set encoder from the DSPN/FSencoder papers
    def __init__(self, input_channels, output_channels, dim,set_dim, mask = False):
        super().__init__()
        self.set_dim = set_dim
        n_out = 30
        if mask == True:
            self.mask = 1
        else:
            self.mask = 0
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + self.mask, dim, 1),
           ## nn.ReLU(),
           ## nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels, 1),
        )
        self.pool = FSPool(output_channels, n_out, relaxed=False)

    def forward(self, x, mask=None):
        ##mask = mask.unsqueeze(1)
        ##x = torch.cat([x, mask], dim=1)  # include mask as part of set
        h = x.reshape((self.set_dim[1],-1)).unsqueeze(2).permute(2,0,1)

        h = self.conv(h)

        h = h / h.size(2)  # normalise so that activations aren't too high with big sets
        h, _ = self.pool(h)
        return h