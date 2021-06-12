from src.dspn.DSPN_copy import hungarian_loss
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
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + 1, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels, 1),
        )
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        return x