from src.dspn.DSPN_copy import hungarian_loss
from src.dspn.FSPool import FSPool

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

import pytorch_lightning as pl

class SetEncoder(pl.LightningModule):
    def __init__(self, env_len=6, obj_in_len=9, env_hidden_dim=64, obj_hidden_dim=512):
        super().__init__()
        self.save_hyperparameters()

        self.env_hidden_dim = env_hidden_dim
        self.obj_hidden_dim = obj_hidden_dim
        self.obj_in_len = obj_in_len
        self.env_len = env_len

        self.dropout = nn.Dropout(p=0.1)

        self.obj_embed = nn.Sequential(
            nn.Linear(obj_in_len, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, obj_hidden_dim),
            self.dropout,
            nn.ReLU()
        )

        if env_len == 0:
            hidden_dim = obj_hidden_dim
        else:
            self.env_embed = nn.Sequential(
                nn.Linear(env_len, env_hidden_dim),
                nn.ReLU()
            )
            hidden_dim = obj_hidden_dim + env_hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, objs, env=None):
        # Calculate the set embedding
        objs = objs.view((1, -1, self.obj_in_len))
        h_objs = self.obj_embed(objs)  # Shape: [BS, N, 64]
        h_set_vector, _ = torch.max(h_objs, dim=1)

        # Calculate environment embedding
        if env is not None:
            h_env_vector = self.env_embed(env)

            # Concatenate
            h = torch.cat((h_set_vector, h_env_vector), dim=1)
        else:
            h = h_set_vector

        h = self.encoder(h)
        return h