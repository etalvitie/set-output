import os
from random import random, randint, seed, uniform
import math
from math import pi, cos, sin, floor, ceil
import glob

from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
torch.set_grad_enabled(True)

from squaredataclass import SquareDataset, sampledata
from simple_matcher import SimpleMatcher

"""
To run the logger, enter
    tensorboard --logdir=runs
"""


class Simple_Set_DETR(pl.LightningModule):
    """
    Set DETR implementation. Modified from DETR Demo.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, env_len, obj_len, out_set_size, hidden_dim=256, nheads=8,
                num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.out_set_size = out_set_size
        self.env_len = env_len
        self.obj_len = obj_len

        # We no longer need the CNN preprocessing
        # However, we need an embedding layer
        self.obj_embed = nn.Linear(obj_len, hidden_dim)
        self.env_embed = nn.Linear(env_len, hidden_dim)
        # self.input_set_embed = nn.Parameter(torch.rand(set_size, hidden_dim))

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_attri = nn.Linear(hidden_dim, obj_len + 1)
        self.linear_pos = nn.Linear(hidden_dim, 2)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # Loss calculation
        self.matcher = SimpleMatcher()
        self.loss_criterion = nn.MSELoss()

        # Logger
        self.writer = SummaryWriter()


    def forward(self, x):
        """
        Input size: [BATCH_SIZE * (2M+1)]
        """
        # batch size
        bs = x.shape[0]

        # Calculate the environment embedding
        env = x[:, 0]
        h_env = self.env_embed(env)
        h_env = h_env.unsqueeze(1).transpose(0, 1)
        h_env = h_env.repeat((1, self.out_set_size, 1))       # Shape: [BS, 4, HIDDEN_DIM]
        # print(h_env.shape)

        # Calculate the object embedding
        objs = x[:, 1:None].reshape(bs, -1, self.obj_len)
        h_objs = self.obj_embed(objs)
        # print(h_objs.shape)                                   # Shape: [BS, 4, HIDDEN_DIM]

        # Feed to the transformer
        h = self.transformer(h_env, h_objs)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_attri': self.linear_attri(h), 
                'pred_pos': self.linear_pos(h).sigmoid()}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        # Calculate the prediction
        x, y = train_batch
        pred = self.forward(x)

        # Calculate the loss
        indices, y_matched = self.matcher(pred, y)
        loss = self.loss_criterion(pred['pred_pos'].view(-1), y_matched.view(-1))
        
        self.log('train_loss', loss)
        self.writer.add_scalar('Loss/train', loss.item(), batch_idx)
        return loss
    

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        x, y = val_batch
        pred = self.forward(x)

        # Calculate the loss
        indices, y_matched = self.matcher(pred, y)
        loss = self.loss_criterion(pred['pred_pos'].view(-1), y_matched.view(-1))
        self.log('val_loss', loss)
        self.writer.add_scalar('Loss/test', loss.item(), batch_idx)


def train_pl():
    """
    Tests
    """
    # Prepare the dataset
    dataset = SquareDataset(1000, generator_type="linear")
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, dataset_size-train_size])
    train_data_loader = DataLoader(train_set, batch_size=1, num_workers=8, pin_memory=True)
    val_data_loader = DataLoader(val_set, batch_size=1, num_workers=8, pin_memory=True)

    # Initialize the model
    model = Simple_Set_DETR(1, 2, 4, hidden_dim=256, nheads=4, num_encoder_layers=3, num_decoder_layers=3)

    # Train
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=10, check_val_every_n_epoch=5)
    trainer.fit(model, train_data_loader, val_data_loader)


def evaluate(path=None):
    # load model
    if path is None:
        list_ckpts = glob.glob(os.path.join("lightning_logs","version_0","checkpoints","*.ckpt"))
        latest_ckpt = max(list_ckpts, key=os.path.getctime)
        print("Using checkpoint ", latest_ckpt)
        path = latest_ckpt

    model = pl.LightningModule.load_from_checkpoint(path)
    model.freeze()

    # Evaluate
    dataset = SquareDataset(5, generator_type="linear")
    eval_data_loader = DataLoader(dataset, batch_size=1)
    model.test(test_dataloaders=eval_data_loader)
    for batch, (x, y) in enumerate(eval_data_loader):
        x, y = x, y
        pred = model(x)
        print("Start")
        print(x[0, 1:None].reshape(-1,2))
        print("GT")
        print(y)
        print("Prediction")
        print(pred['pred_pos'])
        print()


if __name__ == "__main__":
    train_pl()
    evaluate()