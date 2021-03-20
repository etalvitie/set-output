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
torch.set_grad_enabled(True)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from squaredataclass import SquareDataset, sampledata
from SimpleNumberDataset import SimpleNumberDataset
from simple_matcher import SimpleMatcher

"""
To run the logger, enter
    tensorboard --logdir=lightning_logs
"""


class Simple_PointNet(pl.LightningModule):
    """
    A naive implementation of pointnet.
    """
    def __init__(self, env_len=1, obj_len=2, out_set_size=4, hidden_dim=256):
        super().__init__()

        self.out_set_size = out_set_size
        self.env_len = env_len
        self.obj_len = obj_len

        # We no longer need the CNN preprocessing
        # However, we need an embedding layer
        self.obj_embed = nn.Sequential(
            nn.Linear(obj_len, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim)),
            nn.ReLU()
        )
        self.env_embed = nn.Sequential(
            nn.Linear(env_len, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim)),
            nn.ReLU()
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_attri = nn.Sequential(
            nn.Linear(3 * hidden_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), obj_len + 1),
        )
        self.linear_pos = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

        # Loss calculation
        self.matcher = SimpleMatcher()
        self.loss_criterion = nn.MSELoss()


    def forward(self, x, debug=False):
        """
        Input size: [BATCH_SIZE * (2M+1)]
        """
        # batch size
        bs = x.shape[0]

        # Calculate the environment embedding
        env = x[:, 0:self.env_len]
        h_env = self.env_embed(env)
        h_env_vector = h_env.unsqueeze(1).transpose(0, 1)
        h_env = h_env_vector.repeat((1, self.out_set_size, 1))  # Shape: [BS, M, HIDDEN_DIM]
        if debug:
            print("env")
            print(env)
            print(h_env_vector)
        # print(h_env.shape)

        # Calculate the object embedding
        objs = x[:, self.env_len:None].reshape(bs, -1, self.obj_len)
        h_objs = self.obj_embed(objs)
        # print(h_objs.shape)                                   # Shape: [BS, N, HIDDEN_DIM]

        # Obtain the set information by taking the maximum
        h_set_vector, _ = torch.max(h_objs, dim=1)              # Shape: [BS, HIDDEN_DIM]
        h_set = h_set_vector.repeat((1, self.out_set_size, 1))  # Shape: [BS, M, HIDDEN_DIM]

        # Zero-padding the object embedding
        pad_size = self.out_set_size - h_objs.shape[1]
        pad = nn.ZeroPad2d((0, 0, 0, pad_size))
        h_objs = pad(h_objs)                                    # Shape: [BS, M, HIDDEN_DIM]


        if debug:
            print("set")
            print(h_set_vector)

        if debug:
            print("objects")
            print(h_objs)

        # Concat the three matrix
        h = torch.cat((h_objs, h_set, h_env), dim=2)                # Shape: [BS, M, 3*HIDDEN_DIM]
        # h_global = torch.cat((h_env_vector, h_set_vector), dim=1)   # Shape: [BS, 2*HIDDEN_DIM]

        # Predict
        pred_delta = self.linear_pos(h)
        # pred_pos = objs + pred_delta
        if debug:
            print("pred_delta")
            print(pred_delta)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_attri': self.linear_attri(h), 
                'pred_pos': pred_delta}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        # Calculate the prediction
        x, y = train_batch
        pred = self.forward(x)

        # Calculate the loss
        # pred_new_order, y_matched = self.matcher(pred, y)
        y_matched = y
        # y_matched = y
        pred_matched = pred['pred_pos'].view(-1,2)
        loss = self.loss_criterion(pred_matched.view(-1), y_matched.view(-1))
        
        self.log('train_loss', loss)
        return loss
    

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        x, y = val_batch
        pred = self.forward(x)

        # Calculate the loss
        pred_new_order, y_matched = self.matcher(pred, y)
        # y_matched = y
        pred_matched = pred['pred_pos'].view(-1,2)[pred_new_order]
        loss = self.loss_criterion(pred_matched.view(-1), y_matched.view(-1))
        self.log('val_loss', loss)


def train_pl():
    """
    Tests
    """
    # Prepare the dataset
    # dataset = SquareDataset(1000, generator_type="linear")
    dataset = SquareDataset(10000, generator_type="rotation")
    # dataset = SimpleNumberDataset(10000, 10, 100, 10)
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, dataset_size-train_size])
    train_data_loader = DataLoader(train_set, batch_size=1, num_workers=8, pin_memory=True, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=1, num_workers=8, pin_memory=True)

    # Initialize the model
    model = Simple_PointNet(1, 2, 4, hidden_dim=400)

    # Early stop callback
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='min'
    # )

    # Train
    trainer = pl.Trainer(
        gpus=1, 
        precision=16,
        max_epochs=4,
        check_val_every_n_epoch=4,
        accumulate_grad_batches=50
        # callbacks=[early_stop_callback]
    )
    trainer.fit(model, train_data_loader, val_data_loader)

    # Evaluate
    # trainer.test(model, test_dataloaders = val_data_loader)
    evaluate(model=model)


def evaluate(model=None, path=None):
    # load model
    if model is None:
        if path is None:
            list_ckpts = glob.glob(os.path.join("lightning_logs","*","checkpoints","*.ckpt"))
            latest_ckpt = max(list_ckpts, key=os.path.getctime)
            print("Using checkpoint ", latest_ckpt)
            path = latest_ckpt

        model = Simple_PointNet.load_from_checkpoint(path)
        model.freeze()

    # Evaluate
    # dataset = SquareDataset(5, generator_type="linear")
    dataset = SquareDataset(5, generator_type="rotation")
    # dataset = SimpleNumberDataset(5, 10, 100, 10)
    matcher = SimpleMatcher()
    eval_data_loader = DataLoader(dataset, batch_size=1)
    for batch, (x, y) in enumerate(eval_data_loader):
        x, y = x, y
        pred = model(x, debug=True)
        pred_new_order, y_matched = matcher(pred, y)
        pred_matched = pred['pred_pos'].view(-1,2)[pred_new_order]
        print("Start")
        print(x[0, 1:None].reshape(-1,2))
        print("GT")        
        print(y_matched)
        print("Prediction")
        print(pred_matched)
        print("Pred Vel")
        print(pred['pred_pos'] - x[0, 1:None].reshape(-1))
        print()


if __name__ == "__main__":
    train_pl()
    # evaluate()