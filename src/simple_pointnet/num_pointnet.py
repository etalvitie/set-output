import glob
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

torch.set_grad_enabled(True)

class NumPointnet(pl.LightningModule):
    """
    A naive implementation of pointnet.
    """

    def __init__(self,
                 env_len=1,
                 obj_in_len=2,
                 out_len=1,
                 variance_type="separate"):
        """
        Available variance convergence types: ["separate", "hetero"]
        """
        super().__init__()
        self.save_hyperparameters()

        self.obj_in_len = obj_in_len
        self.env_len = env_len
        self.out_len = out_len

        variance_types = ["separate", "hetero"]
        if variance_type not in variance_types:
            raise ValueError("Invalid variance type. Expected one of: %s" % variance_types)
        self.variance_type = variance_type

        self.dropout = nn.Dropout(p=0.1)

        # Embedding layers
        self.obj_embed = nn.Sequential(
            nn.Linear(obj_in_len, 64),
            nn.ReLU(),
        )
        self.obj_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            self.dropout,
            nn.ReLU()
        )

        self.env_embed = nn.Sequential(
            nn.Linear(env_len, 64),
            nn.ReLU()
        )

        # Output heads

        self.decoder = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_len * 2),
            nn.ReLU(),
            self.dropout,
        )
        self.relu = nn.ReLU()

        # Output masks
        self.mask_softmax = nn.Softmax(dim=2)

        # Loss + matching calculation
        self.loss_criterion = nn.MSELoss()

    def forward(self, s, a, debug=False):
        """
        Input size: [BATCH_SIZE, N, M]
        """
        # batch size
        bs = a.shape[0]

        # Calculate the object embedding
        emb_objs = self.obj_embed(s)  # Shape: [BS, N, 64]
        emb_objs_vector, _ = torch.max(emb_objs, dim=1)
        in_set_size = emb_objs.shape[1]

        # Calculate the environment embedding
        env = a
        h_env_vector = self.env_embed(env)

        if debug:
            print("env")
            print(env)
            print(h_env_vector)
        # print(h_env.shape)

        # Obtain the set information by taking the maximum
        h_objs = self.obj_encoder(emb_objs)
        h_set_vector, _ = torch.max(h_objs, dim=1)  # Shape: [BS, HIDDEN_DIM]
        # print(h_objs.shape)                                   # Shape: [BS, N, HIDDEN_DIM]

        if debug:
            print("set")
            print(h_set_vector)

        if debug:
            print("objects")
            print(h_objs)

        # Concat the three matrix
        h = torch.cat((emb_objs_vector, h_set_vector, h_env_vector), dim=1)  # Shape: [BS, M, 3*HIDDEN_DIM]
        # h_global = torch.cat((h_env_vector, h_set_vector), dim=1)   # Shape: [BS, 2*HIDDEN_DIM]

        pred = self.decoder(h)
        pred_val = pred[:, 0:self.out_len]
        pred_var = pred[:, self.out_len:2*self.out_len]
        pred_var = pred_var ** 2 + 1e-6  # Prevent zero covariance causing errors

        # pred_pos = objs + pred_delta
        if debug:
            print("pred")
            print(pred)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_val': pred_val,
                'pred_var': pred_var}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_fn(self, batch):
        s, a, sprime, sappear, r = batch

        pred_result = self.forward(s, a)
        pred_val = pred_result['pred_val']
        pred_var = pred_result['pred_var']

        # Separate training
        if self.variance_type == "separate":
            loss_val = self.loss_criterion(pred_val, r)
            loss_var = self.loss_criterion((pred_val - r)**2, pred_var)
            loss = loss_val + loss_var

        # Heteroscedastic
        if self.variance_type == "hetero":
            loss = torch.mean((pred_val - r)**2 / (2*pred_var) + 0.5*torch.log(pred_var))

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.loss_fn(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.loss_fn(val_batch)
        self.log('train_loss', loss)

