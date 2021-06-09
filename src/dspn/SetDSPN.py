from src.dspn.FSEncoder import FSEncoder
from src.dspn.DSPN import deepsetnet, hungarian_loss
from src.dspn.SetEncoder import SetEncoder
from src.set_utils.variance_matcher import VarianceMatcher

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize
import torch.nn.functional as F
import random
import math
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset

class SetDSPN(pl.LightningModule):
    '''
    Parameters: encoder1: the set encoder
    encoder2: the encoder that gets passed into the DSPN
    latent_dim: the output dimension of encoder1
    set_dim: (number of objects, length of each object)
    n_iters: n_iters parameter for DSPN
    masks: for DSPN, doesn't actually do anything
    '''

    def __init__(self, set_encoder=None, dspn_encoder=None, obj_in_len=9, obj_reg_len=2, obj_attri_len=2, env_len=6, latent_dim=64, out_set_size=5, n_iters=10, masks=False):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.obj_in_len = obj_in_len
        self.obj_reg_len = obj_reg_len
        self.obj_attri_len = obj_attri_len

        obj_out_len =  obj_reg_len*2 + obj_attri_len*2
        out_set_dim = (out_set_size, obj_out_len)
        self.out_set_dim = out_set_dim

        if set_encoder is None:
            set_encoder = SetEncoder(env_len=env_len, obj_in_len=obj_in_len)
        self.set_encoder = set_encoder

        if dspn_encoder is None:
            # dspn_encoder = FSEncoder(out_obj_len, 1088, 1088, out_set_dim)
            dspn_encoder = SetEncoder(env_len=0, obj_in_len=obj_out_len, obj_hidden_dim=1088)
        self.decoder = deepsetnet(dspn_encoder, latent_dim, out_set_dim, n_iters, masks)

        # Output masks
        self.mask_softmax = nn.Softmax(dim=2)

        self.matcher = VarianceMatcher()
        # Loss + matching calculation
        self.matcher = VarianceMatcher()
        self.loss_reg_criterion = nn.MSELoss()
        self.loss_mask_criterion = nn.CrossEntropyLoss()

        self.loss_reg_weight = 1
        self.loss_mask_weight = 1

    def forward(self, x, a=None):
        # z = self.encoder(x)
        # z = z[0]
        # if action_vec is not None:
        #     action_vec = torch.cat(
        #         (action_vec, torch.Tensor([0 for i in range(self.latent_dim - action_vec.size()[0])])), 0)
        #     z = z + action_vec
        h = self.set_encoder(x, a)

        pred = self.decoder(h)
        pred = pred.reshape(self.out_set_dim)
        pred = pred.unsqueeze(0)

        # Regressions
        pred_reg = pred[:, :, 0:self.obj_reg_len]
        pred_reg_var = pred[:, :, self.obj_reg_len:2*self.obj_reg_len]
        pred_reg_var = pred_reg_var ** 2 + 1e-6

        # Attributes
        pred_attri = pred[:, :, 2*self.obj_reg_len:None]
        pred_mask = pred_attri[:, :, 0:2]
        pred_mask = self.mask_softmax(pred_mask)

        return {'pred_attri': pred_attri,
                'pred_mask': pred_mask,
                'pred_reg': pred_reg,
                'pred_reg_var': pred_reg_var}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def loss_fn(self, pred, gt_label):
        # If the output set is empty
        if gt_label.shape[1] == 0:
            pred_mask = pred['pred_mask'][0]
            tgt_mask = torch.zeros(pred_mask.shape[0], device=self.device, dtype=torch.long)
            loss_mask = self.loss_mask_criterion(pred_mask, tgt_mask)
            return loss_mask, 0, 0, loss_mask
        # Perform the matching
        else:
            gt_label = gt_label[:, :, 0:2]
            match_results = self.matcher(pred, gt_label)

            # Calculate the loss for regression and masking seperately
            loss_reg = 0
            loss_reg_var = 0
            loss_mask = 0

            # Iterate through all batches
            for batch_idx, match_result in enumerate(match_results):
                pred_raw = pred["pred_reg"][batch_idx]
                gt_raw = gt_label[batch_idx]
                pred_var_raw = pred["pred_reg_var"][batch_idx]
                pred_var_reordered = match_result["pred_var_reordered"]
                pred_matched = match_result['pred_reordered']
                gt_matched = match_result['gt_reordered']

                # Loss for regression (position)
                # loss_reg += self.loss_reg_criterion(pred_matched, gt_matched)
                loss_reg += self.loss_reg_criterion(pred_matched, gt_matched)

                # Alternative:
                # print(gt_matched)
                # print(pred_matched)
                # loss_reg += torch.mean((gt_raw - pred_raw)**2 / (2*pred_raw_var) + 0.5*torch.log(pred_raw_var))
                # loss_reg += torch.mean((gt_matched - pred_matched) ** 2)

                # Loss for regression variance
                loss_reg_var += torch.mean((torch.abs(pred_var_reordered) - torch.abs(gt_matched - pred_matched)) ** 2)

                # Loss for output mask
                pred_mask = pred['pred_mask'][batch_idx]
                tgt_mask = match_result['tgt_mask']
                # print("Pred mask", pred_mask.shape)
                # print("Tgt mask", tgt_mask.shape)
                loss_mask = self.loss_mask_criterion(pred_mask, tgt_mask)

            # summing all the losses
            loss = (loss_reg + loss_reg_var) * self.loss_reg_weight + loss_mask * self.loss_mask_weight

            return loss, loss_reg, loss_reg_var, loss_mask

    def training_step(self, train_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = train_batch
        pred = self.forward(s, a)

        # Calculate the loss
        loss, loss_reg, loss_reg_var, loss_mask = self.loss_fn(pred, sappear)

        self.log('train_loss', loss)
        self.log('train_reg_loss', loss_reg)
        self.log('train_reg_var_loss', loss_reg_var)
        self.log('train_mask_loss', loss_mask)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = val_batch
        pred = self.forward(s, a)

        # Calculate the loss
        loss, loss_reg, loss_reg_var, loss_mask = self.loss_fn(pred, sappear)

        self.log('val_loss', loss)
        self.log('val_reg_loss', loss_reg)
        self.log('val_reg_var_loss', loss_reg_var)
        self.log('val_mask_loss', loss_mask)