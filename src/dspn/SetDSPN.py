from src.dspn.FSEncoder import FSEncoder
from src.dspn.DSPN_copy import deepsetnet, hungarian_loss
from src.dspn.DSPN import DSPN
from src.dspn.SetEncoder import SetEncoder
from src.set_utils.variance_matcher import VarianceMatcher
from torch.utils.data import Dataset

import random
import math
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class SetDSPN(pl.LightningModule):
    '''
    Parameters: encoder1: the set encoder
    encoder2: the encoder that gets passed into the DSPN
    latent_dim: the output dimension of encoder1
    set_dim: (number of objects, length of each object)
    n_iters: n_iters parameter for DSPN
    masks: for DSPN, doesn't actually do anything
    '''

    def __init__(self,
                 set_encoder=None,
                 dspn_encoder=None,
                 obj_in_len=9,
                 obj_reg_len=2,
                 obj_type_len=2,
                 env_len=6,
                 latent_dim=64,
                 out_set_size=5,
                 n_iters=10,
                 internal_lr=0.5,
                 overall_lr=1e-3,
                 loss_encoder_weight=0.1,
                 loss_spr_weight=10,
                 loss_type_weight=1,
                 type_separate=False):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.obj_in_len = obj_in_len
        self.obj_reg_len = obj_reg_len
        self.obj_type_len = obj_type_len
        self.obj_attri_len = obj_type_len + 1   # +1 because of the mask
        self.n_iters = n_iters
        self.internal_lr = internal_lr
        self.learning_rate = overall_lr
        self.type_separate = type_separate

        out_obj_len =  obj_reg_len*2 + self.obj_attri_len
        out_set_dim = (out_set_size, out_obj_len)
        self.out_set_dim = out_set_dim

        if type_separate:
            self.modules = nn.ModuleList()
            for _ in range(obj_type_len):
                self.modules.append(SetDSPN(
                    set_encoder=set_encoder,
                    dspn_encoder=dspn_encoder,
                    obj_in_len=obj_in_len,
                    obj_reg_len=obj_reg_len,
                    obj_type_len=obj_type_len,
                    env_len=env_len,
                    latent_dim=latent_dim,
                    out_set_size=out_set_size,
                    n_iters=n_iters,
                    internal_lr=internal_lr,
                    overall_lr=overall_lr,
                    loss_encoder_weight=loss_encoder_weight,
                    loss_spr_weight=loss_spr_weight,
                    loss_type_weight=loss_type_weight,
                    type_separate=False
                ))
        else:
            if set_encoder is None:
                set_encoder = SetEncoder(env_len=env_len, obj_in_len=obj_in_len)
            self.set_encoder = set_encoder

            if dspn_encoder is None:
                dspn_encoder = FSEncoder(out_obj_len, 576, dim=512)
                # dspn_encoder = SetEncoder(env_len=0, obj_in_len=obj_out_len, obj_hidden_dim=1088)
            self.dspn_encoder = dspn_encoder
            # self.decoder = deepsetnet(dspn_encoder, latent_dim, out_set_dim, n_iters, masks)
            self.decoder = DSPN(dspn_encoder, out_obj_len, out_set_size, n_iters, internal_lr)

            # Output masks
            self.mask_softmax = nn.Softmax(dim=2)

            self.matcher = VarianceMatcher()
            # Loss + matching calculation
            self.matcher = VarianceMatcher()
            self.loss_reg_criterion = nn.MSELoss()
            self.loss_mask_criterion = nn.MSELoss()
            self.loss_encoder_criterion = nn.MSELoss()
            self.loss_type_criterion = nn.CrossEntropyLoss()

            self.loss_reg_weight = 1
            self.loss_mask_weight = 1
            self.loss_encoder_weight = loss_encoder_weight
            self.loss_spr_weight = loss_spr_weight
            self.loss_type_weight = loss_type_weight



    def forward(self, x, a=None):
        bs = x.shape[0]

        if self.type_separate:
            results = defaultdict(None)
            for i in range(len(self.modules)):
                result = self.modules[i](x, a)

                # Force the prediction type
                pred_type = torch.zeros((bs, self.out_set_size, self.obj_type_len), device=self.device)
                pred_type[:,:,i] = 1
                result['pred_type'] = pred_type

                for key in result.keys():
                    results[key] = torch.cat([results[key], result[key]], dim=1)

            return results

        else:
            h = self.set_encoder(x, a)

            intermediate_sets, intermediate_masks, repr_losses, grad_norms = self.decoder(h)
            # pred = pred.reshape(self.out_set_dim)
            # pred = pred.unsqueeze(0)

            # Regressions
            # pred_reg = pred[:, :, 0:self.obj_reg_len]
            # pred_reg_var = pred[:, :, self.obj_reg_len:2*self.obj_reg_len]
            # pred_reg_var = pred_reg_var ** 2 + 1e-6

            # Attributes
            # pred_attri = pred[:, :, 2*self.obj_reg_len:None]
            # pred_mask = pred_attri[:, :, 0:2]
            # pred_mask = self.mask_softmax(pred_mask)

            pred = intermediate_sets[-1].permute(0, 2, 1)
            pred_reg = pred[:, :, 0:self.obj_reg_len]
            pred_reg_var = pred[:, :, self.obj_reg_len:2*self.obj_reg_len]
            pred_type = pred[:, :, 2*self.obj_reg_len:2*self.obj_reg_len+self.obj_type_len]
            pred_mask = intermediate_masks[-1]
            result = {'pred_type': pred_type,
                    'pred_mask': pred_mask,
                    'pred_reg': pred_reg,
                    'scene_vector': h,
                    'pred_reg_var': pred_reg_var}
            return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def loss_fn(self, pred, gt_label):
        # Putting into a dictionary
        zero_loss = torch.zeros(1, device=self.device)
        losses = {
            'loss': zero_loss,
            'loss_reg': zero_loss,
            'loss_reg_var': zero_loss,
            'loss_mask': zero_loss,
            'loss_encoder': zero_loss,
            'loss_type': zero_loss
        }

        # If the output set is empty
        if gt_label.shape[1] == 0:
            pred_mask = pred['pred_mask'][0]
            tgt_mask = torch.zeros(pred_mask.shape[0], device=self.device, dtype=torch.float)
            loss_mask = self.loss_mask_criterion(pred_mask, tgt_mask)
            losses['loss'] = loss_mask
            losses['loss_mask'] = loss_mask

        # Perform the matching
        else:
            # gt_label = gt_label[:, :, 0:2]
            match_results = self.matcher(pred, gt_label)

            # Calculate the loss for regression and masking seperately
            loss_reg = 0
            loss_reg_var = 0
            loss_mask = 0
            loss_spr = 0
            loss_type = 0

            # Iterate through all batches
            for batch_idx, match_result in enumerate(match_results):
                pred_raw = pred["pred_reg"][batch_idx].contiguous()     # Set to be contiguous for seperation loss cdist calculation
                gt_raw = gt_label[batch_idx]
                pred_var_raw = pred["pred_reg_var"][batch_idx]
                pred_var_matched = match_result["pred_var_reordered"]
                pred_reg_matched = match_result['pred_reg_reordered']
                gt_matched = match_result['gt_reordered']
                gt_reg_matched = gt_matched[:, 0:self.obj_reg_len]

                # Loss for regression (position)
                # loss_reg += self.loss_reg_criterion(pred_matched, gt_matched)
                loss_reg += self.loss_reg_criterion(pred_reg_matched, gt_reg_matched)

                # Alternative:
                # print(gt_matched)
                # print(pred_matched)
                # loss_reg += torch.mean((gt_matched - pred_matched)**2 / (2*pred_var_matched) + 0.5*torch.log(pred_var_matched))
                # loss_reg += torch.mean((gt_matched - pred_matched) ** 2)

                # Loss for regression variance
                loss_reg_var += torch.mean((torch.abs(pred_var_matched) - torch.abs(gt_reg_matched - pred_reg_matched)**2) **2)

                # Loss for output mask
                pred_mask = pred['pred_mask'][batch_idx]
                tgt_mask = match_result['tgt_mask']
                # print("Pred mask", pred_mask.shape)
                # print("Tgt mask", tgt_mask.shape)
                loss_mask += self.loss_mask_criterion(pred_mask, tgt_mask)

                # Loss for object type prediction
                pred_type_matched = match_result['pred_type_reordered']
                gt_type_matched = match_result['gt_type_reordered']
                gt_type_matched_idx = torch.argmax(gt_type_matched, dim=1)
                loss_type += self.loss_type_criterion(pred_type_matched, gt_type_matched_idx)

                # Loss for prediction separation
                dist_matrix = torch.cdist(pred_raw, pred_raw, p=2)
                # loss_spr += 1/torch.sum(dist_matrix)
                loss_spr += torch.exp(-torch.sum(dist_matrix)/10)

            # Loss for the encoder
            gt_scene, gt_mask = self._cvt_to_scene_vect(gt_label)
            gt_new_set_vector = self.dspn_encoder(gt_scene, gt_mask)
            scene_vector = pred['scene_vector']
            loss_encoder = self.loss_encoder_criterion(gt_new_set_vector, scene_vector)

            # summing all the losses
            loss = (loss_reg + loss_reg_var) * self.loss_reg_weight +\
                   loss_mask * self.loss_mask_weight +\
                   loss_encoder * self.loss_encoder_weight +\
                   loss_spr * self.loss_spr_weight + \
                   loss_type * self.loss_type_weight

            # Putting into a dictionary
            losses = {
                'loss': loss,
                'loss_reg': loss_reg,
                'loss_reg_var': loss_reg_var,
                'loss_mask': loss_mask,
                'loss_encoder': loss_encoder,
                'loss_spr': loss_spr,
                'loss_type': loss_type
            }

        return losses

    def training_step(self, train_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = train_batch
        pred = self.forward(s, a)

        # Calculate the loss
        losses = self.loss_fn(pred, sappear)

        self.log('train_loss', losses['loss'])
        self.log('train_reg_loss', losses['loss_reg'])
        self.log('train_reg_var_loss', losses['loss_reg_var'])
        self.log('train_mask_loss', losses['loss_mask'])
        self.log('train_encoder_loss', losses['loss_encoder'])
        if 'loss_spr' in losses:
            self.log('train_spr_loss', losses['loss_spr'])
        self.log('train_type_loss', losses['loss_type'])

        return losses['loss']

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = val_batch
        pred = self.forward(s, a)

        # Calculate the loss
        losses = self.loss_fn(pred, sappear)

        self.log('val_loss', losses['loss'])
        self.log('val_reg_loss', losses['loss_reg'])
        self.log('val_reg_var_loss', losses['loss_reg_var'])
        self.log('val_mask_loss', losses['loss_mask'])
        self.log('val_encoder_loss', losses['loss_encoder'])
        self.log('val_spr_loss', losses['loss_spr'])
        self.log('val_type_loss', losses['loss_type'])

    def _cvt_to_scene_vect(self, gt_label):
        bs, num_obj, _ = gt_label.shape

        # Transpose
        gt_label = gt_label.permute(0, 2, 1)

        # Paddings
        zero_var_pad = torch.zeros([bs, self.obj_reg_len, num_obj], device=self.device, dtype=torch.float) + 1e-6
        vis_pad = torch.ones([bs, 1, num_obj], device=self.device, dtype=torch.float)

        # Concatenate back
        gt_label = torch.cat([gt_label[:, 0:self.obj_reg_len],
                              zero_var_pad,
                              gt_label[:, self.obj_reg_len:self.obj_reg_len + self.obj_type_len],
                              vis_pad],
                             dim=1)

        # Also generate a mask
        gt_mask = torch.ones([bs, num_obj], device=self.device, dtype=torch.float)

        return gt_label, gt_mask
