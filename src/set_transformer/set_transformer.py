"""
Modifired based on
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
"""
from .modules import *
from src.set_utils.variance_matcher import VarianceMatcher

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SetTransformer(pl.LightningModule):
    def __init__(self,
                 obj_in_len=9,
                 obj_reg_len=2,
                 obj_type_len=2,
                 env_len=6,
                 out_set_size=5,
                 num_inds=32,
                 dim_hidden=512,
                 num_heads=4,
                 ln=False,
                 loss_spr_weight=10,
                 loss_type_weight=1,
                 learning_rate=1e-3):
        super(SetTransformer, self).__init__()

        self.learning_rate = learning_rate

        self.obj_in_len = obj_in_len
        self.obj_reg_len = obj_reg_len
        self.obj_type_len = obj_type_len

        dim_output = obj_in_len
        dim_input = obj_in_len + env_len

        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, out_set_size, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

        # Loss + matching calculation
        self.matcher = VarianceMatcher()
        self.loss_reg_criterion = nn.MSELoss()
        self.loss_mask_criterion = nn.MSELoss()
        self.loss_encoder_criterion = nn.MSELoss()
        self.loss_type_criterion = nn.CrossEntropyLoss()

        # Loss Weights
        self.loss_reg_weight = 1
        self.loss_mask_weight = 1
        self.loss_spr_weight = loss_spr_weight
        self.loss_type_weight = loss_type_weight

    def forward(self, objs, a):
        assert len(objs.shape) == 3
        in_set_size = objs.shape[1]

        a = a.repeat((1, in_set_size, 1))
        X = torch.cat([objs, a], dim=2)

        result = self.dec(self.enc(X))

        # Regression prediction postprocessing
        pred_reg = result[:, :, 0:self.obj_reg_len]
        pred_reg_var = result[:, :, self.obj_reg_len:2*self.obj_reg_len]
        pred_reg_var = pred_reg_var ** 2 + 1e-6  # Prevent zero covariance causing errors

        # Extract the output masks
        pred_type = result[:, :, 2*self.obj_reg_len:2*self.obj_reg_len + self.obj_type_len]
        pred_mask = result[:, :, 2*self.obj_reg_len + self.obj_type_len]

        return {'pred_mask': pred_mask,
                'pred_reg': pred_reg,
                'pred_reg_var': pred_reg_var,
                'pred_type': pred_type}

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
            'loss_type': zero_loss,
            'loss_spr': zero_loss
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

            # summing all the losses
            loss = (loss_reg + loss_reg_var) * self.loss_reg_weight +\
                   loss_mask * self.loss_mask_weight +\
                   loss_spr * self.loss_spr_weight + \
                   loss_type * self.loss_type_weight

            # Putting into a dictionary
            losses = {
                'loss': loss,
                'loss_reg': loss_reg,
                'loss_reg_var': loss_reg_var,
                'loss_mask': loss_mask,
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

        # Log all the losses
        for key in losses.keys():
            key_description = "train_" + key
            self.log(key_description, losses[key])

        return losses['loss']

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = val_batch
        pred = self.forward(s, a)

        # Calculate the loss
        losses = self.loss_fn(pred, sappear)

        # Log all the losses
        for key in losses.keys():
            key_description = "val_" + key
            self.log(key_description, losses[key])