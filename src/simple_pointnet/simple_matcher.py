import torch
from torch import nn
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

class SimpleMatcher(pl.LightningModule):
    """
    1-to-1 matching of 4 objects.
    """

    def __init__(self, cost_pos: float = 1, cost_attri: float = 1):
        """Creates the matcher
        Params:
            costs coefficients (not used rn)
        """
        super().__init__()
        self.cost_pos = cost_pos
        self.cost_attri = cost_attri

    @torch.no_grad()
    def forward(self, pred, gt, VERBOSE=False):
        """ Performs the matching
        Params:
    
        """
        # Retrieve the basic info
        num_queries = pred["pred_reg"].shape[-2]
        pred_reg = pred["pred_reg"].squeeze()
        gt_reg = gt.squeeze()

        # Make sure that the input array is 2D
        if len(pred_reg.shape) == 1:
            pred_reg = pred_reg.unsqueeze(1)
            gt_reg = gt_reg.unsqueeze(1)

        # print(pred_reg.shape)
        # print(gt_reg.shape)

        # calculate the position cost
        pos_cost = torch.cdist(pred_reg, gt_reg)

        # Convert to Numpy array to make scipy happy
        C = pos_cost.squeeze()
        C = C.cpu().numpy()

        # build the index mapping
        indices = linear_sum_assignment(C)

        # Reorder the labels
        pred_order = indices[0]
        gt_order = indices[1]
        gt_reordered = gt_reg[gt_order]

        # Calculate the mask corresponding to the selected outputs
        gt_mask = torch.zeros((num_queries), device=self.device)
        gt_mask[pred_order] = 1
        # gt_mask[~pred_order] = -1

        if VERBOSE:
            print("Cost")
            print(C)
            print("Pred pos")
            print(pred_reg)
            print("Ground Truth pos")
            print(gt_reordered)

        # return as a dictionary
        matching_results = {
            "pred_order": pred_order,
            "gt_order": gt_order,
            "gt_mask": gt_mask,
            "gt_reordered": gt_reordered
        }

        return matching_results


        
