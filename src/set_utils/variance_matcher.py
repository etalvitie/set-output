from math import pi

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


class VarianceMatcher(pl.LightningModule):
    def __init__(self, cost_reg: float = 1, cost_attri: float = 1):
        """Creates the matcher
        Params:
            costs coefficients (not used rn)
        """
        super().__init__()
        self.cost_reg = cost_reg
        self.cost_attri = cost_attri

        self.relu = torch.nn.ReLU()

    def forward(self, pred, gt, VERBOSE=False):
        """ Performs the matching
        Params:
            pred: A dictionary containing the following keys:
                "pred_reg": Regression predictions.
                "pred_reg_var": (Optional) Regression prediction variance.
                "pred_attri": Binary label predictions.
            gt: Ground truth label in its raw format.

        Returns:
            A list of dictionaries. Each dictionary contains the matched results:
                "pred_order": the new order of the prediction.
                "gt_order": the new order of the ground truth label.
                "pred_reordered": the prediction after reordering.
                "gt_reordered": the ground truth after reordering.
                "pred_diff": the difference between the prediction and the ground truth label.
                "tgt_mask": target mask.
        """
        # Retrieve the predictions
        pred_reg = pred["pred_reg"]
        pred_reg_var = pred["pred_reg_var"]
        pred_type = pred['pred_type']

        # Retrieve the dimensions
        assert len(pred_reg.shape) == 3
        num_batch, num_queries, len_reg = pred_reg.shape
        _, _, len_type = pred_type.shape


        # Process each batch individually
        match_results = []
        for batch_idx in range(num_batch):
            # Find out how many objects in the ground truth label
            num_gt_queries = gt[batch_idx].shape[0]

            # Regression data in ground truth labels
            gt_reg = gt[batch_idx][:, 0:len_reg]
            gt_type = gt[batch_idx][:, len_reg:len_reg+len_type]

            # Calculate the cost matrix
            with torch.set_grad_enabled(False):
                cost = torch.zeros((num_queries, num_gt_queries), device=self.device)

                mode = None
                if mode == "EM":
                    for i in range(num_queries):
                        mean = pred_reg[batch_idx][i]
                        covar = pred_reg_var[batch_idx][i]
                        x = gt[batch_idx]
                        # print(mean, covar)
                        cost_ = normal_log_prob(mean, covar, x)
                        assert not torch.isnan(cost_[0])
                        cost[i,:] = cost_
                    # cost = torch.log(cost)          # Convert into log
                    cost = -cost
                else:
                    reg_cost = torch.cdist(pred_reg[batch_idx], gt_reg)
                    # type_cost = torch.exp(torch.cdist(pred_type[batch_idx], gt_type))
                    type_cost = self.relu(torch.cdist(pred_type[batch_idx], gt_type)) + 1
                    # cost = reg_cost
                    cost = reg_cost * type_cost
                    # print(cost)

                # Convert to Numpy array to make scipy happy
                C = cost
                if len(cost.shape)== 3:
                    C = C.squeeze()
                C = C.cpu().numpy()

            # Build the index mapping which leads the maximum probability
            indices = linear_sum_assignment(C)

            # Reorder the labels
            gt_order = indices[1]
            gt_reordered = gt[batch_idx][gt_order]
            gt_reg_reordered = gt_reg[gt_order]
            gt_type_reordered = gt_type[gt_order]

            # Reorder the predictions
            pred_order = indices[0]
            # print("Pred Reg: ", pred_reg.shape)
            pred_reg_reordered = pred_reg[batch_idx][pred_order]
            pred_var_reordered = pred_reg_var[batch_idx][pred_order]
            pred_type_reordered = pred_type[batch_idx][pred_order]

            # Calculate the difference between the prediction and the ground truth label
            pred_diff = pred_reg_reordered - gt_reg_reordered

            # Calculate the target mask corresponding to the selected outputs
            tgt_mask = torch.zeros(num_queries, device=self.device, dtype=torch.float)
            tgt_mask[pred_order] = 1

            # Find the disappearing objects
            if (tgt_mask == 1).all():
                pred_unmatched = torch.Tensor()
            else:
                unmatched_mask = tgt_mask != 1
                pred_unmatched = pred_reg[batch_idx][unmatched_mask]

            # Find the appearing objects
            gt_mask = torch.zeros(num_gt_queries, device=self.device, dtype=torch.long)
            gt_mask[gt_order] = 1
            if (gt_mask == 1).all():
                gt_unmatched = torch.Tensor()
            else:
                unmatched_mask = gt_mask != 1
                gt_unmatched = gt[batch_idx][unmatched_mask]


            # return as a dictionary
            match_result = {
                "pred_order": pred_order,
                "gt_order": gt_order,
                "pred_reg_reordered": pred_reg_reordered,
                "pred_var_reordered": pred_var_reordered,
                "gt_reordered": gt_reordered,
                "tgt_mask": tgt_mask,
                "pred_unmatched": pred_unmatched,
                "gt_unmatched": gt_unmatched,
                "pred_type_reordered": pred_type_reordered,
                "gt_type_reordered": gt_type_reordered
            }

            # print("Pred_before:")
            # print(pred_reg[batch_idx])
            # print("Gt_before:")
            # print(gt[batch_idx])
            #
            # print("Pred_reordered")
            # print(pred_reordered)
            # print("GT_reordered")
            # print(gt_reordered)

            match_results.append(match_result)

        return match_results

def normal_log_prob(mean, covar, data):
    """
    Calculates the log of the probability density.

    Args:
        mean: Length D vector.
        covar: Notice that this is a length D vector rather than the complete covariance matrix.
        data: Queries. Shape : [N, D].

    Returns:

    """
    # Assert the dimensions of the inputs
    d = mean.shape[0]
    assert covar.shape[0] == d
    assert data.shape[1] == d
    num_queries, _ = data.shape

    # Calculates the probability density
    # prob = torch.zeros(num_queries, device=data.device)
    x = data
    diff = x - mean
    # diff_t = diff.transpose()
    exp = (-0.5) * diff**2 / covar
    scale = torch.sqrt(2*pi*covar)
    prob = torch.exp(exp) / scale

    # Takes the log
    log_prob = torch.log(prob + 1e-10)

    # Added across multiple variables
    log_prob = torch.sum(log_prob, dim=1)

    return log_prob


"""
Test Code
"""

if __name__ == "__main__":
    gt_points = [[[4, 3.9, 1, 0], [5, 4.9, 0, 1], [2, 1.9, 1, 0], [1.8, 1.7, 0, 1]]]

    pred_points = [[[3.9, 4], [4, 3.85], [1.7, 1.8]]]
    pred_types = [[[1, 0, 1], [1, 0, 1], [1, 1, 0]]]

    pred_reg = torch.Tensor(pred_points)
    gt = torch.Tensor(gt_points)
    pred_types = torch.Tensor(pred_types)

    pred = {
        "pred_reg": pred_reg,
        "pred_reg_var": torch.ones(pred_reg.shape),
        "pred_attri": pred_types
    }

    matcher = VarianceMatcher()
    match_results = matcher(pred, gt)

    pred_matched = match_results[0]["pred_reg_reordered"].numpy()
    pred_type_matched = match_results[0]["pred_type_reordered"].numpy()
    pred_type_idx = np.argmax(pred_type_matched, axis=1)
    gt_matched = match_results[0]["gt_reordered"].numpy()
    gt_type_matched = match_results[0]["gt_type_reordered"].numpy()
    gt_type_idx = np.argmax(gt_type_matched, axis=1)

    print(pred_matched)
    print(gt_matched)

    num_matches = len(pred_matched)
    idx_list = np.arange(num_matches)

    sns.scatterplot(
        x=pred_matched[:, 0], y=pred_matched[:, 1],
        hue=idx_list, style=pred_type_idx, alpha=0.5
    )

    sns.scatterplot(
        x=gt_matched[:, 0], y=gt_matched[:, 1],
        hue=idx_list, style=gt_type_idx,
    )
    plt.show()
