import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

from math import pi


class VarianceMatcher(pl.LightningModule):
    def __init__(self, cost_reg: float = 1, cost_attri: float = 1):
        """Creates the matcher
        Params:
            costs coefficients (not used rn)
        """
        super().__init__()
        self.cost_reg = cost_reg
        self.cost_attri = cost_attri

    @torch.no_grad()
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

        # Retrieve the dimensions
        assert len(pred_reg.shape) == 3
        num_batch, num_queries, len_reg = pred_reg.shape

        # Process each batch individually
        match_results = []
        for batch_idx in range(num_batch):
            # Find out how many objects in the ground truth label
            num_gt_queries = gt[batch_idx].shape[0]

            # Regression data in ground truth labels
            gt_reg = gt[batch_idx][:, 0:len_reg]

            # Calculate the cost matrix
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
                cost = torch.cdist(pred_reg[batch_idx], gt_reg)
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

            # Reorder the predictions
            pred_order = indices[0]
            # print("Pred Reg: ", pred_reg.shape)
            pred_reordered = pred_reg[batch_idx][pred_order]
            pred_var_reordered = pred_reg_var[batch_idx][pred_order]

            # Calculate the difference between the prediction and the ground truth label
            pred_diff = pred_reordered - gt_reg_reordered

            # Calculate the target mask corresponding to the selected outputs
            tgt_mask = torch.zeros(num_queries, device=self.device, dtype=torch.long)
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
                "pred_reordered": pred_reordered,
                "pred_var_reordered": pred_var_reordered,
                "gt_reordered": gt_reordered,
                "tgt_mask": tgt_mask,
                "pred_unmatched": pred_unmatched,
                "gt_unmatched": gt_unmatched
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
    points2 = [[[99, 98], [50, 49], [20, 19], [18, 17]]]
    points1 = [[[98, 99], [19, 20], [17, 18]]]

    pred_reg = torch.Tensor(points1)
    gt = torch.Tensor(points2)
    pred = {
        "pred_reg": pred_reg,
        "pred_reg_var": torch.ones(pred_reg.shape)
    }

    matcher = VarianceMatcher()
    match_results = matcher(pred, gt)
    print(match_results[0]["pred_reordered"])
    print(match_results[0]["gt_reordered"])
