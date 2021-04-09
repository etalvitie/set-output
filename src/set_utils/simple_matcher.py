import torch
from torch import nn
from torch.distributions.normal import Normal
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

class SimpleMatcher(pl.LightningModule):
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
            pred: An dictionary containing the following values:
                "pred_reg": Regression predictions.
                "pred_reg_var": (Optional) Regression prediction variance.
                "pred_attri": Binary label predictions.
        """
        # Retrieve the predictions
        pred_reg = pred["pred_reg"]
        pred_reg_var = pred["pred_reg_var"]

        # Retrieve the dimensions
        assert len(pred_reg.shape) == 3
        num_batch, num_queries, len_reg = pred_reg.shape
        _, num_gt_queries, _ = gt.shape

        # Process each batch individually
        match_results = []
        for batch_idx in range(num_batch):
            # Calculate the cost matrix
            cost = torch.zeros((num_queries, num_gt_queries), device=self.device)
            for i in range(num_queries):
                print(mean.shape, var.shape)
                distri = Normal(loc=mean, scale=var)
                cost[i, :] = distri.cdf(gt[batch_idx])
            cost = torch.log(cost)          # Convert into log

            # calculate the position cost
            # print(pred_reg.shape, gt_reg.shape)
            # pos_cost = torch.cdist(pred_reg, gt_reg)

            # Convert to Numpy array to make scipy happy
            C = cost.squeeze()
            C = C.cpu().numpy()

            # build the index mapping
            indices = linear_sum_assignment(C)

            # Reorder the labels
            pred_order = indices[0]
            gt_order = indices[1]
            gt_reordered = gt_reg[gt_order]

            # debug
            # print("C", C.shape)
            # print("Pred_order", pred_order.shape)
            # print("gt_order", gt_order.shape)

            # Calculate the mask corresponding to the selected outputs
            gt_mask = torch.zeros((num_queries), device=self.device, dtype=torch.long)
            gt_mask[pred_order] = 1

            # for i in range(len(pred_order)):
            #     print(pred_reg[pred_order[i]], gt_reg[gt_order[i]], gt_mask[pred_order[i]])

            # print("gt mask")
            # print(gt_mask)

            # return as a dictionary
            match_result = {
                "pred_order": pred_order,
                "gt_order": gt_order,
                "gt_mask": gt_mask,
                "gt_reordered": gt_reordered
            }

            match_results.append(match_result)

        return match_results
