import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

class SimpleMatcher(nn.Module):
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
        bs, num_queries = pred["pred_pos"].shape[:2]
        pred_pos = pred["pred_pos"].squeeze()
        gt_pos = gt.squeeze()

        # Make sure that the input array is 2D
        if len(pred_pos.shape) == 1:
            pred_pos = pred_pos.unsqueeze(1)
            gt_pos = gt_pos.unsqueeze(1)

        # print(pred_pos.shape)
        # print(gt_pos.shape)

        # calculate the position cost
        pos_cost = torch.cdist(pred_pos, gt_pos)

        # Convert to Numpy array to make scipy happy
        C = pos_cost.squeeze()
        C = C.cpu().numpy()

        # build the index mapping
        indices = linear_sum_assignment(C)

        # Reorder the labels
        pred_new_order = indices[0]
        gt_new_order = indices[1]
        gt_reordered = gt_pos[gt_new_order]

        if VERBOSE:
            print("Cost")
            print(C)
            print("Pred pos")
            print(pred_pos)
            print("Ground Truth pos")
            print(gt_reordered)

        return pred_new_order, gt_reordered


        
