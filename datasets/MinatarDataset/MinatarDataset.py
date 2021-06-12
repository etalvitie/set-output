import csv
import math
from math import pi, cos, sin, floor, ceil
from random import random, randint, seed, uniform
import glob
import os
import json

from src.set_utils.variance_matcher import VarianceMatcher

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MinatarDataset(Dataset):
    def __init__(self, name=None, dataset_size=None):
        # Set size to None to use the full dataset
        self.dataset_size = dataset_size

        # Select one (ranked by names) if dataset name is not specified
        if name is None:
            files = glob.glob("*.json")
            name = files[0]

        # Find if there is matched dataset
        file_matched = name.split(".")[0] + "_matched" + ".json"
        if os.path.isfile(file_matched):
            print("Matched dataset found. Loading...")
            with open(file_matched) as f:
                data_matched = json.load(f)
                self.data_matched = data_matched

            template = self.data_matched[0]
            s, a, sprime, sappear, r = template
            self.action_len = len(a)
            self.obj_len = len(s[0])
            self.num_types = self.obj_len - 5

        else:
            print("Matched dataset not found. Matching...")
            with open(name) as f:
                data_mat = json.load(f)
                self.data = data_mat

            # Retrieve the vector division information
            template = data_mat[0]
            s, a, sprime, r = template
            self.action_len = len(a)
            self.obj_len = len(s[0][0])
            self.num_types = self.obj_len - 5

            # Perform matching
            self.data_matched = []
            self.matching()
            with open(file_matched, 'w') as json_file:
                json.dump(self.data_matched, json_file, indent=2)

        # Print out
        print("Dataset " + name + " loaded.")
        print("Dataset Size: ", len(self.data_matched))
        print("Action Vector Length: ", self.action_len)
        print("Object Vector Length: ", self.obj_len)

    def matching(self):
        matcher = VarianceMatcher()
        for i, batch in tqdm(enumerate(self.data)):
            # batch = self.data[9]
            s, a, sprime, r = batch
            s, sprime = torch.Tensor(s), torch.Tensor(sprime)

            out_set, in_set = torch.Tensor(), torch.Tensor()
            appear_set = torch.Tensor()
            for type in range(self.num_types):
                # Select the current objects with the right type
                s_mask = s[:, :, 4+type] == 1
                s_masked = s[s_mask]

                # Select the prime objects with the right type
                sprime_mask = sprime[:, :, 4+type] == 1
                sprime_masked = sprime[sprime_mask]

                # Find out the disappearing objects, appearing objects
                exist_objs = torch.Tensor()
                disappear_objs = torch.Tensor()
                appear_objs = torch.Tensor()
                if len(s_masked) == 0:
                    # If the category does not exit
                    if len(sprime_masked) == 0:
                        continue

                    # If there are new objects in this new category
                    else:
                        appear_objs = sprime_masked
                        appear_set = torch.cat([appear_set, appear_objs], dim=0)
                        continue

                else:
                    # If there are no corresponding output objects
                    # List all current objects as unmatched
                    if len(sprime_masked) == 0:
                        s_unmatched = s_masked
                        s_matched = torch.Tensor()

                    # When there are objects in this category both before and after
                    else:
                        s_dict = {
                            "pred_reg": s_masked.unsqueeze(0),
                            "pred_reg_var": torch.ones(s_masked.shape).unsqueeze(0)
                        }

                        match_result = matcher(s_dict, sprime_masked.unsqueeze(0))[0]
                        s_unmatched = match_result["pred_unmatched"]
                        s_matched = match_result["pred_reordered"]
                        sprime_unmatched = match_result["gt_unmatched"]
                        sprime_matched = match_result["gt_reordered"]

                        # Construct the existing objects for training
                        exist_objs = sprime_matched[:, 0:2]
                        exist_flag = torch.ones((sprime_matched.shape[0], 1))
                        exist_objs = torch.cat([exist_objs, exist_flag], dim=1)

                    # Construct the disppearing objects for training
                    if len(s_unmatched) != 0:
                        disappear_objs = s_unmatched[:, 0:2]
                        disappear_flag = torch.zeros((s_unmatched.shape[0], 1))
                        disappear_objs = torch.cat([disappear_objs, disappear_flag], dim=1)

                    # Construct the new objects for training
                    if len(sprime_unmatched != 0):
                        appear_objs = sprime_unmatched

                    # Check if the dimensions are correct
                    assert s_matched.shape[0] == exist_objs.shape[0]
                    assert s_unmatched.shape[0] == disappear_objs.shape[0]

                    # Append them to the output set and input set
                    in_set = torch.cat([in_set, s_matched], dim=0)
                    in_set = torch.cat([in_set, s_unmatched], dim=0)
                    out_set = torch.cat([out_set, exist_objs], dim=0)
                    out_set = torch.cat([out_set, disappear_objs], dim=0)
                    appear_set = torch.cat([appear_set, appear_objs], dim=0)
                pass

            # store in the data
            assert in_set.shape[0] == out_set.shape[0]

            # Only use the bullet
            if len(appear_set.numpy().tolist()) != 0:
                bullet_mask = (appear_set.numpy()[:, 8] == 1)
                if bullet_mask.any():
                    bullet = appear_set[bullet_mask]
                    self.data_matched.append(
                        (in_set.numpy().tolist(), a, out_set.numpy().tolist(), bullet.tolist(), r))


            # self.data_matched.append((in_set.numpy().tolist(), a, out_set.numpy().tolist(), appear_set.numpy().tolist(), r))

    def __getitem__(self, idx):
        record = self.data_matched[idx]
        s, a, sprime, sappear, r = record

        # Convert to torch Tensor
        s = torch.Tensor(s)
        a = torch.Tensor(a)
        sprime = torch.Tensor(sprime)
        sappear = torch.Tensor(sappear)
        r = torch.Tensor([r])

        return s.float(), a.float(), sprime.float()[:, 0:2], sappear, r.float()

    def __len__(self):
        if self.dataset_size is None:
            return len(self.data_matched)
        else:
            return self.dataset_size

    def get_dims(self):
        dim_dict = {
            "action_len": self.action_len,
            "obj_len": self.obj_len
        }
        return dim_dict


# Test Code
if __name__ == "__main__":
    dataset = MinatarDataset()
    print("Sample Output")
    for i in range(8, 12):
        print(i)
        s, a, sprime, sappear, r = dataset[i]
        print(s)
        # print(a)
        print(sprime)
        print(sappear)
        # print(r)
