import csv
import math
from math import pi, cos, sin, floor, ceil
from random import random, randint, seed, uniform
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MinatarDataset(Dataset):
    def __init__(self, name=None):
        # Select one (ranked by names) if dataset name is not specified
        if name is None:
            files = glob.glob("*.npy")
            name = files[0]

        data_mat = np.load(name)
        self.data = torch.from_numpy(data_mat)

        # Retrieve the vector division information
        template = data_mat[0]
        self.action_len = int(template[1])
        self.num_obj = int(template[2])
        self.obj_len = int(template[3])

        # Print out
        print("Dataset " + name + " loaded.")
        print("Shape: ", self.data.shape)
        print("Action Vector Length: ", self.action_len)
        print("# of Objects: ", self.num_obj)
        print("Object Vector Length: ", self.obj_len)

    def __getitem__(self, idx):
        record = self.data[idx]
        reward = record[0]

        # Extract the action vector
        action = record[4: 4 + self.action_len]
        p = 4 + self.action_len     # Cursor

        # Extract the current state vector and convert into matrix
        s_cont = record[p: p + self.num_obj * self.obj_len]
        p = p + self.num_obj * self.obj_len
        s_cont = s_cont.reshape((self.num_obj, self.obj_len))

        # Extract the state prime vector and convert into matrix
        s_cont_prime = record[p: p + self.num_obj * self.obj_len]
        p = p + self.num_obj * self.obj_len
        s_cont_prime = s_cont_prime.reshape((self.num_obj, self.obj_len))

        return s_cont, action, s_cont_prime, reward

    def __len__(self):
        return len(self.data)


# Test Code
if __name__ == "__main__":
    dataset = MinatarDataset()
    print("Sample Output")
    s_cont, action, s_cont_prime, reward = dataset[1]
    print(s_cont.shape)
    print(action)
    print(s_cont_prime.shape)
    print(reward)