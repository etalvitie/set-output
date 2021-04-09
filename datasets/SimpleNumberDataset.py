import csv
import math
from math import pi, cos, sin, floor, ceil
from random import random, randint, seed, uniform

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SimpleNumberDataset(Dataset):
    generator_type=None

    def __init__(self, data_root): 
        self.data = []

        results = []
        with open(data_root) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        self.data = torch.Tensor(results)

    def __init__(self, num, set_size, max, threshold):
        """
        Generate a Simple Number Dataset.

        Args:
            num: The size of the dataset.
            set_size:  The size of the input set.
            max: The maximum of the input number.
            threshold: The maximum distance between any two input numbers to trigger the average computation.
        """
        self.set_size = set_size
        self.data = []

        for _ in range(num):
            input_data = []
            output_data = []
            # A brute force approach to generate the dataset
            for _ in range(set_size):
                new_num = random() * max

                for existing_num in input_data:
                    if abs(new_num - existing_num) < threshold:
                        output_data.append(np.mean([new_num, existing_num]))

                input_data.append(new_num)
            
            self.data.append( torch.Tensor([threshold, len(output_data)] + input_data + output_data) )
        
    def __getitem__(self,idx):
        return self.data[idx][0:(2+self.set_size)], self.data[idx][(2+self.set_size):None]

    def __len__(self): 
        return len(self.data)

"""
Test module
"""
def show_sample_data():
    dataset = SimpleNumberDataset(5, 10, 100, 10)
    for input, output in dataset:
        print(input)
        print(output)
        print()


if __name__ == "__main__":
    show_sample_data()