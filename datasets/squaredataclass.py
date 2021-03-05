import csv
from random import random, randint, seed, uniform
import math
from math import pi, cos, sin, floor, ceil

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class SquareDataset(Dataset):
    generator_type=None

    def __init__(self, data_root): 
        self.data = []

        results = []
        with open(data_root) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        self.data = torch.Tensor(results)

    def __init__(self, num, generator_type="basic"):
        """
        Generate SquareDataset using data generation.
        Available generators: basic[default], linear movement, rotation.
        """
        generator = {
            "basic": sampledata,
            "linear": sample_linear_data,
            "rotation": sample_rotation_data
        }[generator_type]
        self.generator_type = generator_type 

        data = []
        for i in tqdm(range(1000)):
            data.append(generator())

        self.data = torch.Tensor(data)
        
    def __getitem__(self,idx):
        if self.generator_type == "linear":
            return self.data[idx][0:-1], self.data[idx][-1]
        else:
            return self.data[idx][0:9], self.data[idx][9:None]

    def __len__(self): 
        return len(self.data)


def sample_rotation_data():
    # Generate two random angles
    theta1 = uniform(0,1)*2*pi
    theta2 = uniform(0,1)*2*pi
    d_theta = theta2 - theta1

    # Calculate the coordinates of all vertices
    v1 = [(math.e)**(1j*theta1 + i*1j*pi/2) for i in range(4)]
    v2 = [(math.e)**(1j*theta2 + i*1j*pi/2) for i in range(4)]
    v_comb = v1 + v2

    # Convert and store into a list
    data = []
    for i in range(len(v_comb)): 
        data.extend([v_comb[i].real, v_comb[i].imag])
    ##random.shuffle(data)

    return [d_theta] + data

def sample_linear_data():
    """
    Squares with linear velocities.
    """
    # Generate initial objects
    theta = uniform(0,1)*2*pi

    # Generate random speed
    speed = uniform(0,1)
    
    # Calculate the coordinates of all vertices
    vertices = [(math.e)**(1j*theta + i*1j*pi/2) for i in range(4)]

    # Calculate the vertices positions
    data_0 = []
    data_1 = []
    for i in range(len(vertices)): 
        data_0.extend([vertices[i].real, vertices[i].imag])
        data_1.extend([vertices[i].real + speed, vertices[i].imag])

    return [speed] + data_0 + data_1


def sampledata(): 
    b = uniform(0,1)
    a = b*2*pi
    data = [(math.e)**(1j*a + i*1j*pi/2) for i in range(4)]
    for i in range(len(data)): 
        data[i] = (data[i].real, data[i].imag)
    ##random.shuffle(data)
   
    newdata = []
    for i in data: 
        
        newdata += list(i)
    return newdata + [b]
