import csv
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader

class SquareDataset(Dataset):
    def __init__(self, data_root): 
        self.data = []


        results = []
        with open(data_root) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC, delimiter='\t') # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        self.data = torch.Tensor(results)
        
    def __getitem__(self,idx): 
        return self.data[idx][0:-1], self.data[idx][-1]
    def __len__(self): 
        return len(self.data)
 
def sampledata(): 
    b = random.uniform(0,1)
    a = b*2*math.pi
    data = [(math.e)**(1j*a + i*1j*math.pi/2) for i in range(4)]
    for i in range(len(data)): 
        data[i] = (data[i].real, data[i].imag)
    ##random.shuffle(data)
   
    newdata = []
    for i in data: 
        
        newdata += list(i)
    return newdata + [b]
