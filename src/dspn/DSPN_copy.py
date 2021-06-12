import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize
import torch.nn.functional as F
import random
import math
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt

class deepsetnet(nn.Module): 
    '''
    Initialize the deepsetnet. 
    Parameters: 
    Encoder: the encoder network to be used 
    latent_dim: the dimension of the encoder output
    set_dim: a tuple of (number of elements, length of elements), for square data is (4,2)
    n_iters: number of gradient descent iterations for the forward pass 
    masks: True if we are using variable size sets with masks 
    '''
    def __init__(self, encoder, latent_dim, set_dim, n_iters, masks = False):
        ## Becuase of how the data turned out, set dim needs to be the transpose of the real set dim
        super().__init__()
        self.masks = masks
        self.encoder = encoder.requires_grad_(True)
        self.set_dim = set_dim 
        self.latent_dim = latent_dim 
        self.n_iters = n_iters
        self.learn_rate = 0.5
        self.loss_func = nn.smooth_l1_loss(reduction = 'mean')
        
        self.length = 1
        for i in set_dim: 
            self.length = self.length*i
        self.Y0 = nn.Parameter(torch.randn(self.length)/10, requires_grad = True)

    def forward(self, x): 
        input1 = self.Y0

        with torch.enable_grad(): 
            ##gradient descent loop for the forward pass 
            for i in range(self.n_iters): 
                
                output = self.encoder(input1)
                loss = self.loss_func(output, x)
                
                output_grad = torch.autograd.grad(inputs = input1 ,outputs = loss,only_inputs=True,
                    create_graph=True,)

                input1 = input1 - self.learn_rate*output_grad[0]
                # assert i!=8

        y = input1
        return y 

def hungarian_loss(output,target, set_dim_transpose):

    with torch.enable_grad():
        if target.shape[1] == 0:
            loss = torch.sum(output) * 0
            return loss

        output = output.reshape(set_dim_transpose).transpose(0,1)
        target = target[:, :]
        target = target.reshape(set_dim_transpose).transpose(0,1)
        target=target[torch.randperm(target.size()[0])]
        diff_mat  = torch.Tensor([[sum((i-j)**2) for i in output] for j in target])
        assignments = scipy.optimize.linear_sum_assignment(diff_mat.numpy())[1]
        loss = 0
        for i in range(len(assignments)):
            loss += (target[i]-output[assignments[i]])**2

    return sum(loss)

def chamfer_loss(output, target, set_dim_transpose):  
    with torch.enable_grad():
        output = output.reshape(set_dim_transpose).transpose(0,1)
        target = target.reshape(set_dim_transpose).transpose(0,1)
        diff_mat  = [[sum((i-j)**2) for i in output] for j in target]
        diff_mat2 = [0 for i in range(len(diff_mat))]
        for i in range(len(diff_mat)): 
            diff_mat2[i] = torch.stack(diff_mat[i])
        diff_mat2 = torch.stack(diff_mat2,dim = 1)
        
        min1 = diff_mat2.min(0)
        min2 = diff_mat2.min(1)
    return sum(min2.values) + sum(min1.values)



        
