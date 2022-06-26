import json
import math
import os
import urllib.request
import warnings
from urllib.error import HTTPError

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

        
class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}    
        
        
        
class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


class Tanh(ActivationFunction):
    def forward(self, x):
        x_exp, neg_x_exp = torch.exp(x), torch.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)
    
    
    
class ReLU(ActivationFunction):
    def forward(self, x):
        return x * (x > 0).float()


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.config["alpha"] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["alpha"] * x)


class ELU(ActivationFunction):
    def forward(self, x):
        return torch.where(x > 0, x, torch.exp(x) - 1)


class Swish(ActivationFunction):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TanhELU(ActivationFunction):
    def __init__(self, p=.95):
        super().__init__()
        self.config["p"] = p

    def elu(self, x):
        return torch.where(x > 0, x, torch.exp(x) - 1)

    def tanh(self, x):
        x_exp, neg_x_exp = torch.exp(x), torch.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)

    def forward(self, x):
        device = x.device
        
        if x.get_device()<0:
            # use cpu
            flips = torch.rand_like(x)
        else:
            # use gpu
            flips = torch.rand_like(x).to(device)
        z = torch.where(flips > self.config["p"], self.elu(x), self.tanh(x))
        return z
        


class TanhReLU(ActivationFunction):
    def __init__(self, p=.95):
        super().__init__()
        self.config["p"] = p

    def relu(self, x):
        return x * (x > 0).float()

    def tanh(self, x):
        x_exp, neg_x_exp = torch.exp(x), torch.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)

    def forward(self, x):
        device = x.device
        
        if x.get_device()<0:
            # use cpu
            flips = torch.rand_like(x)
        else:
            # use gpu
            flips = torch.rand_like(x).to(device)
        z = torch.where(flips > self.config["p"], self.relu(x), self.tanh(x))
        return z
        
    