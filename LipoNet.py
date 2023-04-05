#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

#import matplotlib.pyplot as plt

class LipoNet(nn.Module):
    def __init__(self):
        super(LipoNet, self).__init__()
        # super is saying that class LipoNet is inheriting traits from nn.Module class
        input_size = 1024
        output_size = 1
        self.fc1 = nn.Linear(input_size, output_size) #(raw data set) # nums inside nn.Linear() are wrong
        # nn.Linear(size of dataset, size of output)
        self.fc2 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        # output (op) will have the same dimensions as the target output (the ground truth?)
        return out #op