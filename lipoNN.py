import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# first I need to get the training data from the csv file I think using pandas
lipo_data = pd.read_cvs('lipo_fp_processed.csv')

training_data = lipo_data
# training data will be the matrix excluding the very last column and the first row
# turn into tensor maybe- use the numbers at the top to make column index maybe


# next get the testing data from the csv file
testing_data = lipo_data
# testing data will be the final column in the matrix without the first row maybe


class LipoNet(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear()
    # because this is a matrix data set would only fully connected layers be best to use
    # how does the size of my data set change how I define my layers- Just one layer right
    def forward(self, x):
        x = self.fc1(x)
        pass

# should I make my dataset a class% is it necessary if all I'm doing is isolating the last column%
# is that even what I am doing%
class LipDS():
    def __init__(self,path):
        pass


model = LipoNet()

def train(model, device, optim, epoch):
    model.train()