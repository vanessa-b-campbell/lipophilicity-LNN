import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#%%

###############training will be 80% of the whole thing derivative of the loss functuion feed in batches pg 31 in pytorch
# no droup during testing 
# update loss each batch - medium 32 - can increase for debugging
# 1 epoc is going through all batches 
################testing eill be the other 20%

# next get the testing data from the csv file

# testing data will be the final column in the matrix without the first row maybe

# can use triple quotation marks to create big comments
# think about having the loops outside of the fucntion when you call it. 

#%%
#class LipDS():
    #def __init__(self): 
    
df = pd.read_csv('C:/Users/color/Documents/Bilodeau_Research_Python/lipo_fp_processed.csv')


# taking 80% of the lipo data and storing it in training_number
training_number = int(0.80*(len(df)))

test_data= df.iloc[training_number:]
train_data= df.iloc[0:training_number]

print(train_data)


#%% 
class LipoNet(nn.Module):
    def __init__(self):
        super(LipoNet, self).__init__()
        # super is saying that class LipoNet is inheriting traits from nn.Module class
        input_size = 1024
        output_size = 1
        self.fc1 = nn.Linear(input_size) #(raw data set) # nums inside nn.Linear() are wrong
        # nn.Linear(size of dataset, )
        x = F.relu(x) # logical equivalnet of- keeps the two layers from being linearly related 
        self.fc2 = nn.Linear(output_size)
        x = F.relu(x) #keeps the two layers from being lineraly related
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        op = F.log_softmax(x, dim=1)
        # output (op) will have the same dimensions as the target output (the ground truth?)
        return op

model = LipoNet().to(device)
print(model)
batch_size = 32

# creating the dataset
# 80% is training 
# other 20% is the testing data





#%%
def train(model, device, optim, epoch):
    model.train() #
    # go through each batch for loop 
    # 
    # 
    # 
    # 
# epoch will then run the train function each iteration is  


