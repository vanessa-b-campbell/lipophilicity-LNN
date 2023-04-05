import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from LipoNet import LipoNet # pretty sure this is importing my LipoNet class
from LipDS import LipDS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dataset = LipDS('C:/Users/color/Documents/Bilodeau_Research_Python/lipo_fp_processed.csv')
print(len(dataset))
print(dataset.input_vector[55])
print(dataset.output_targets[55])
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
# class LipDS():
#     def __init__(self,path): 

#         self.df = pd.read_csv(path)


# # taking 80% of the lipo data and storing it in training_number
#     training_number = int(0.80*(len(self.df)))

#     test_data= df.iloc[training_number:]
#     train_data= df.iloc[0:training_number]

#     print(train_data)




#%% 
# class LipoNet(nn.Module):
#     def __init__(self):
#         super(LipoNet, self).__init__()
#         # super is saying that class LipoNet is inheriting traits from nn.Module class
#         input_size = 1024
#         output_size = 1
#         self.fc1 = nn.Linear(input_size, output_size) #(raw data set) # nums inside nn.Linear() are wrong
#         # nn.Linear(size of dataset, size of output)
#         self.fc2 = nn.Linear(input_size, output_size)
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
#         # output (op) will have the same dimensions as the target output (the ground truth?)
#         return out #op

# model = LipoNet().to(device)
# print(model)
# batch_size = 32



# #%%
# #def train(model, device, optim, epoch):
# model.train() 

# #we need to be able to compute the gradients of loss function with respect 
# # to those variables. In order to do that, we set the requires_grad property of those tensors.

# x = torch.ones(1024)
# y = torch.zeros(1)

# # w and b parameters to optimize
# w = torch.randn(1024, 1, requires_grad=True) 
# b = torch.randn(1, requires_grad=True)
# z = torch.matmul(x,w) +b
# # compute the gradients of loss function with respect to those variables
# loss_func = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# #loss_collect = 0;

# print(f"Gradient function for z = {z.grad_fn}")
# print(f"Gradient function for loss = {loss_func.grad_fn}")

# loss_func.backward()
# print(w.grad)
# print(b.grad)



# go through each batch for loop 
# 
# 
# 
# 
# epoch will then run the train function each iteration is  



# %%
