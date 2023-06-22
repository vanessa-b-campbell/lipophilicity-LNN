# training function: for epoch in range- train the model- calculate the loss 
#%%
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#%%
from Lip_data import LipDS
from LipoNet import LipoNet
from Lipo_process import train, validation

## SET UP DATALOADERS: ---

#%%


lipo_dataset = LipDS('C:/Users/color/Documents/Bilodeau_Research_Python/lipo_fp_processed.csv')
print(lipo_dataset)
print(lipo_dataset.input_vector[55])
#%%
d_train = int(len(lipo_dataset)* 0.8)
d_val = len(lipo_dataset) - d_train

# Define pytorch training and validation set objects:
# also random seeded split
train_set, val_set = torch.utils.data.random_split(
    lipo_dataset, [d_train, d_val], generator=torch.Generator().manual_seed(42)
)
print(train_set)




#%%
# Build pytorch training and validation set dataloaders:
train_dataloader = DataLoader(train_set, batch_size = 32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size = 32, shuffle=True)




#%%
## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Assign training to a device (often cpu when we are just starting out)
device = torch.device("cpu")


model = LipoNet()
model.to(device)

# Set up optimizer:
optimizer = optim.Adam(model.parameters(), lr=0.1) # learning rate ex: 1*10^-3

train_losses = []
val_losses = []

start_time = time.time()

#%%
for epoch in range(1, 10):
    
    train_loss = train(model, device, train_dataloader, optimizer)
    train_losses.append(train_loss)
#%%
    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)

end_time = time.time()
print("Time Elapsed = {}s".format(end_time - start_time))
# %%
