# training function: for epoch in range- train the model- calculate the loss 
#%%
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#%%
from Lip_data import LipDS
from LipoNet import LipoNet
from Lipo_process import train, validation, predict

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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
    lipo_dataset, [d_train, d_val], generator=torch.Generator().manual_seed(0)
)
print(train_set)



# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Assign training to a device (often cpu when we are just starting out)
device = torch.device("cpu")
#################################################################################### hyperparameters


model = LipoNet(800, 500, 256)
model.to(device)
epoch = 100
learn_rate = 0.001
batch_size = 32




#%%
# Build pytorch training and validation set dataloaders:
train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)


#%%
## RUN TRAINING LOOP: ---

# Set up optimizer:
optimizer = optim.Adam(model.parameters(), lr = learn_rate) # learning rate ex: 1*10^-3

train_losses = []
val_losses = []

start_time = time.time()



for e in range(1,epoch+1):
    
    train_loss = train(model, device, train_dataloader, optimizer, e)
    train_losses.append(train_loss)
    
    val_loss = validation(model, device, val_dataloader, e)
    val_losses.append(val_loss)

end_time = time.time()
print("Time Elapsed = {}s".format(end_time - start_time))
# %%


#%% Model Statistics

input_all, target_all, pred_prob_all = predict(model, device, val_dataloader)

# print("input_all: {i}".format(i = input_all))
# print("target_all: {t}".format(t = target_all))
# print("pred_prob_all: {p}".format(p = pred_prob_all))

r2_function = r2_score(target_all, pred_prob_all)
mae = mean_absolute_error(target_all, pred_prob_all)
rmse = mean_squared_error(target_all, pred_prob_all, squared=False)

# only a few digits are relevant
print("R2 Score: {:.4f}".format(r2_function))
print("MAE: {:.4f}".format(mae))
print("RMSE: {:.4f}".format(rmse))

# plotting loss vs epochs for validation and training
#fig1 = plt.figure()
plt.plot(train_losses, label ='train losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('train losses')

#fig2 = plt.figure()
plt.plot(val_losses, label ='validation losses')
plt.legend()
plt.xlabel('time')
plt.ylabel('losses')

plt.show()


plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all, pred_prob_all, alpha=0.3)
plt.plot([min(target_all), max(target_all)], [min(target_all),
    max(target_all)], color="k", ls="--")
plt.xlim([min(target_all), max(target_all)])
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.show()
# %%
