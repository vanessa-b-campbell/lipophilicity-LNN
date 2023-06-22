import torch
import torch.nn as nn
import torch.nn.functional as F


# defines the model structure
# dataset is [4200 rows x 1025 columns]
# training set input vector is 1024 columns
# training set outout targets is a 1 column
class LipoNet(nn.Module):
    def __init__(self):
        super(LipoNet, self).__init__()
        # super is saying that class LipoNet is inheriting traits from nn.Module class
        input_size = 1024 
        output_size = 1
        hidden_layer_size = 256

        # two linear layers
        self.fc1 = nn.Linear(input_size, hidden_layer_size) #(raw training dataset)
        # nn.Linear(size of dataset, size of output)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output (op) will have the same dimensions as the target output (1)
        return x #op
    

# testing 
model = LipoNet()
print(model)