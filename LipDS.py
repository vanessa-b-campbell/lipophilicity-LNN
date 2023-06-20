import pandas as pd
import torch
from torch.utils.data import DataLoader

# 1. Defines the dataset as a class 
# 2. organizes the data into test and train
# 3. seperates the training data into input vectors and output targets
# 4. creates 2 attributes: 
#           -output the number of samples
#           -display the input vector of a given index and disply the output value at a given index

class LipDS(DataLoader):
    def __init__(self,path): 

        # read a csv from that path:
        self.df = pd.read_csv(path)
        


# taking 80% of the lipo data and storing it in training_number
        training_number = int(0.80*(len(self.df)))

        # a suprise tool that will help us later (this is the testing data to compare to the model's predicted output)
        self.test_data= self.df.iloc[training_number:]
        
        
        # training data (80% of the given dataset)
        self.train_data= self.df.iloc[0:training_number]
        
        # taking all the columns except the very last one and assigning it to input vector
        # (this is our x)
        self.input_vector = self.test_data[self.test_data.columns[0:-1]].values

        # very last column is the output target (the y)
        self.output_targets = self.test_data[self.test_data.columns[-1]].values

        #print(self.output_targets)

    # create an attribute that will output the number of samples
    def __len__(self):
        return len(self.output_targets)
    
    # attribute that will display the input vector of a given index (input is a mxn vector)
    # will also disply the output value at a given index (output is a 1xn vector)
    def __getitem__(self, idx):
        x = self.input_vector[idx]
        y = self.output_targets[idx]

        return torch.tensor(x, dtype = torch.float32),torch.tensor(y, dtype = torch.float32)




# testing to make sure the object creates instances correctly

# [4200 rows x 1025 columns]
set = LipDS('C:/Users/color/Documents/Bilodeau_Research_Python/lipo_fp_processed.csv')
print(set)
print(len(set))
print(set.input_vector[55])
print(set.output_targets[55])