import pandas as pd
import torch
from torch.utils.data import DataLoader



class LipDS(DataLoader):
    def __init__(self,path): 

        # read a csv from that path:
        self.df = pd.read_csv(path)


# taking 80% of the lipo data and storing it in training_number
        training_number = int(0.80*(len(self.df)))

        # a suprise tool that will help us later
        self.test_data= self.df.iloc[training_number:]
        
        
        
        self.train_data= self.df.iloc[0:training_number]
        
        # pulling the last column (index: -1)
        self.input_vector = self.test_data[self.test_data.columns[0:-1]].values

        self.output_targets = self.test_data[self.test_data.columns[-1]].values

        #print(self.output_targets)

    # create an attribute that will output the number of samples
    def __len__(self):
        return len(self.output_targets)
    
    def __getitem__(self, idx):
        x = self.input_vector[idx]
        y = self.output_targets[idx]

        return torch.tensor(x, dtype = torch.float32),torch.tensor(y, dtype = torch.float32)

