import torch
import torch.nn as nn
import config
import data_loader

# Version: 0.1 

class Classifier(nn.Module):
    def __init__(self, in_num):
        super().__init__()
        self.linear1 = nn.Linear(in_num, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Sigmoid()
    
    def forward(self, x):
        x=self.linear1(x)
        x=self.act1(x)
        x=self.linear2(x)
        x=self.act2(x)
        x=self.linear3(x)
        x=self.act3(x)
        return x

    

    