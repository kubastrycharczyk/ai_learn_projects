import torch
import torch.nn as nn
import config
import data_loader

# Version: 0.2

class Classifier(nn.Module):
    def __init__(self, in_num):
        super().__init__()
        self.norm0 =  nn.BatchNorm1d(in_num)
        self.linear1 = nn.Linear(in_num, 128)
        self.norm1 =  nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.norm2 =  nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        #self.act3 = nn.Sigmoid()
    
    def forward(self, x):
        x=self.norm0(x)
        x=self.linear1(x)
        x=self.norm1(x)
        x=self.act1(x)
        x=self.linear2(x)
        x=self.norm2(x)
        x=self.act2(x)
        x=self.linear3(x)
        #x=self.act3(x)
        return x

    

    