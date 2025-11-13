import torch
import torch.nn as nn
import config
import data_loader

class Classifier(nn.Module):
    def __init__(self, in_num):
        super(self, in_num).__init__()
        self.linear1 = nn.Linear()
        self.linear2 = nn.Linear()
        self.linear3 = nn.Linear()
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

    

    