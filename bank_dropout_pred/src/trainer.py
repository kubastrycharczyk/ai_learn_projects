import torch
import torch.nn as nn
import torch.optim as optim



class Trainer():
    def __init__(self, model, opt , loss_f = nn.BCELoss(), epoch = 10):
        self.model = model
        self.loss_f = loss_f
        self.opt = opt
        self.epoch = epoch