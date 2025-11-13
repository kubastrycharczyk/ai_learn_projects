import torch
import config
import torch.nn as nn
import torch.optim as optim
import data_loader
import importlib
importlib.reload(config)
importlib.reload(data_loader)




class ClassNet(nn.Module):
    def __init__(self, in_size):
        super(ClassNet, self).__init__()
        self.linear_1 = nn.Linear(in_size, 128)
        self.act1 = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.linear_3 = nn.Linear(64, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act1(x)
        x = self.linear_2(x)
        x = self.act2(x)
        x = self.linear_3(x)
        x = self.act3(x)
        return x
    

model = ClassNet(config.lenght)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)

num_epoki = 5

for epoka in range(num_epoki):
    for wejscia, etykiety in data_loader.loader:
        optimizer.zero_grad()
        wyjscia = model(wejscia)
        strata = loss_func(wyjscia, etykiety)
        strata.backward()
        optimizer.step()
        
    print(f"Epoka [{epoka+1}/{num_epoki}], Strata: {strata.item():.4f}")