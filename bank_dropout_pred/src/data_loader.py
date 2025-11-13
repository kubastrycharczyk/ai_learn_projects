import config 
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import pandas as pd
import numpy as np
import importlib
importlib.reload(config)


df = pd.DataFrame(pd.read_csv(config.path))
X_df = df[(config.x_labels)]
y_df = df[(config.y_labels)]
X_df = pd.get_dummies(X_df, columns=config.x_labels_cat, drop_first=True)
if X_df.isnull().values.any():
    print("WARNING: Missing values found. Imputing with the mean.")
X_np = X_df.values.astype(np.float32)
y_np = y_df.values.astype(np.float32)
X_tesnor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32).squeeze()
y_tensor = y_tensor.unsqueeze(1)

data = TensorDataset(X_tesnor, y_tensor)
data_len = len(data)
train_set, val_set, test_set = random_split(
    data, [int(config.split_size[0]*data_len),int(config.split_size[1]*data_len),int(config.split_size[2]*data_len)], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)


