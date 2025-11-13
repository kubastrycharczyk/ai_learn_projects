import config 
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import importlib
importlib.reload(config)


df = pd.DataFrame(pd.read_csv(config.path))
X_df = df(config.x_labels)
y_df = df(config.y_labels)
X_df = pd.get_dummies(X_df, columns=config.x_labels_cat, drop_first=True)
X_np = X_df.values
y_np = y_df.values
X_tesnor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
data = (X_tesnor, y_tensor)
loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)

