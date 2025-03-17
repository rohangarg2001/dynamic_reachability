import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset

class EnvironmentDataset(Dataset):
    """
    NOTE : In this dataset, we need to store the reduced state X_r, LiDAR input o_e, V(X_r), delV(X_r)
    """
    def __init__(self, X_r, O_e, V_val, del_V):        
        self.X_r = X_r          # (batch_size, 3)
        self.O_e = O_e          # (batch_size, 100)
        self.V_val = V_val              # (batch_size, 1)
        self.del_V = del_V      # (batch_size, 3)
    
    def __len__(self):
        return self.X_r.shape[0]
    
    def __getitem__(self, index):
        X_r = self.X_r[index, :]
        O_e = self.O_e[index, :]
        V_val = self.V_val[index]
        del_V = self.del_V[index, :]

        return {"X_r": torch.tensor(X_r), "O_e": torch.tensor(O_e), "V_val": torch.tensor(V_val), "del_V": torch.tensor(del_V)}
    

