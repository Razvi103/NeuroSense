import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class CHBMITDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = len(self.h5_file['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Read data directly from disk (low RAM usage)
        # Shape is (Channels, Time) -> (23, 400)
        data = self.h5_file['data'][idx]
        label = self.h5_file['labels'][idx]
        
        # Convert to Tensor
        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        return data, label