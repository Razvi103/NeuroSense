import json
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
        # Shape is (Channels, Time) -> (23, 400)
        data = self.h5_file['data'][idx]
        label = self.h5_file['labels'][idx]
        
        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        return data, label


class TUSZAdversarialDataset(Dataset):
    """
    TUSZ dataset that also returns patient IDs for adversarial training.
    Falls back to patient_id=-1 if the HDF5 lacks the patient_ids field.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = len(self.h5_file['labels'])
        self.has_patient_ids = 'patient_ids' in self.h5_file

        if self.has_patient_ids:
            pids = self.h5_file['patient_ids']
            self.num_patients = int(np.max(pids)) + 1
        else:
            self.num_patients = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.h5_file['data'][idx]
        label = self.h5_file['labels'][idx]
        patient_id = int(self.h5_file['patient_ids'][idx]) if self.has_patient_ids else -1

        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=10000.0, neginf=-10000.0)

        return data, label, patient_id