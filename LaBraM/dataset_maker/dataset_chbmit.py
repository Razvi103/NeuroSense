import bisect
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
        data = self.h5_file['data'][idx]
        label = self.h5_file['labels'][idx]
        
        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        return data, label


class CHBMITAdversarialDataset(Dataset):

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


class TUSZAdversarialDataset(Dataset):
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


class MultiPatientAdversarialDataset(Dataset):

    def __init__(self, h5_paths):
        self.h5_files = []
        self.lengths = []
        self._cum_lengths = []
        all_patient_ids = set()

        cumulative = 0
        for path in h5_paths:
            f = h5py.File(path, 'r')
            n = len(f['labels'])
            self.h5_files.append(f)
            self.lengths.append(n)
            cumulative += n
            self._cum_lengths.append(cumulative)

            if 'patient_ids' in f and n > 0:
                pid_val = int(f['patient_ids'][0])
                all_patient_ids.add(pid_val)

        self.length = cumulative
        self.num_patients = max(all_patient_ids) + 1 if all_patient_ids else 1

    def _locate(self, idx):
        file_idx = bisect.bisect_right(self._cum_lengths, idx)
        local_idx = idx if file_idx == 0 else idx - self._cum_lengths[file_idx - 1]
        return file_idx, local_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        f = self.h5_files[file_idx]

        data = f['data'][local_idx]
        label = f['labels'][local_idx]
        patient_id = int(f['patient_ids'][local_idx])

        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=10000.0, neginf=-10000.0)

        return data, label, patient_id

    def close(self):
        for f in self.h5_files:
            f.close()