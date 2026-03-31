"""
Extract raw sigmoid probabilities from a frozen seizure detector for all
data splits. The resulting .npz files are used to train the ScoreNet
learned postprocessor.

Usage:
    python extract_probs.py \
        --data_path ./datasets/TUSZ \
        --checkpoint ./checkpoints/best.pth \
        --output_dir ./scorenet_data \
        --dataset TUSZ
"""

import argparse
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm

from evaluate_checkpoint import load_model, get_ch_names_for_dataset, CHBMITDataset
import utils


class H5DatasetWithPIDs(Dataset):
    """HDF5 dataset that returns (data, label, patient_id)."""

    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = len(self.h5_file['labels'])
        self.has_pids = 'patient_ids' in self.h5_file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.from_numpy(self.h5_file['data'][idx]).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=1e4, neginf=-1e4)
        label = int(self.h5_file['labels'][idx])
        pid = int(self.h5_file['patient_ids'][idx]) if self.has_pids else -1
        return data, label, pid


@torch.no_grad()
def extract_split(model, h5_path, input_chans, device, batch_size):
    """Run the frozen model on one split and return probs, labels, pids."""
    dset = H5DatasetWithPIDs(h5_path)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs, all_labels, all_pids = [], [], []
    for data, labels, pids in tqdm(loader, desc=os.path.basename(h5_path)):
        data = data.to(device) / 100.0
        data = rearrange(data, 'B N (A T) -> B N A T', T=200)
        with torch.cuda.amp.autocast():
            logits = model(data, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_pids.append(pids.numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_pids),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Extract probability sequences from a frozen detector')
    parser.add_argument('--data_path', required=True, type=str,
                        help='Directory containing train.h5, val.h5, test.h5')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--output_dir', default='./scorenet_data', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='TUSZ', type=str,
                        help='Dataset for channel names: CHBMIT | TUSZ')
    parser.add_argument('--adversarial', action='store_true',
                        help='Load an adversarial checkpoint')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args, device)

    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    splits = [
        ('train.h5', 'train'),
        ('val.h5', 'val'),
        ('test.h5', 'test'),
    ]

    for filename, name in splits:
        h5_path = os.path.join(args.data_path, filename)
        if not os.path.isfile(h5_path):
            print(f"Skipping {h5_path} (not found)")
            continue

        print(f"\nExtracting {name} split ...")
        probs, labels, pids = extract_split(
            model, h5_path, input_chans, device, args.batch_size)

        out_path = os.path.join(args.output_dir, f'{name}.npz')
        np.savez_compressed(out_path, probs=probs, labels=labels, patient_ids=pids)
        n_seiz = int(labels.sum())
        print(f"  Saved {out_path}  ({len(labels)} windows, "
              f"{n_seiz} seizure [{n_seiz/len(labels)*100:.2f}%])")

    print("\nDone.")


if __name__ == '__main__':
    main()
