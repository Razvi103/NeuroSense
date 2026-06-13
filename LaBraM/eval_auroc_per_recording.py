import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from evaluate_checkpoint import load_model, get_ch_names_for_dataset, CHBMITDataset
import utils


def load_recording_ids(data_path):
    h5_path = os.path.join(data_path, 'test.h5')
    with h5py.File(h5_path, 'r') as f:
        recording_ids = f['recording_ids'][:]
        labels = f['labels'][:]
    return recording_ids, labels


def extract_probs_from_checkpoint(args, data_path):
    device = torch.device(args.device)
    model = load_model(args, device)

    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    h5_path = os.path.join(data_path, 'test.h5')
    dset = CHBMITDataset(h5_path)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    all_probs, all_labels = [], []
    for samples, targets in tqdm(loader, desc="Inference"):
        samples = samples.to(device) / 100.0
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(targets.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_per_recording_auroc(y_prob, y_true, recording_ids):

    unique_rids = np.unique(recording_ids)
    aurocs = []
    rec_labels = []
    n_skipped = 0

    for rid in unique_rids:
        mask = recording_ids == rid
        y_t = y_true[mask]
        y_p = y_prob[mask]
        if len(np.unique(y_t)) < 2:
            n_skipped += 1
            continue
        aurocs.append(roc_auc_score(y_t, y_p))
        rec_labels.append(rid)

    return np.array(aurocs), n_skipped, np.array(rec_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--cached_probs', nargs='+', default=None)
    parser.add_argument('--labels', nargs='+', default=None)

    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint (if not using cached_probs)')
    parser.add_argument('--model', default='labram_base_patch200_200')
    parser.add_argument('--dataset', default='TUSZ')
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--plot', default='', help='Output path for violin plot (PDF/PNG)')
    args = parser.parse_args()

    if args.cached_probs is None and args.checkpoint is None:
        parser.error("Provide either --cached_probs or --checkpoint")

    recording_ids, h5_labels = load_recording_ids(args.data_path)

    all_aurocs = []
    display_labels = []

    if args.cached_probs:
        for i, npz_path in enumerate(args.cached_probs):
            data = np.load(npz_path)
            y_prob = data['probs']
            y_true = data['labels'] if 'labels' in data else h5_labels

            aurocs, n_skip, _ = compute_per_recording_auroc(
                y_prob, y_true, recording_ids)

            label = (args.labels[i] if args.labels and i < len(args.labels)
                     else os.path.basename(npz_path))
            all_aurocs.append(aurocs)
            display_labels.append(label)


            print(f"mean AUROC: {np.mean(aurocs)} +/- {np.std(aurocs)}")
            print(f"median: {np.median(aurocs):}  "
                  f"min: {np.min(aurocs)}  max: {np.max(aurocs)}")
    else:
        y_prob, y_true = extract_probs_from_checkpoint(args, args.data_path)

        aurocs, n_skip, _ = compute_per_recording_auroc(
            y_prob, y_true, recording_ids)

        label = os.path.basename(os.path.dirname(args.checkpoint))
        all_aurocs.append(aurocs)
        display_labels.append(label)

        print(f"mean AUROC: {np.mean(aurocs)} +/- {np.std(aurocs)}")
        print(f"median: {np.median(aurocs)}  "
              f"min: {np.min(aurocs)}  max: {np.max(aurocs)}")



if __name__ == '__main__':
    main()
