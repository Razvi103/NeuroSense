"""
Compute per-recording AUROC on the TUSZ test set, replicating the
evaluation methodology from Wu et al. (2025).

For each continuous EEG recording, AUROC is computed from raw sigmoid
probabilities vs ground-truth labels. Recordings with only one class
(all-background or all-seizure) are excluded. Reports mean +/- std and
optionally generates a violin/density plot.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def load_recording_ids(data_path):
    """Load recording_ids from test.h5."""
    h5_path = os.path.join(data_path, 'test.h5')
    with h5py.File(h5_path, 'r') as f:
        if 'recording_ids' not in f:
            print("ERROR: test.h5 does not contain 'recording_ids'. "
                  "Regenerate with the updated make_TUSZ.py.", file=sys.stderr)
            sys.exit(1)
        recording_ids = f['recording_ids'][:]
        labels = f['labels'][:]
    return recording_ids, labels


def extract_probs_from_checkpoint(args, data_path):
    """Run inference with a checkpoint to get probabilities and labels."""
    import torch
    from torch.utils.data import DataLoader
    from einops import rearrange
    from tqdm import tqdm
    from evaluate_checkpoint import load_model, get_ch_names_for_dataset, CHBMITDataset
    import utils

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
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(targets.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_per_recording_auroc(y_prob, y_true, recording_ids):
    """Compute AUROC for each unique recording.
    """
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


def plot_violin(all_aurocs, labels, output_path):
    """Generate a violin plot of AUROC distributions."""

    fig, ax = plt.subplots(figsize=(max(3 * len(all_aurocs), 6), 5))

    parts = ax.violinplot(
        all_aurocs, positions=range(len(all_aurocs)),
        showmeans=True, showmedians=True, showextrema=False,
    )

    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linestyle('--')

    for i, aurocs in enumerate(all_aurocs):
        mean_val = np.mean(aurocs)
        ax.text(i, 1.03, f"Mean: {mean_val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(all_aurocs)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("AUROC", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Per-recording AUROC evaluation (Wu et al. 2025 style)')
    parser.add_argument('--data_path', required=True,
                        help='Directory containing test.h5 with recording_ids')
    parser.add_argument('--cached_probs', nargs='+', default=None,
                        help='Path(s) to .npz file(s) with probs and labels arrays')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Display labels for each cached_probs entry (for plot)')

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
    n_recordings = len(np.unique(recording_ids))
    print(f"Test set: {len(h5_labels)} windows, {n_recordings} recordings")

    all_aurocs = []
    display_labels = []

    if args.cached_probs:
        for i, npz_path in enumerate(args.cached_probs):
            print(f"\nLoading {npz_path} ...")
            data = np.load(npz_path)
            y_prob = data['probs']
            y_true = data['labels'] if 'labels' in data else h5_labels

            if len(y_prob) != len(recording_ids):
                print(f"  WARNING: probs length ({len(y_prob)}) != "
                      f"recording_ids length ({len(recording_ids)}). "
                      "Ensure the H5 and probs come from the same test set.",
                      file=sys.stderr)

            aurocs, n_skip, _ = compute_per_recording_auroc(
                y_prob, y_true, recording_ids)

            label = (args.labels[i] if args.labels and i < len(args.labels)
                     else os.path.basename(npz_path))
            all_aurocs.append(aurocs)
            display_labels.append(label)

            print(f"  [{label}] {len(aurocs)} recordings scored "
                  f"({n_skip} skipped, single-class)")
            print(f"  Mean AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
            print(f"  Median: {np.median(aurocs):.4f}  "
                  f"Min: {np.min(aurocs):.4f}  Max: {np.max(aurocs):.4f}")
    else:
        print(f"\nRunning inference from {args.checkpoint} ...")
        y_prob, y_true = extract_probs_from_checkpoint(args, args.data_path)

        aurocs, n_skip, _ = compute_per_recording_auroc(
            y_prob, y_true, recording_ids)

        label = os.path.basename(os.path.dirname(args.checkpoint))
        all_aurocs.append(aurocs)
        display_labels.append(label)

        print(f"\n  {len(aurocs)} recordings scored "
              f"({n_skip} skipped, single-class)")
        print(f"  Mean AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
        print(f"  Median: {np.median(aurocs):.4f}  "
              f"Min: {np.min(aurocs):.4f}  Max: {np.max(aurocs):.4f}")

    if args.plot:
        plot_violin(all_aurocs, display_labels, args.plot)


if __name__ == '__main__':
    main()
