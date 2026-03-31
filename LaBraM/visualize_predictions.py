import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
import utils
import modeling_finetune
from evaluate_checkpoint import (
    CHBMITDataset, post_process_probs, get_ch_names_for_dataset,
    load_model, load_scorenet, scorenet_postprocess,
)


@torch.no_grad()
def visualize(args):
    device = torch.device(args.device)

    print("Loading Model & Checkpoint...")
    model = load_model(args, device)

    ch_names_raw = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names_raw)

    print("Running Inference on Test Set...")
    dataset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=16)

    all_probs = []
    all_targets = []

    for samples, targets in tqdm(loader):
        samples = samples.to(device)
        samples = samples / 100.0
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        with torch.amp.autocast('cuda'):
            output = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(output).squeeze()
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)

    if args.smooth > 1:
        y_smooth = pd.Series(y_prob).rolling(
            window=args.smooth, center=True).mean().fillna(0).values
    else:
        y_smooth = y_prob

    # --- Post-processing ---
    y_pred_ht = post_process_probs(
        y_prob, args.t_high, args.t_low, args.smooth, args.min_dur)

    use_scorenet = args.scorenet_checkpoint is not None
    y_pred_sn = None
    y_refined_sn = None
    if use_scorenet:
        sn_model = load_scorenet(args.scorenet_checkpoint, device)
        y_pred_sn, y_refined_sn = scorenet_postprocess(
            y_prob, sn_model, device,
            threshold=args.sn_threshold, min_dur_sec=args.sn_min_dur)

    # --- Find seizure events ---
    diffs = np.diff(np.concatenate(([0], y_true, [0])))
    seizure_starts = np.where(diffs == 1)[0]

    print(f"\nFound {len(seizure_starts)} seizures in the test set.")
    if len(seizure_starts) == 0:
        print("No seizures found in ground truth. Skipping plots.")
        return

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    num_to_plot = min(args.num_plots, len(seizure_starts))
    print(f"Generating {num_to_plot} plots ...")

    for i in range(num_to_plot):
        start_idx = seizure_starts[i]
        margin = args.margin
        plot_start = max(0, start_idx - margin)
        offset = 0
        while start_idx + offset < len(y_true) and y_true[start_idx + offset] == 1:
            offset += 1
        seizure_end = start_idx + offset
        plot_end = min(len(y_true), seizure_end + margin)

        t = np.arange(plot_start, plot_end)
        s = slice(plot_start, plot_end)

        n_rows = 2 if use_scorenet else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 5 * n_rows),
                                 sharex=True, squeeze=False)

        # ---- Row 1: Hand-tuned ----
        ax = axes[0, 0]
        ax.fill_between(t, 0, 1, where=(y_true[s] == 1),
                        color='green', alpha=0.25, label='Ground Truth')
        ax.plot(t, y_prob[s], color='blue', alpha=0.3, lw=1, label='Raw Prob')
        ax.plot(t, y_smooth[s], color='darkorange', lw=2, label=f'Smoothed ({args.smooth}s)')
        ax.axhline(y=args.t_high, color='red', ls='--', alpha=0.6,
                   label=f'High Thresh ({args.t_high})')
        ax.axhline(y=args.t_low, color='darkred', ls=':', alpha=0.6,
                   label=f'Low Thresh ({args.t_low})')
        ax.fill_between(t, 0, -0.04, where=(y_pred_ht[s] == 1),
                        color='red', alpha=0.7, label='Hand-Tuned Alarm')
        ax.set_ylim(-0.08, 1.05)
        ax.set_ylabel('Probability')
        ax.set_title(f'Seizure #{i+1} — Hand-Tuned Post-Processing')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # ---- Row 2: ScoreNet ----
        if use_scorenet:
            ax2 = axes[1, 0]
            ax2.fill_between(t, 0, 1, where=(y_true[s] == 1),
                             color='green', alpha=0.25, label='Ground Truth')
            ax2.plot(t, y_prob[s], color='blue', alpha=0.3, lw=1, label='Raw Prob')
            ax2.plot(t, y_refined_sn[s], color='purple', lw=2,
                     label='ScoreNet Refined Prob')
            ax2.axhline(y=args.sn_threshold, color='purple', ls='--', alpha=0.6,
                        label=f'SN Thresh ({args.sn_threshold})')
            ax2.fill_between(t, 0, -0.04, where=(y_pred_sn[s] == 1),
                             color='purple', alpha=0.7, label='ScoreNet Alarm')
            ax2.set_ylim(-0.08, 1.05)
            ax2.set_ylabel('Probability')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title(f'Seizure #{i+1} — ScoreNet Post-Processing')
            ax2.legend(loc='upper right', fontsize=8, ncol=2)
            ax2.grid(True, alpha=0.3)
        else:
            axes[0, 0].set_xlabel('Time (seconds)')

        plt.tight_layout()
        filename = os.path.join(out_dir, f'seizure_viz_{i+1}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved {filename}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint',
                        default='./checkpoints/finetune_chbmit_v1/checkpoint-9.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str,
                        help='Dataset for channel names: CHBMIT | TUSZ')
    parser.add_argument('--adversarial', action='store_true',
                        help='Load an adversarial (GRL+Attention) checkpoint')

    # Hand-tuned post-processing
    parser.add_argument('--t_high', default=0.4, type=float)
    parser.add_argument('--t_low', default=0.2, type=float)
    parser.add_argument('--smooth', default=5, type=int)
    parser.add_argument('--min_dur', default=5, type=int)

    # ScoreNet post-processing (optional)
    parser.add_argument('--scorenet_checkpoint', default=None, type=str,
                        help='Path to trained ScoreNet .pth; adds comparison row')
    parser.add_argument('--sn_threshold', default=0.5, type=float)
    parser.add_argument('--sn_min_dur', default=10, type=int)

    # Plot settings
    parser.add_argument('--num_plots', default=5, type=int,
                        help='Max number of seizure events to plot')
    parser.add_argument('--margin', default=180, type=int,
                        help='Seconds of context before/after each seizure')
    parser.add_argument('--output_dir', default='.', type=str,
                        help='Directory to save plot images')

    args = parser.parse_args()
    visualize(args)