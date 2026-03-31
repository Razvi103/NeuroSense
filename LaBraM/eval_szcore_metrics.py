import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from einops import rearrange
from evaluate_checkpoint import (
    CHBMITDataset, post_process_probs, get_ch_names_for_dataset,
    load_model, load_scorenet, scorenet_postprocess,
)


def compute_epoch_metrics(y_true, y_pred, fs=1, epoch_sec=10):
    """Divides the timeline into fixed epoch windows and applies max pooling."""
    n_samples = len(y_true)
    samples_per_epoch = int(fs * epoch_sec)
    n_epochs = n_samples // samples_per_epoch

    y_true_trunc = y_true[:n_epochs * samples_per_epoch]
    y_pred_trunc = y_pred[:n_epochs * samples_per_epoch]

    y_true_epoch = y_true_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    y_pred_epoch = y_pred_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)

    return y_true_epoch, y_pred_epoch


def print_epoch_results(tag, yt_epoch, yp_epoch, epoch_sec):
    epoch_f1 = f1_score(yt_epoch, yp_epoch, zero_division=0)
    kappa = cohen_kappa_score(yt_epoch, yp_epoch)
    tn, fp, fn, tp = confusion_matrix(yt_epoch, yp_epoch, labels=[0, 1]).ravel()

    print(f"\n--- Epoch-Based Evaluation ({tag}, {epoch_sec}s epochs) ---")
    print(f"  F1 Score:    {epoch_f1:.4f}")
    print(f"  Cohen Kappa: {kappa:.4f}")
    print(f"  Specificity: {tn / (tn + fp):.4f}" if (tn + fp) > 0 else "  Specificity: N/A")
    print(f"  Sensitivity: {tp / (tp + fn):.4f}" if (tp + fn) > 0 else "  Sensitivity: N/A")
    print(f"  Confusion:   [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")


@torch.no_grad()
def run_rigorous_eval(args):
    device = torch.device(args.device)

    print(f"Loading Model: {args.model}")
    model = load_model(args, device)

    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    print("Loading Test Set...")
    dataset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8)

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

    # --- Hand-tuned postprocessing ---
    y_pred_ht = post_process_probs(
        y_prob, t_high=args.t_high, t_low=args.t_low,
        smooth_window=args.smooth, min_duration=args.min_dur)

    for es in args.epoch_secs:
        yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_ht, epoch_sec=es)
        print_epoch_results("Hand-Tuned", yt_ep, yp_ep, es)

    # --- ScoreNet postprocessing ---
    if args.scorenet_checkpoint:
        sn_model = load_scorenet(args.scorenet_checkpoint, device)
        y_pred_sn, _ = scorenet_postprocess(
            y_prob, sn_model, device,
            threshold=args.sn_threshold, min_dur_sec=args.sn_min_dur)

        for es in args.epoch_secs:
            yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_sn, epoch_sec=es)
            print_epoch_results("ScoreNet", yt_ep, yp_ep, es)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-9.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str,
                        help='Dataset for channel names: CHBMIT | TUSZ')
    parser.add_argument('--adversarial', action='store_true',
                        help='Load an adversarial (GRL+Attention) checkpoint')

    # Hand-tuned post-processing parameters
    parser.add_argument('--t_high', default=0.40, type=float)
    parser.add_argument('--t_low', default=0.20, type=float)
    parser.add_argument('--smooth', default=5, type=int)
    parser.add_argument('--min_dur', default=5, type=int)

    # ScoreNet post-processing (optional)
    parser.add_argument('--scorenet_checkpoint', default=None, type=str,
                        help='Path to trained ScoreNet .pth; adds learned postprocessing')
    parser.add_argument('--sn_threshold', default=0.5, type=float)
    parser.add_argument('--sn_min_dur', default=10, type=int)

    # Epoch sizes to evaluate
    parser.add_argument('--epoch_secs', nargs='+', type=int, default=[5, 10],
                        help='Epoch durations in seconds (can specify multiple)')

    args = parser.parse_args()
    run_rigorous_eval(args)