import argparse
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
import itertools
import pandas as pd

import utils
from evaluate_checkpoint import (
    CHBMITDataset, post_process_probs, get_ch_names_for_dataset, load_model,
    load_scorenet, scorenet_postprocess,
)


def compute_epoch_metrics(y_true, y_pred, fs=1, epoch_sec=10):
    n_samples = len(y_true)
    samples_per_epoch = int(fs * epoch_sec)
    n_epochs = n_samples // samples_per_epoch

    y_true_trunc = y_true[:n_epochs * samples_per_epoch]
    y_pred_trunc = y_pred[:n_epochs * samples_per_epoch]

    y_true_epoch = y_true_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    y_pred_epoch = y_pred_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)

    return y_true_epoch, y_pred_epoch


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    model = load_model(args, device)

    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

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

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"AUROC: {roc_auc}  |  AUPRC: {pr_auc}")

    # grid search over post-processing parameters
    grid = {
        't_high': [0.3, 0.4, 0.5, 0.6, 0.7],
        't_low':  [0.2, 0.3, 0.4],
        'smooth': [5, 10, 30],
        'min_dur': [0, 5, 7, 10]
    }

    best_params = {}
    results = []

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for p in tqdm(combinations, desc="Grid search"):
        if p['t_low'] >= p['t_high']:
            continue
        y_pred_pp = post_process_probs(
            y_prob, p['t_high'], p['t_low'], p['smooth'], p['min_dur'])
        yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_pp, epoch_sec=10)

        tn, fp, fn, tp = confusion_matrix(yt_ep, yp_ep, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(yt_ep, yp_ep, zero_division=0)

        res = {**p, 'f1': f1, 'sens': sens, 'spec': spec}
        results.append(res)


    df = pd.DataFrame(results)
    print("top 5 configurations by f1:")
    print(df.sort_values(by='f1', ascending=False).head(5).to_string(index=False))

    if args.scorenet_checkpoint:
        sn_model = load_scorenet(args.scorenet_checkpoint, device)
        sn_thresholds = [0.3, 0.4, 0.5, 0.6]
        sn_results = []
        for thr in sn_thresholds:
            y_pred_sn, _ = scorenet_postprocess(
                y_prob, sn_model, device,
                threshold=thr, min_dur_sec=args.sn_min_dur)
            yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_sn, epoch_sec=10)
            tn, fp, fn, tp = confusion_matrix(yt_ep, yp_ep, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(yt_ep, yp_ep, zero_division=0)
            sn_results.append({'threshold': thr, 'f1': f1, 'sens': sens, 'spec': spec})

        sn_df = pd.DataFrame(sn_results)
        print("\nScoreNet results by threshold:")
        print(sn_df.sort_values(by='f1', ascending=False).to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/CHBMIT', type=str)
    parser.add_argument('--checkpoint',
                        default='./checkpoints/finetune_chbmit_v1/checkpoint-19.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str)
    parser.add_argument('--adversarial', action='store_true')

    parser.add_argument('--scorenet_checkpoint', default=None, type=str)
    parser.add_argument('--sn_min_dur', default=10, type=int)

    args = parser.parse_args()
    main(args)
