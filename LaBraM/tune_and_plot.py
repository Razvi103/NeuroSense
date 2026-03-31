import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
import itertools
import pandas as pd

import utils
from evaluate_checkpoint import (
    CHBMITDataset, post_process_probs, get_ch_names_for_dataset,
    load_model, load_scorenet, scorenet_postprocess,
    get_events, compute_event_metrics,
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


def eval_predictions(y_true, y_pred, tag, epoch_sec=10):
    """Compute and print point-wise, event, and epoch metrics for one config."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pw_f1 = f1_score(y_true, y_pred, zero_division=0)

    evt = compute_event_metrics(y_true, y_pred)

    yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred, epoch_sec=epoch_sec)
    ep_f1 = f1_score(yt_ep, yp_ep, zero_division=0)
    kappa = cohen_kappa_score(yt_ep, yp_ep)

    return {
        'tag': tag,
        'pw_f1': pw_f1, 'sens': sens, 'spec': spec,
        'evt_f1': evt['F1'], 'evt_prec': evt['Precision'],
        'evt_recall': evt['Recall'], 'far_hr': evt['FAR/hr'],
        'evt_fp': evt['FP'],
        'ep_f1': ep_f1, 'kappa': kappa,
    }


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    print(f"Loading Model: {args.model} from {args.checkpoint}")
    model = load_model(args, device)

    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    print("Loading Test Set...")
    dataset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8)

    print("Running Inference...")
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

    # ---- ROC / PR curves ----
    print("\nGenerating ROC / PR curves...")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall_arr, precision_arr, color='blue', lw=2,
             label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('metrics_curves.png', dpi=150)
    print("Saved plots to 'metrics_curves.png'")

    # ---- Grid search over hand-tuned parameters ----
    print("\n--- Tuning Hand-Tuned Post-Processing Parameters ---")

    grid = {
        't_high': [0.3, 0.4, 0.5, 0.6, 0.7],
        't_low':  [0.2, 0.3, 0.4],
        'smooth': [5, 10, 30],
        'min_dur': [5, 7, 10]
    }

    best_kappa = -1
    best_params = {}
    results = []

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for p in tqdm(combinations, desc="Hand-tuned grid"):
        if p['t_low'] >= p['t_high']:
            continue
        y_pred_pp = post_process_probs(
            y_prob, p['t_high'], p['t_low'], p['smooth'], p['min_dur'])
        yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_pp, epoch_sec=10)

        kappa = cohen_kappa_score(yt_ep, yp_ep)
        tn, fp, fn, tp = confusion_matrix(yt_ep, yp_ep, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(yt_ep, yp_ep, zero_division=0)

        res = {**p, 'kappa': kappa, 'f1': f1, 'sens': sens, 'spec': spec}
        results.append(res)

        if kappa > best_kappa:
            best_kappa = kappa
            best_params = res

    df = pd.DataFrame(results)
    df = df.sort_values(by='kappa', ascending=False)

    print("\nTop 5 Configurations by Kappa:")
    print(df.head(5).to_string(index=False))

    print("\nTop 5 Configurations by F1 Score:")
    print(df.sort_values(by='f1', ascending=False).head(5).to_string(index=False))

    print("-" * 50)
    print(f"Best Hand-Tuned Kappa: {best_params['kappa']:.4f}")
    print(f"  Params: High={best_params['t_high']}, Low={best_params['t_low']}, "
          f"Smooth={best_params['smooth']}, MinDur={best_params['min_dur']}")
    print(f"  Metrics: Sens={best_params['sens']:.4f}, Spec={best_params['spec']:.4f}")

    # ---- ScoreNet comparison ----
    if args.scorenet_checkpoint:
        print("\n--- ScoreNet Post-Processing Comparison ---")
        sn_model = load_scorenet(args.scorenet_checkpoint, device)

        sn_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        sn_results = []
        for thr in sn_thresholds:
            y_pred_sn, _ = scorenet_postprocess(
                y_prob, sn_model, device,
                threshold=thr, min_dur_sec=args.sn_min_dur)
            m = eval_predictions(y_true, y_pred_sn, f"SN thr={thr:.1f}")
            sn_results.append({'threshold': thr, **m})

        # Use best hand-tuned as baseline
        y_pred_best_ht = post_process_probs(
            y_prob, best_params['t_high'], best_params['t_low'],
            best_params['smooth'], best_params['min_dur'])
        ht_m = eval_predictions(y_true, y_pred_best_ht, "Best Hand-Tuned")

        print(f"\n{'Method':<20} {'PW-F1':>6} {'Sens':>6} {'Spec':>6} "
              f"{'Evt-F1':>7} {'Evt-P':>6} {'Evt-R':>6} "
              f"{'FAR/hr':>7} {'Ep-F1':>6} {'Kappa':>6}")
        print("-" * 95)

        def _row(m):
            print(f"{m['tag']:<20} {m['pw_f1']:6.4f} {m['sens']:6.4f} {m['spec']:6.4f} "
                  f"{m['evt_f1']:7.4f} {m['evt_prec']:6.4f} {m['evt_recall']:6.4f} "
                  f"{m['far_hr']:7.2f} {m['ep_f1']:6.4f} {m['kappa']:6.4f}")

        _row(ht_m)
        for r in sn_results:
            _row(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint',
                        default='./checkpoints/finetune_chbmit_v1/checkpoint-19.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str,
                        help='Dataset for channel names: CHBMIT | TUSZ')
    parser.add_argument('--adversarial', action='store_true',
                        help='Load an adversarial (GRL+Attention) checkpoint')

    # ScoreNet post-processing (optional, adds comparison)
    parser.add_argument('--scorenet_checkpoint', default=None, type=str,
                        help='Path to trained ScoreNet .pth; adds comparison table')
    parser.add_argument('--sn_min_dur', default=10, type=int,
                        help='Min event duration for ScoreNet hard constraints')

    args = parser.parse_args()
    main(args)