"""
Full LOPOCV re-evaluation: produces a thesis-ready results table from
existing checkpoints, covering raw / hand-tuned / ScoreNet post-processing
with both strict and SzCORE event-based metrics.

No training is performed -- only inference (cached) + post-processing + metrics.

Usage:
    python eval_lopocv_full.py \
        --lopocv_dir /path/to/lopocv_results \
        --data_dir /path/to/CHBMIT_per_patient
"""

import argparse
import glob
import json
import math
import os
import re
import time
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, average_precision_score,
)
from timm.models import create_model
from torch.utils.data import DataLoader

import modeling_finetune
from modeling_finetune import AdversarialNeuralTransformer
from dataset_maker.dataset_chbmit import MultiPatientAdversarialDataset
import utils
from scorenet import ScoreNet, build_toeplitz, hard_constraints
from evaluate_checkpoint import (
    post_process_probs,
    compute_event_metrics,
    compute_szcore_event_metrics,
)


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser('Full LOPOCV re-evaluation table')
    p.add_argument('--lopocv_dir', required=True,
                   help='Root dir with fold_XX_chbYY/ subdirectories')
    p.add_argument('--data_dir', required=True,
                   help='Dir with per-patient H5 files (chb01.h5, ...)')
    p.add_argument('--folds', default='', type=str,
                   help='Comma-separated fold indices to evaluate (default: all)')

    # Model config (must match the training run)
    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--drop_path', default=0.2, type=float)
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--intermediate_layers', default='', type=str)

    # Hand-tuned post-processing (Option A: fixed literature-motivated defaults)
    p.add_argument('--smooth', default=5, type=int,
                   help='Rolling-average smoothing window (seconds)')
    p.add_argument('--t_high', default=0.5, type=float,
                   help='High threshold for dual hysteresis')
    p.add_argument('--t_low', default=0.3, type=float,
                   help='Low threshold for dual hysteresis')
    p.add_argument('--min_dur', default=10, type=int,
                   help='Min event duration (seconds) for hand-tuned')

    # ScoreNet post-processing
    p.add_argument('--sn_threshold', default=0.5, type=float,
                   help='Threshold for ScoreNet refined probabilities')
    p.add_argument('--sn_min_dur', default=10, type=int,
                   help='Min event duration for ScoreNet hard_constraints')

    # SzCORE parameters
    p.add_argument('--pre_ictal', default=30.0, type=float,
                   help='Pre-ictal tolerance in seconds')
    p.add_argument('--post_ictal', default=60.0, type=float,
                   help='Post-ictal tolerance in seconds')
    p.add_argument('--merge_gap', default=90.0, type=float,
                   help='Merge gap in seconds for SzCORE')
    p.add_argument('--max_event', default=300.0, type=float,
                   help='Max event duration in seconds for SzCORE')

    # Runtime
    p.add_argument('--batch_size', default=2048, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='', type=str,
                   help='Output JSON path (default: lopocv_full_eval.json in lopocv_dir)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading & probability extraction (from eval_scorenet_threshold.py)
# ---------------------------------------------------------------------------

def load_fold_model(ckpt_path, args, device):
    backbone = create_model(
        args.model, pretrained=False, num_classes=1,
        drop_rate=0.0, drop_path_rate=args.drop_path, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True,
        init_values=0.1,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    clean = {k.replace('module.', ''): v for k, v in state.items()}

    disc_keys = [k for k in clean
                 if k.startswith('patient_discriminator') and k.endswith('.weight')]
    num_patients = clean[disc_keys[-1]].shape[0]

    il = args.intermediate_layers
    intermediate = tuple(int(x) for x in il.split(',') if x.strip()) if il else ()

    model = AdversarialNeuralTransformer(
        backbone, num_patients=num_patients,
        adv_hidden_dim=args.adv_hidden_dim,
        intermediate_layers=intermediate,
    )

    if 'seizure_head.0.weight' in clean:
        model.seizure_head = torch.nn.Sequential(
            torch.nn.Linear(backbone.embed_dim, backbone.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(backbone.embed_dim, backbone.num_classes),
        )
        print("    Detected 2-layer MLP seizure head checkpoint")
    else:
        print("    Detected single-layer seizure head checkpoint")

    model.load_state_dict(clean, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def extract_test_probs(model, h5_path, device, batch_size, num_workers):
    input_chans = utils.get_input_chans(CHBMIT_CH_NAMES)
    ds = MultiPatientAdversarialDataset([h5_path])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_probs, all_labels = [], []
    for batch in loader:
        samples = batch[0].float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        with torch.amp.autocast('cuda'):
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch[1].numpy())
    ds.close()
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_pointwise(y_true, y_pred, y_prob):
    """Point-wise metrics from binary predictions and raw probabilities."""
    r = {}
    if len(np.unique(y_true)) > 1:
        r['auc'] = float(roc_auc_score(y_true, y_prob))
        r['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        r['auc'] = r['auprc'] = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    r['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    return r


def evaluate_strategy(y_true, y_pred, y_prob, args):
    """Compute all metrics for a single post-processing strategy."""
    pw = compute_pointwise(y_true, y_pred, y_prob)

    strict = compute_event_metrics(y_true, y_pred)
    strict_out = {
        'evt_f1': strict['F1'],
        'evt_recall': strict['Recall'],
        'evt_precision': strict['Precision'],
        'evt_far_hr': strict['FAR/hr'],
        'evt_tp': strict['TP'],
        'evt_fn': strict['FN'],
        'evt_fp': strict['FP'],
        'n_seizures': strict['Total Seizures'],
    }

    szcore = compute_szcore_event_metrics(
        y_true, y_pred,
        pre_ictal_sec=args.pre_ictal,
        post_ictal_sec=args.post_ictal,
        merge_gap_sec=args.merge_gap,
        max_event_sec=args.max_event,
    )
    szcore_out = {
        'szcore_evt_f1': szcore['F1'],
        'szcore_evt_recall': szcore['Sensitivity'],
        'szcore_evt_precision': szcore['Precision'],
        'szcore_far_hr': szcore['FAR/hr'],
        'szcore_far_day': szcore['FAR/day'],
        'szcore_tp': szcore['TP'],
        'szcore_fn': szcore['FN'],
        'szcore_fp': szcore['FP'],
    }

    return {**pw, **strict_out, **szcore_out}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    cudnn.benchmark = True
    device = torch.device(args.device)

    patient_h5s = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in sorted(glob.glob(os.path.join(args.data_dir, 'chb*.h5')))
    }

    # Discover folds
    folds = []
    for entry in sorted(os.listdir(args.lopocv_dir)):
        m = re.match(r'fold_(\d+)_(chb\d+)', entry)
        if not m:
            continue
        fd = os.path.join(args.lopocv_dir, entry)
        ckpt = os.path.join(fd, 'checkpoint-best.pth')
        if os.path.isfile(ckpt):
            folds.append((int(m.group(1)), m.group(2), fd, ckpt))

    if args.folds:
        sel = set(int(x) for x in args.folds.split(','))
        folds = [f for f in folds if f[0] in sel]

    print(f"Evaluating {len(folds)} folds")
    print(f"  Hand-tuned: smooth={args.smooth}, t_high={args.t_high}, "
          f"t_low={args.t_low}, min_dur={args.min_dur}")
    print(f"  ScoreNet:   threshold={args.sn_threshold}, min_dur={args.sn_min_dur}")
    print(f"  SzCORE:     pre={args.pre_ictal}s, post={args.post_ictal}s, "
          f"merge={args.merge_gap}s, max={args.max_event}s")

    strategies = ['raw', 'hand_tuned', 'scorenet']
    results_by_strategy = {s: [] for s in strategies}

    start = time.time()

    for fold_idx, test_pid, fold_dir, ckpt_path in folds:
        if test_pid not in patient_h5s:
            print(f"  Warning: {test_pid} not in data_dir, skipping")
            continue

        print(f"\n  Fold {fold_idx:2d} {test_pid}:")

        # --- Load / cache probabilities ---
        cache_path = os.path.join(fold_dir, 'test_probs.npz')
        if os.path.isfile(cache_path):
            cached = np.load(cache_path)
            probs, labels = cached['probs'], cached['labels']
            print(f"    Loaded cached probs ({len(probs)} windows)")
        else:
            print(f"    Extracting test probs...")
            labram = load_fold_model(ckpt_path, args, device)
            probs, labels = extract_test_probs(
                labram, patient_h5s[test_pid], device,
                args.batch_size, args.num_workers)
            del labram
            torch.cuda.empty_cache()
            np.savez_compressed(cache_path, probs=probs, labels=labels)
            print(f"    Cached to {cache_path} ({len(probs)} windows)")

        y_true = labels.astype(int)

        # --- Strategy 1: Raw (threshold at 0.5) ---
        y_pred_raw = (probs >= 0.5).astype(int)
        raw_metrics = evaluate_strategy(y_true, y_pred_raw, probs, args)
        raw_metrics['fold'] = fold_idx
        raw_metrics['patient'] = test_pid
        results_by_strategy['raw'].append(raw_metrics)
        print(f"    raw:        F1={raw_metrics['f1']:.4f}  "
              f"EvtF1={raw_metrics['evt_f1']:.4f}  "
              f"ScEvF1={raw_metrics['szcore_evt_f1']:.4f}")

        # --- Strategy 2: Hand-tuned ---
        y_pred_ht = post_process_probs(
            probs, t_high=args.t_high, t_low=args.t_low,
            smooth_window=args.smooth, min_duration=args.min_dur,
        )
        ht_metrics = evaluate_strategy(y_true, y_pred_ht, probs, args)
        ht_metrics['fold'] = fold_idx
        ht_metrics['patient'] = test_pid
        results_by_strategy['hand_tuned'].append(ht_metrics)
        print(f"    hand_tuned: F1={ht_metrics['f1']:.4f}  "
              f"EvtF1={ht_metrics['evt_f1']:.4f}  "
              f"ScEvF1={ht_metrics['szcore_evt_f1']:.4f}")

        # --- Strategy 3: ScoreNet (if checkpoint exists) ---
        sn_path = os.path.join(fold_dir, 'scorenet_best.pth')
        if os.path.isfile(sn_path):
            sn_ckpt = torch.load(sn_path, map_location='cpu', weights_only=False)
            sn_model = ScoreNet(
                w=sn_ckpt.get('w', 6), gamma=sn_ckpt.get('gamma', 0.5),
            ).to(device)
            sn_model.load_state_dict(sn_ckpt['model_state_dict'])
            sn_model.eval()

            Z = torch.from_numpy(
                build_toeplitz(probs.astype(np.float32), sn_model.w)
            ).to(device)
            with torch.no_grad():
                refined = sn_model(Z, [len(probs)]).cpu().numpy()

            y_pred_sn = (refined >= args.sn_threshold).astype(int)
            y_pred_sn = hard_constraints(y_pred_sn, min_dur_sec=args.sn_min_dur)

            sn_metrics = evaluate_strategy(y_true, y_pred_sn, probs, args)
            sn_metrics['fold'] = fold_idx
            sn_metrics['patient'] = test_pid
            results_by_strategy['scorenet'].append(sn_metrics)
            print(f"    scorenet:   F1={sn_metrics['f1']:.4f}  "
                  f"EvtF1={sn_metrics['evt_f1']:.4f}  "
                  f"ScEvF1={sn_metrics['szcore_evt_f1']:.4f}")

            del sn_model, Z, refined
            torch.cuda.empty_cache()
        else:
            print(f"    scorenet:   no checkpoint found, skipping")

    elapsed = time.time() - start
    print(f"\n{'='*90}")
    print(f"Evaluation completed in {datetime.timedelta(seconds=int(elapsed))}")
    print(f"{'='*90}")

    # --- Aggregation ---
    metric_keys = [
        'sensitivity', 'specificity', 'f1', 'auc', 'auprc',
        'evt_f1', 'evt_recall', 'evt_precision', 'evt_far_hr',
        'szcore_evt_f1', 'szcore_evt_recall', 'szcore_evt_precision',
        'szcore_far_hr', 'szcore_far_day',
    ]

    summary = {}
    for strategy in strategies:
        fold_results = results_by_strategy[strategy]
        if not fold_results:
            continue

        print(f"\n{'─'*90}")
        print(f"  Strategy: {strategy.upper()}  ({len(fold_results)} folds)")
        print(f"{'─'*90}")

        header = (f"  {'Fold':>4}  {'Patient':>7}  {'Sens':>7}  {'F1':>7}  "
                  f"{'AUC':>7}  {'EvtF1':>7}  {'ScEvF1':>7}  "
                  f"{'EvtRec':>7}  {'FAR/hr':>7}  {'Szrs':>5}")
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for r in fold_results:
            print(f"  {r['fold']:4d}  {r['patient']:>7}  "
                  f"{r['sensitivity']:7.4f}  {r['f1']:7.4f}  "
                  f"{r['auc']:7.4f}  {r['evt_f1']:7.4f}  "
                  f"{r['szcore_evt_f1']:7.4f}  "
                  f"{r['evt_recall']:7.4f}  {r['evt_far_hr']:7.2f}  "
                  f"{r['n_seizures']:5d}")

        print('  ' + '-' * (len(header) - 2))

        strat_summary = {}
        for key in metric_keys:
            values = [r[key] for r in fold_results
                      if key in r and not (isinstance(r[key], float)
                                           and math.isnan(r[key]))]
            if values:
                strat_summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
                print(f"    {key:>22}: "
                      f"{strat_summary[key]['mean']:.4f} +/- "
                      f"{strat_summary[key]['std']:.4f}  "
                      f"[{strat_summary[key]['min']:.4f}, "
                      f"{strat_summary[key]['max']:.4f}]")

        summary[strategy] = {
            'per_fold': fold_results,
            'summary': strat_summary,
        }

    # --- Save ---
    output_path = args.output or os.path.join(args.lopocv_dir, 'lopocv_full_eval.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == '__main__':
    main()
