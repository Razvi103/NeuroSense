"""
Evaluate saved ScoreNet checkpoints across multiple thresholds.

First run: extracts test probs via LaBraM and caches them as test_probs.npz
           in each fold directory (only 1 patient per fold — fast).
Subsequent runs: loads cached probs, so evaluation is instant.

Usage:
    python eval_scorenet_threshold.py \
        --lopocv_dir /path/to/lopocv_results \
        --data_dir /path/to/CHBMIT_per_patient \
        --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
        --sn_min_dur 10
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


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]


def get_args():
    p = argparse.ArgumentParser('Evaluate ScoreNet at multiple thresholds')
    p.add_argument('--lopocv_dir', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--folds', default='', type=str)

    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--drop_path', default=0.2, type=float)
    p.add_argument('--intermediate_layers', default='', type=str)
    p.add_argument('--batch_size', default=2048, type=int)

    p.add_argument('--thresholds', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                   type=str, help='Comma-separated thresholds to evaluate')
    p.add_argument('--sn_min_dur', default=10, type=int)

    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', default=4, type=int)
    return p.parse_args()


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


def get_events(arr):
    if len(arr) == 0:
        return []
    padded = np.concatenate(([0], arr, [0]))
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def compute_event_metrics(y_true, y_pred):
    te, pe = get_events(y_true), get_events(y_pred)
    tp = sum(1 for ts, t_e in te
             if any(max(ts, ps) < min(t_e, p_e) for ps, p_e in pe))
    fp = sum(1 for ps, p_e in pe
             if not any(max(ts, ps) < min(t_e, p_e) for ts, t_e in te))
    fn = len(te) - tp
    rec = tp / len(te) if te else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / (len(y_true) / 3600.0) if len(y_true) > 0 else 0.0
    return dict(n_seizures=len(te), evt_recall=rec, evt_precision=prec,
                evt_f1=f1, far_per_hr=far)


def evaluate_at_threshold(refined, y_true, threshold, min_dur):
    preds = hard_constraints((refined >= threshold).astype(int), min_dur)
    r = {}
    if len(np.unique(y_true)) > 1:
        r['roc_auc'] = float(roc_auc_score(y_true, refined))
        r['auprc'] = float(average_precision_score(y_true, refined))
    else:
        r['roc_auc'] = r['auprc'] = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    r['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r['f1'] = float(f1_score(y_true, preds, zero_division=0))
    evt = compute_event_metrics(y_true, preds)
    r.update(evt)
    return r


def main():
    args = get_args()
    cudnn.benchmark = True
    device = torch.device(args.device)
    thresholds = [float(t) for t in args.thresholds.split(',')]

    patient_h5s = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in sorted(glob.glob(os.path.join(args.data_dir, 'chb*.h5')))}

    folds = []
    for entry in sorted(os.listdir(args.lopocv_dir)):
        m = re.match(r'fold_(\d+)_(chb\d+)', entry)
        if not m:
            continue
        fd = os.path.join(args.lopocv_dir, entry)
        sn = os.path.join(fd, 'scorenet_best.pth')
        ckpt = os.path.join(fd, 'checkpoint-best.pth')
        if os.path.isfile(sn) and os.path.isfile(ckpt):
            folds.append((int(m.group(1)), m.group(2), fd, ckpt, sn))

    if args.folds:
        sel = set(int(x) for x in args.folds.split(','))
        folds = [f for f in folds if f[0] in sel]

    print(f"Evaluating {len(folds)} folds x {len(thresholds)} thresholds "
          f"({thresholds}), min_dur={args.sn_min_dur}")

    # {threshold: [per-fold result dicts]}
    results_by_thr = {t: [] for t in thresholds}
    start = time.time()

    for fold_idx, test_pid, fold_dir, ckpt_path, sn_path in folds:
        if test_pid not in patient_h5s:
            print(f"  Warning: {test_pid} not in data_dir, skipping")
            continue

        cache_path = os.path.join(fold_dir, 'test_probs.npz')

        # --- Get raw LaBraM probs (cached or extracted) ---
        if os.path.isfile(cache_path):
            cached = np.load(cache_path)
            test_probs, test_labels = cached['probs'], cached['labels']
            print(f"  Fold {fold_idx:2d} {test_pid}: loaded cached probs "
                  f"({len(test_probs)} windows)")
        else:
            print(f"  Fold {fold_idx:2d} {test_pid}: extracting test probs...")
            labram = load_fold_model(ckpt_path, args, device)
            test_probs, test_labels = extract_test_probs(
                labram, patient_h5s[test_pid], device,
                args.batch_size, args.num_workers)
            del labram
            torch.cuda.empty_cache()
            np.savez_compressed(cache_path, probs=test_probs, labels=test_labels)
            print(f"           cached to {cache_path} ({len(test_probs)} windows)")

        # --- Run ScoreNet forward once ---
        sn_ckpt = torch.load(sn_path, map_location='cpu', weights_only=False)
        sn_model = ScoreNet(
            w=sn_ckpt.get('w', 6), gamma=sn_ckpt.get('gamma', 0.5),
        ).to(device)
        sn_model.load_state_dict(sn_ckpt['model_state_dict'])
        sn_model.eval()

        Z = torch.from_numpy(
            build_toeplitz(test_probs.astype(np.float32), sn_model.w)
        ).to(device)
        with torch.no_grad():
            refined = sn_model(Z, [len(test_probs)]).cpu().numpy()
        del sn_model, Z
        torch.cuda.empty_cache()

        y_true = test_labels.astype(int)

        # --- Evaluate at every threshold (instant) ---
        for thr in thresholds:
            r = evaluate_at_threshold(refined, y_true, thr, args.sn_min_dur)
            r['fold'] = fold_idx
            r['test_patient'] = test_pid
            r['threshold'] = thr
            results_by_thr[thr].append(r)

    elapsed = time.time() - start
    print(f"\nDone in {datetime.timedelta(seconds=int(elapsed))}")

    # --- Summary table ---
    metric_keys = ['sensitivity', 'specificity', 'f1', 'roc_auc', 'auprc',
                   'precision', 'evt_f1', 'evt_recall', 'evt_precision', 'far_per_hr']

    print(f"\n{'='*100}")
    print("MEAN METRICS ACROSS FOLDS BY THRESHOLD")
    print(f"{'='*100}")
    header = (f"{'Thr':>5}  {'Sens':>7}  {'Spec':>7}  {'F1':>7}  {'AUC':>7}  "
              f"{'Prec':>7}  {'EvtF1':>7}  {'EvtRec':>7}  {'EvtP':>7}  {'FAR/hr':>7}")
    print(header)
    print('-' * len(header))

    summary = {}
    for thr in thresholds:
        fold_results = results_by_thr[thr]
        means = {}
        for k in metric_keys:
            vals = [r[k] for r in fold_results
                    if not (isinstance(r[k], float) and math.isnan(r[k]))]
            means[k] = float(np.mean(vals)) if vals else float('nan')
        summary[thr] = means
        print(f"{thr:5.1f}  {means['sensitivity']:7.4f}  {means['specificity']:7.4f}  "
              f"{means['f1']:7.4f}  {means['roc_auc']:7.4f}  "
              f"{means['precision']:7.4f}  {means['evt_f1']:7.4f}  "
              f"{means['evt_recall']:7.4f}  {means['evt_precision']:7.4f}  "
              f"{means['far_per_hr']:7.2f}")

    # Identify best threshold by F1
    best_thr = max(thresholds, key=lambda t: summary[t].get('f1', 0))
    print(f"\nBest mean F1: {summary[best_thr]['f1']:.4f} at threshold={best_thr}")

    # Per-fold detail for best threshold
    print(f"\n{'='*100}")
    print(f"PER-FOLD DETAIL AT BEST THRESHOLD = {best_thr}")
    print(f"{'='*100}")
    detail_header = (f"{'Fold':>4}  {'Patient':>7}  {'Sens':>7}  {'Spec':>7}  "
                     f"{'F1':>7}  {'AUC':>7}  {'EvtF1':>7}  {'FAR/hr':>7}  {'Szrs':>5}")
    print(detail_header)
    print('-' * len(detail_header))
    for r in results_by_thr[best_thr]:
        print(f"{r['fold']:4d}  {r['test_patient']:>7}  "
              f"{r['sensitivity']:7.4f}  {r['specificity']:7.4f}  "
              f"{r['f1']:7.4f}  {r['roc_auc']:7.4f}  "
              f"{r['evt_f1']:7.4f}  {r['far_per_hr']:7.2f}  "
              f"{r['n_seizures']:5d}")

    # Also print std for best threshold
    print('-' * len(detail_header))
    for k in metric_keys:
        vals = [r[k] for r in results_by_thr[best_thr]
                if not (isinstance(r[k], float) and math.isnan(r[k]))]
        if vals:
            print(f"  {k:>16}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    # Save everything
    out_path = os.path.join(args.lopocv_dir, 'scorenet_threshold_sweep.json')
    out = {str(t): results_by_thr[t] for t in thresholds}
    out['summary'] = {str(t): summary[t] for t in thresholds}
    out['best_threshold'] = best_thr
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()
