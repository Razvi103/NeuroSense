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
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from timm.models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

import modeling_finetune
from modeling_finetune import AdversarialNeuralTransformer
from dataset_maker.dataset_chbmit import MultiPatientAdversarialDataset
import utils

from scorenet import (
    ScoreNet, build_toeplitz, log_dice_loss, hard_constraints,
)


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument('--lopocv_dir', required=True, type=str)
    p.add_argument('--data_dir', required=True, type=str)
    p.add_argument('--folds', default='', type=str)

    p.add_argument('--model', default='labram_base_patch200_200', type=str)
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--drop_path', default=0.2, type=float)
    p.add_argument('--intermediate_layers', default='', type=str)
    p.add_argument('--batch_size', default=2048, type=int)

    p.add_argument('--sn_epochs', default=200, type=int)
    p.add_argument('--sn_lr', default=1e-2, type=float)
    p.add_argument('--sn_w', default=6, type=int)
    p.add_argument('--sn_gamma', default=0.5, type=float)
    p.add_argument('--sn_max_len', default=5000, type=int)
    p.add_argument('--sn_batch_size', default=32, type=int)
    p.add_argument('--sn_threshold', default=0.5, type=float)
    p.add_argument('--sn_min_dur', default=10, type=int)

    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--num_workers', default=4, type=int)

    return p.parse_args()


def load_fold_model(checkpoint_path, args, device):
    backbone = create_model(
        args.model, pretrained=False, num_classes=1,
        drop_rate=0.0, drop_path_rate=args.drop_path, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True,
        init_values=0.1,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}

    disc_keys = [k for k in clean_state
                 if k.startswith('patient_discriminator') and k.endswith('.weight')]

    if disc_keys:
        num_patients = clean_state[disc_keys[-1]].shape[0]
        il_str = args.intermediate_layers
        intermediate = tuple(int(x) for x in il_str.split(',') if x.strip()) if il_str else ()

        model = AdversarialNeuralTransformer(
            backbone, num_patients=num_patients,
            adv_hidden_dim=args.adv_hidden_dim,
            intermediate_layers=intermediate,
        )

        model.seizure_head = torch.nn.Sequential(
            torch.nn.Linear(backbone.embed_dim, backbone.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(backbone.embed_dim, backbone.num_classes),
        )

        model.load_state_dict(clean_state, strict=False)
    else:
        backbone.load_state_dict(clean_state, strict=False)
        model = backbone

    model.to(device).eval()
    return model

@torch.no_grad()
def extract_probs_from_dataset(model, dataset, device, batch_size, num_workers):
    input_chans = utils.get_input_chans(CHBMIT_CH_NAMES)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

    all_probs, all_labels, all_pids = [], [], []
    for batch in loader:
        samples = batch[0].float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        labels = batch[1]
        pids = batch[2]

        with torch.amp.autocast('cuda'):
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_pids.append(pids.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_pids)


def build_scorenet_items(probs, labels, pids, w, max_len):
    items = []
    for pid in np.unique(pids):
        mask = pids == pid
        p = probs[mask].astype(np.float32)
        lab = labels[mask].astype(np.float32)
        if max_len and len(p) > max_len:
            for start in range(0, len(p), max_len):
                end = min(start + max_len, len(p))
                Z = build_toeplitz(p[start:end], w)
                items.append((Z, lab[start:end], end - start))
        else:
            Z = build_toeplitz(p, w)
            items.append((Z, lab, len(p)))
    return items


def scorenet_collate(batch):
    Zs, labels, ns = zip(*batch)
    Z_cat = torch.cat([torch.from_numpy(z) for z in Zs], dim=1)
    labels_cat = torch.cat([torch.from_numpy(l) for l in labels], dim=0)
    return Z_cat, labels_cat, list(ns)


class InMemoryScoreNetDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_scorenet_fold(train_items, args, device):
    train_ds = InMemoryScoreNetDataset(train_items)
    train_loader = DataLoader(
        train_ds, batch_size=args.sn_batch_size, shuffle=True,
        collate_fn=scorenet_collate, num_workers=0, pin_memory=True,
    )

    model = ScoreNet(w=args.sn_w, gamma=args.sn_gamma).to(device)
    loss_fn = log_dice_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.sn_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.sn_epochs, eta_min=1e-5)

    best_loss = float('inf')
    best_state = None

    for epoch in range(1, args.sn_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for Z, labels, n_samples in train_loader:
            Z = Z.to(device)
            labels = labels.to(device)
            yhat = model(Z, n_samples)
            loss = loss_fn(yhat, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_loss

def get_events(binary_arr):
    if len(binary_arr) == 0:
        return []
    padded = np.concatenate(([0], binary_arr, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))


def compute_event_metrics(y_true, y_pred, stride_sec=1.0):
    true_events = get_events(y_true)
    pred_events = get_events(y_pred)

    tp_events = 0
    for t_s, t_e in true_events:
        for p_s, p_e in pred_events:
            if max(t_s, p_s) < min(t_e, p_e):
                tp_events += 1
                break

    fp_events = 0
    for p_s, p_e in pred_events:
        is_tp = False
        for t_s, t_e in true_events:
            if max(t_s, p_s) < min(t_e, p_e):
                is_tp = True
                break
        if not is_tp:
            fp_events += 1

    fn_events = len(true_events) - tp_events
    recall = tp_events / len(true_events) if true_events else 0.0
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total_hours = len(y_true) * stride_sec / 3600.0
    far = fp_events / total_hours if total_hours > 0 else 0.0

    return {
        'n_seizures': len(true_events),
        'tp': tp_events, 'fn': fn_events, 'fp': fp_events,
        'evt_recall': recall, 'evt_precision': precision, 'evt_f1': f1,
        'far_per_hr': far,
    }


@torch.no_grad()
def evaluate_scorenet(sn_model, test_probs, test_labels, device, threshold, min_dur):
    w = sn_model.w
    Z = build_toeplitz(test_probs.astype(np.float32), w)
    Z_t = torch.from_numpy(Z).to(device)
    refined = sn_model(Z_t, [len(test_probs)]).cpu().numpy()

    preds = (refined >= threshold).astype(int)
    preds = hard_constraints(preds, min_dur_sec=min_dur)
    y_true = test_labels.astype(int)

    results = {}
    if len(np.unique(y_true)) > 1:
        results['sn_roc_auc'] = float(roc_auc_score(y_true, refined))
        results['sn_auprc'] = float(average_precision_score(y_true, refined))
    else:
        results['sn_roc_auc'] = float('nan')
        results['sn_auprc'] = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    results['sn_sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    results['sn_specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    results['sn_precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    results['sn_f1'] = float(f1_score(y_true, preds, zero_division=0))

    evt = compute_event_metrics(y_true, preds)
    results['sn_evt_f1'] = evt['evt_f1']
    results['sn_evt_recall'] = evt['evt_recall']
    results['sn_evt_precision'] = evt['evt_precision']
    results['sn_far_per_hr'] = evt['far_per_hr']
    results['sn_n_seizures'] = evt['n_seizures']

    return results


def discover_folds(lopocv_dir):
    folds = []
    for entry in sorted(os.listdir(lopocv_dir)):
        m = re.match(r'fold_(\d+)_(chb\d+)', entry)
        if not m:
            continue
        fold_dir = os.path.join(lopocv_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        ckpt = os.path.join(fold_dir, 'checkpoint-best.pth')
        if not os.path.isfile(ckpt):
            continue
        fold_idx = int(m.group(1))
        test_pid = m.group(2)
        folds.append((fold_idx, test_pid, fold_dir, ckpt))
    return folds


def discover_patients(data_dir):
    h5_files = sorted(glob.glob(os.path.join(data_dir, 'chb*.h5')))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in h5_files}


def process_fold(fold_idx, test_pid, fold_dir, ckpt_path, patient_h5s, args, device):
    result_path = os.path.join(fold_dir, 'scorenet_results.json')
    if os.path.exists(result_path):
        print(f"  [Fold {fold_idx}] {test_pid}: scorenet_results.json exists, skipping.")
        with open(result_path) as f:
            return json.load(f)
    model = load_fold_model(ckpt_path, args, device)

    test_h5 = patient_h5s[test_pid]
    train_h5s = [p for pid, p in sorted(patient_h5s.items()) if pid != test_pid]

    test_ds = MultiPatientAdversarialDataset([test_h5])
    test_probs, test_labels, test_pids = extract_probs_from_dataset(
        model, test_ds, device, args.batch_size, args.num_workers)
    test_ds.close()

    train_ds = MultiPatientAdversarialDataset(train_h5s)
    train_probs, train_labels, train_pids = extract_probs_from_dataset(
        model, train_ds, device, args.batch_size, args.num_workers)
    train_ds.close()

    del model
    torch.cuda.empty_cache()

    max_len = args.sn_max_len if args.sn_max_len > 0 else None
    train_items = build_scorenet_items(train_probs, train_labels, train_pids,
                                       args.sn_w, max_len)
    sn_model, sn_best_loss = train_scorenet_fold(train_items, args, device)
    sn_ckpt_path = os.path.join(fold_dir, 'scorenet_best.pth')
    torch.save({
        'model_state_dict': sn_model.state_dict(),
        'args': {'w': args.sn_w, 'gamma': args.sn_gamma},
        'w': args.sn_w,
        'gamma': args.sn_gamma,
        'train_loss': sn_best_loss,
    }, sn_ckpt_path)

    results = evaluate_scorenet(
        sn_model, test_probs, test_labels, device,
        threshold=args.sn_threshold, min_dur=args.sn_min_dur)
    results['fold'] = fold_idx
    results['test_patient'] = test_pid
    results['sn_train_loss'] = float(sn_best_loss)
    results['n_train_windows'] = int(len(train_labels))
    results['n_test_windows'] = int(len(test_labels))

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  [Fold {fold_idx}] {test_pid}: "
          f"SN_Sens={results['sn_sensitivity']:.4f}  "
          f"SN_Spec={results['sn_specificity']:.4f}  "
          f"SN_F1={results['sn_f1']:.4f}  "
          f"SN_AUC={results['sn_roc_auc']:.4f}  "
          f"SN_EvtF1={results['sn_evt_f1']:.4f}  "
          f"SN_FAR={results['sn_far_per_hr']:.2f}/hr")

    return results


def aggregate_results(all_results, output_dir):
    metrics_keys = [
        'sn_sensitivity', 'sn_specificity', 'sn_f1', 'sn_roc_auc', 'sn_auprc',
        'sn_precision', 'sn_evt_f1', 'sn_evt_recall', 'sn_evt_precision',
        'sn_far_per_hr',
    ]

    header = (f"{'Fold':>4}  {'Patient':>7}  {'Sens':>7}  {'Spec':>7}  "
              f"{'F1':>7}  {'AUC':>7}  {'EvtF1':>7}  {'FAR/hr':>7}  {'Szrs':>5}")
    print(header)
    print('-' * len(header))

    for r in all_results:
        print(f"{r['fold']:4d}  {r['test_patient']:>7}  "
              f"{r['sn_sensitivity']:7.4f}  {r['sn_specificity']:7.4f}  "
              f"{r['sn_f1']:7.4f}  {r['sn_roc_auc']:7.4f}  "
              f"{r['sn_evt_f1']:7.4f}  {r['sn_far_per_hr']:7.2f}  "
              f"{r['sn_n_seizures']:5d}")

    print('-' * len(header))

    summary = {}
    for key in metrics_keys:
        values = [r[key] for r in all_results
                  if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        if values:
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            print(f"  {key:>20}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}  "
                  f"[{summary[key]['min']:.4f}, {summary[key]['max']:.4f}]")

    summary_path = os.path.join(output_dir, 'scorenet_lopocv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'per_fold': all_results, 'summary': summary}, f, indent=2)
    print(f"\nFull results saved to {summary_path}")
    return summary


def main():
    args = get_args()
    cudnn.benchmark = True
    device = torch.device(args.device)

    folds = discover_folds(args.lopocv_dir)
    print(f"Discovered {len(folds)} fold checkpoints in {args.lopocv_dir}")

    patient_h5s = discover_patients(args.data_dir)
    print(f"Discovered {len(patient_h5s)} patient H5 files in {args.data_dir}")

    if args.folds:
        selected = set(int(x) for x in args.folds.split(','))
        folds = [f for f in folds if f[0] in selected]
        print(f"Running {len(folds)} selected folds: {sorted(selected)}")

    all_results = []
    for fold_idx, test_pid, fold_dir, ckpt_path in folds:
        results = process_fold(fold_idx, test_pid, fold_dir, ckpt_path,
                               patient_h5s, args, device)
        all_results.append(results)

    if all_results:
        aggregate_results(all_results, args.lopocv_dir)


if __name__ == '__main__':
    main()
