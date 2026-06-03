"""
Leave-One-Patient-Out Cross-Validation (LOPOCV) for CHB-MIT.
Implements the standard 24-fold LOPOCV protocol used in the CHB-MIT seizure
detection literature:
  - Each fold holds out 1 patient for testing, trains on the remaining 23
  - Supports both adversarial (GRL + channel attention) and pure baseline mode
  - Fixed training schedule
"""

import argparse
import copy
import datetime
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from timm.models import create_model
from torch.utils.data import DataLoader

import modeling_finetune
from modeling_finetune import AdversarialNeuralTransformer
from dataset_maker.dataset_chbmit import MultiPatientAdversarialDataset
from engine_for_finetuning import train_one_epoch, train_one_epoch_adversarial, evaluate
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from evaluate_checkpoint import compute_szcore_event_metrics


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]


def get_args():
    p = argparse.ArgumentParser('CHB-MIT LOPOCV with adversarial training')

    # Data
    p.add_argument('--data_dir', required=True, type=str,
                   help='Directory containing per-patient H5 files (chb01.h5, ...)')
    p.add_argument('--output_dir', required=True, type=str,
                   help='Root directory for per-fold outputs and aggregated results')
    p.add_argument('--folds', default='', type=str,
                   help='Comma-separated fold indices to run (default: all)')

    # Model
    p.add_argument('--model', default='labram_base_patch200_200', type=str)
    p.add_argument('--finetune', default='', type=str,
                   help='Path to pretrained LaBraM checkpoint')
    p.add_argument('--input_size', default=200, type=int)

    # Training (defaults match proven CHB-MIT adversarial config)
    p.add_argument('--epochs', default=20, type=int)
    p.add_argument('--batch_size', default=1024, type=int)
    p.add_argument('--lr', default=3e-6, type=float)
    p.add_argument('--min_lr', default=1e-6, type=float)
    p.add_argument('--warmup_lr', default=1e-6, type=float)
    p.add_argument('--warmup_epochs', default=5, type=int)
    p.add_argument('--weight_decay', default=0.1, type=float)
    p.add_argument('--layer_decay', default=0.65, type=float)
    p.add_argument('--clip_grad', default=1.0, type=float)
    p.add_argument('--drop_path', default=0.2, type=float)
    p.add_argument('--opt', default='adamw', type=str)
    p.add_argument('--opt_eps', default=1e-8, type=float)
    p.add_argument('--opt_betas', default=None, type=float, nargs='+')
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--update_freq', default=1, type=int)
    p.add_argument('--pos_weight', default=10.0, type=float,
                   help='pos_weight for BCEWithLogitsLoss')

    # Model details
    p.add_argument('--drop', default=0.0, type=float)
    p.add_argument('--attn_drop_rate', default=0.0, type=float)
    p.add_argument('--qkv_bias', action='store_true')
    p.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    p.set_defaults(qkv_bias=False)
    p.add_argument('--rel_pos_bias', action='store_true')
    p.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    p.set_defaults(rel_pos_bias=False)
    p.add_argument('--abs_pos_emb', action='store_true')
    p.set_defaults(abs_pos_emb=True)
    p.add_argument('--layer_scale_init_value', default=0.1, type=float)
    p.add_argument('--use_mean_pooling', action='store_true')
    p.set_defaults(use_mean_pooling=True)
    p.add_argument('--init_scale', default=0.001, type=float)

    # Baseline vs adversarial
    p.add_argument('--no_adversarial', action='store_true', default=False,
                   help='Run pure baseline (plain NeuralTransformer, no channel '
                        'attention, no GRL). Overrides all adversarial settings.')

    # Adversarial 
    p.add_argument('--adv_lambda', default=0.01, type=float)
    p.add_argument('--adv_gamma', default=5.0, type=float)
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--intermediate_layers', default='', type=str,
                   help='Comma-separated backbone block indices for multi-layer adversarial heads')

    # Misc
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--pin_mem', action='store_true')
    p.set_defaults(pin_mem=True)
    p.add_argument('--save_ckpt_freq', default=1, type=int)

    # Checkpoint loading keys
    p.add_argument('--model_key', default='model|module', type=str)
    p.add_argument('--model_prefix', default='', type=str)
    p.add_argument('--model_filter_name', default='gzp', type=str)

    return p.parse_args()


def discover_patients(data_dir):
    """Find all per-patient H5 files, return sorted list of (patient_id, h5_path)."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, 'chb*.h5')))
    patients = []
    for path in h5_files:
        pid = os.path.splitext(os.path.basename(path))[0]
        patients.append((pid, path))
    return patients


def load_pretrained_state_dict(args):
    """Load and clean the pretrained checkpoint state dict once."""
    if not args.finetune:
        return None
    checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    if args.model_filter_name != '':
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('student.'):
                new_dict[key[8:]] = checkpoint_model[key]
        if new_dict:
            checkpoint_model = new_dict

    for k in list(checkpoint_model.keys()):
        if 'relative_position_index' in k:
            del checkpoint_model[k]

    return checkpoint_model


def build_model(args, num_patients, pretrained_state, device):
    """Create a model and load pretrained weights.
    """
    backbone = create_model(
        args.model,
        pretrained=False,
        num_classes=1,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    if pretrained_state is not None:
        state = backbone.state_dict()
        filtered = {k: v for k, v in pretrained_state.items()
                    if k in state and v.shape == state[k].shape}
        utils.load_state_dict(backbone, filtered, prefix=args.model_prefix)

    if args.no_adversarial:
        model = backbone
    else:
        il_str = args.intermediate_layers
        intermediate = tuple(int(x) for x in il_str.split(',') if x.strip()) if il_str else ()
        model = AdversarialNeuralTransformer(
            backbone,
            num_patients=num_patients,
            adv_hidden_dim=args.adv_hidden_dim,
            intermediate_layers=intermediate,
        )

    model.to(device)
    return model


class _DropPatientID(torch.utils.data.Dataset):
    """Wraps MultiPatientAdversarialDataset, dropping the patient_id column
    so the dataloader yields (data, label) for the non-adversarial train loop."""

    def __init__(self, dataset):
        self._ds = dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        data, label, _pid = self._ds[idx]
        return data, label

    def close(self):
        self._ds.close()


@torch.no_grad()
def evaluate_fold(model, data_loader, device, ch_names, threshold=0.5):
    """Run inference on the held-out patient, return comprehensive metrics."""
    input_chans = utils.get_input_chans(ch_names)
    model.eval()
    all_probs = []
    all_targets = []

    from einops import rearrange
    for batch in data_loader:
        samples = batch[0].float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        targets = batch[1]

        with torch.amp.autocast('cuda'):
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    y_pred = (y_prob >= threshold).astype(int)

    results = {}
    if len(np.unique(y_true)) > 1:
        results['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        results['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        results['roc_auc'] = float('nan')
        results['auprc'] = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    results['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    results['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    results['n_samples'] = int(len(y_true))
    results['n_seizure_samples'] = int(y_true.sum())

    szcore_evt = compute_szcore_event_metrics(y_true, y_pred)
    results['szcore_evt_f1'] = szcore_evt['F1']
    results['szcore_evt_recall'] = szcore_evt['Sensitivity']
    results['szcore_evt_precision'] = szcore_evt['Precision']
    results['szcore_far_per_hr'] = szcore_evt['FAR/hr']
    results['szcore_far_per_day'] = szcore_evt['FAR/day']
    results['n_seizures'] = szcore_evt['Total Seizures (ref)']

    return results


def run_fold(fold_idx, test_pid, test_path, train_paths, args, pretrained_state, device):
    """Train and evaluate a single LOPOCV fold."""
    fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx:02d}_{test_pid}')
    os.makedirs(fold_dir, exist_ok=True)

    result_path = os.path.join(fold_dir, 'results.json')
    if os.path.exists(result_path):
        print(f"\n[Fold {fold_idx}] {test_pid}: results already exist, skipping.")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"[Fold {fold_idx}] Test patient: {test_pid}")
    print(f"  Training on {len(train_paths)} patient files")
    print(f"{'='*60}")

    raw_train_dataset = MultiPatientAdversarialDataset(train_paths)
    raw_test_dataset = MultiPatientAdversarialDataset([test_path])
    num_patients = raw_train_dataset.num_patients

    if args.no_adversarial:
        train_dataset = _DropPatientID(raw_train_dataset)
        test_dataset = _DropPatientID(raw_test_dataset)
    else:
        train_dataset = raw_train_dataset
        test_dataset = raw_test_dataset

    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    if not args.no_adversarial:
        print(f"  Num patients (discriminator classes): {num_patients}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(1.5 * args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )

    model = build_model(args, num_patients, pretrained_state, device)
    model_without_ddp = model

    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            [args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        )
    else:
        assigner = None

    skip_wd_list = model.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_wd_list,
        get_num_layer=assigner.get_layer_id if assigner else None,
        get_layer_scale=assigner.get_scale if assigner else None,
    )
    loss_scaler = NativeScaler()

    total_batch_size = args.batch_size * args.update_freq
    num_steps_per_epoch = len(train_dataset) // total_batch_size

    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_steps_per_epoch,
    )

    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ch_names = CHBMIT_CH_NAMES
    best_train_loss = float('inf')

    for epoch in range(args.epochs):
        if args.no_adversarial:
            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema=None,
                log_writer=None, start_steps=epoch * num_steps_per_epoch,
                lr_schedule_values=lr_schedule, wd_schedule_values=wd_schedule,
                num_training_steps_per_epoch=num_steps_per_epoch,
                update_freq=args.update_freq, ch_names=ch_names, is_binary=True,
            )
        else:
            train_stats = train_one_epoch_adversarial(
                model, criterion, train_loader, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema=None,
                log_writer=None, start_steps=epoch * num_steps_per_epoch,
                lr_schedule_values=lr_schedule, wd_schedule_values=wd_schedule,
                num_training_steps_per_epoch=num_steps_per_epoch,
                update_freq=args.update_freq, ch_names=ch_names, is_binary=True,
                total_epochs=args.epochs, adv_lambda=args.adv_lambda,
                adv_gamma=args.adv_gamma,
            )

        current_loss = train_stats.get('loss', float('inf'))
        if current_loss < best_train_loss:
            best_train_loss = current_loss
            ckpt = {
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
                'train_loss': current_loss,
            }
            torch.save(ckpt, os.path.join(fold_dir, 'checkpoint-best.pth'))

    best_ckpt = torch.load(os.path.join(fold_dir, 'checkpoint-best.pth'),
                           map_location='cpu', weights_only=False)
    model_without_ddp.load_state_dict(best_ckpt['model'])
    model_without_ddp.to(device)
    print(f"  Loaded best checkpoint (epoch {best_ckpt['epoch']}, "
          f"loss {best_ckpt['train_loss']:.4f})")

    results = evaluate_fold(model_without_ddp, test_loader, device, ch_names)
    results['fold'] = fold_idx
    results['test_patient'] = test_pid
    results['best_epoch'] = int(best_ckpt['epoch'])
    results['best_train_loss'] = float(best_ckpt['train_loss'])

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  [Fold {fold_idx}] {test_pid}: "
          f"Sens={results['sensitivity']:.4f}  "
          f"Spec={results['specificity']:.4f}  "
          f"F1={results['f1']:.4f}  "
          f"AUC={results['roc_auc']:.4f}  "
          f"ScEvF1={results['szcore_evt_f1']:.4f}  "
          f"FAR={results['szcore_far_per_hr']:.2f}/hr")

    raw_train_dataset.close()
    raw_test_dataset.close()

    return results


def aggregate_results(all_results, output_dir):
    """Print and save a summary table with mean +/- std."""
    metrics_keys = [
        'sensitivity', 'specificity', 'f1', 'roc_auc', 'auprc',
        'precision',
        'szcore_evt_f1', 'szcore_evt_recall', 'szcore_evt_precision',
        'szcore_far_per_hr', 'szcore_far_per_day',
    ]

    print(f"\n{'='*80}")
    print("LOPOCV RESULTS SUMMARY")
    print(f"{'='*80}")

    header = (f"{'Fold':>4}  {'Patient':>7}  {'Sens':>7}  {'Spec':>7}  "
              f"{'F1':>7}  {'AUC':>7}  {'ScEvF1':>7}  "
              f"{'FAR/hr':>7}  {'Szrs':>5}")
    print(header)
    print('-' * len(header))

    for r in all_results:
        print(f"{r['fold']:4d}  {r['test_patient']:>7}  "
              f"{r['sensitivity']:7.4f}  {r['specificity']:7.4f}  "
              f"{r['f1']:7.4f}  {r['roc_auc']:7.4f}  "
              f"{r.get('szcore_evt_f1', float('nan')):7.4f}  "
              f"{r.get('szcore_far_per_hr', float('nan')):7.2f}  "
              f"{r.get('n_seizures', 0):5d}")

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
            print(f"  {key:>16}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}  "
                  f"[{summary[key]['min']:.4f}, {summary[key]['max']:.4f}]")

    summary_path = os.path.join(output_dir, 'lopocv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'per_fold': all_results, 'summary': summary}, f, indent=2)
    print(f"\nFull results saved to {summary_path}")

    return summary


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device)
    patients = discover_patients(args.data_dir)
    n_patients = len(patients)
    print(f"Discovered {n_patients} patients in {args.data_dir}")

    if args.folds:
        fold_indices = [int(x) for x in args.folds.split(',')]
    else:
        fold_indices = list(range(n_patients))

    pretrained_state = load_pretrained_state_dict(args)
    if pretrained_state is not None:
        print(f"Loaded pretrained checkpoint from {args.finetune}")

    all_results = []
    start_time = time.time()

    for fold_idx in fold_indices:
        if fold_idx >= n_patients:
            print(f"Warning: fold {fold_idx} >= {n_patients} patients, skipping.")
            continue

        test_pid, test_path = patients[fold_idx]
        train_paths = [p for i, (_, p) in enumerate(patients) if i != fold_idx]

        results = run_fold(fold_idx, test_pid, test_path, train_paths,
                           args, pretrained_state, device)
        all_results.append(results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {datetime.timedelta(seconds=int(elapsed))}")

    if all_results:
        aggregate_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
