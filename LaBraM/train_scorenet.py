"""
Train the ScoreNet learned postprocessor on extracted probability sequences.

Usage:
    python train_scorenet.py \
        --data_dir ./scorenet_data \
        --output_dir ./scorenet_checkpoints \
        --epochs 200 --lr 1e-2
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from scorenet import (
    ScoreNet, ProbSequenceDataset, collate_fn,
    log_dice_loss, build_toeplitz,
)


def run_inference(model, loader, device):
    """Run ScoreNet on a dataloader, return flat refined probs, targets,
    and the mean loss."""
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for Z, labels, n_samples in loader:
            Z = Z.to(device)
            labels = labels.to(device)
            yhat = model(Z, n_samples)

            total_loss += log_dice_loss(yhat, labels).item()
            n_batches += 1

            all_probs.append(yhat.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets),
        total_loss / max(n_batches, 1),
    )


def compute_pointwise_metrics(probs, targets, threshold=0.5):
    """Point-wise precision, recall, F1, ROC-AUC, sensitivity, specificity."""
    preds = (probs >= threshold).astype(int)
    y_true = targets.astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc,
        'sensitivity': recall,
        'specificity': specificity,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def main():
    parser = argparse.ArgumentParser(description='Train ScoreNet postprocessor')
    parser.add_argument('--data_dir', default='./scorenet_data', type=str,
                        help='Directory with train.npz, val.npz from extract_probs.py')
    parser.add_argument('--output_dir', default='./scorenet_checkpoints', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--w', default=6, type=int,
                        help='Half-width of conv filter (filter_len = 2w+1)')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='Candidate threshold (fixed)')
    parser.add_argument('--max_len', default=None, type=int,
                        help='Max sub-sequence length for chunking (None=full patient)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Threshold for point-wise eval')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading datasets ...")
    train_dset = ProbSequenceDataset(
        os.path.join(args.data_dir, 'train.npz'),
        w=args.w, max_len=args.max_len)
    val_dset = ProbSequenceDataset(
        os.path.join(args.data_dir, 'val.npz'),
        w=args.w, max_len=args.max_len)

    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False)

    print(f"  Train: {len(train_dset)} sequences")
    print(f"  Val:   {len(val_dset)} sequences")

    model = ScoreNet(w=args.w, gamma=args.gamma).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ScoreNet parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for Z, labels, n_samples in tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            Z = Z.to(device)
            labels = labels.to(device)

            yhat = model(Z, n_samples)
            loss = log_dice_loss(yhat, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        val_probs, val_targets, avg_val_loss = run_inference(
            model, val_loader, device)
        pw = compute_pointwise_metrics(
            val_probs, val_targets, threshold=args.threshold)

        record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_f1': pw['f1'],
            'val_precision': pw['precision'],
            'val_recall': pw['recall'],
            'val_roc_auc': pw['roc_auc'],
            'val_sensitivity': pw['sensitivity'],
            'val_specificity': pw['specificity'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(record)

        improved = pw['f1'] > best_val_f1
        if improved:
            best_val_f1 = pw['f1']
            ckpt_path = os.path.join(args.output_dir, 'scorenet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'args': vars(args),
            }, ckpt_path)

        star = ' *' if improved else ''
        print(f"Epoch {epoch:3d} | "
              f"train {avg_train_loss:.4f} | "
              f"val {avg_val_loss:.4f} | "
              f"F1 {pw['f1']:.4f} | "
              f"P {pw['precision']:.4f} | "
              f"R {pw['recall']:.4f} | "
              f"AUC {pw['roc_auc']:.4f} | "
              f"Sens {pw['sensitivity']:.4f} | "
              f"Spec {pw['specificity']:.4f}"
              f"{star}")

    log_path = os.path.join(args.output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'scorenet_best.pth')}")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    main()
