"""
Train the ScoreNet learned postprocessor on extracted probability sequences.

Usage:
    python train_scorenet.py \
        --data_dir ./scorenet_data \
        --output_dir ./scorenet_checkpoints \
        --epochs 50 --lr 1e-3
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

from scorenet import ScoreNet, ProbSequenceDataset, collate_fn, combined_loss


def run_inference(model, loader, device):
    """Run ScoreNet on a dataloader, return flat arrays of refined probs,
    targets, and the mean loss."""
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for seqs, targets, lengths in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            refined = model(seqs, lengths)

            mask = targets >= 0
            total_loss += combined_loss(refined[mask], targets[mask]).item()
            n_batches += 1

            for i, length in enumerate(lengths):
                all_probs.append(refined[i, :length, 0].cpu().numpy())
                all_targets.append(targets[i, :length, 0].cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets),
        total_loss / max(n_batches, 1),
    )


def compute_pointwise_metrics(probs, targets, threshold=0.5):
    """Compute point-wise precision, recall, F1, ROC-AUC, sensitivity,
    specificity from flat probability and label arrays."""
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
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def main():
    parser = argparse.ArgumentParser(description='Train ScoreNet postprocessor')
    parser.add_argument('--data_dir', default='./scorenet_data', type=str,
                        help='Directory with train.npz, val.npz from extract_probs.py')
    parser.add_argument('--output_dir', default='./scorenet_checkpoints', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--max_len', default=4096, type=int,
                        help='Max sub-sequence length for chunking')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Weight for dice in combined loss (1-alpha for BCE)')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading datasets ...")
    train_dset = ProbSequenceDataset(
        os.path.join(args.data_dir, 'train.npz'), max_len=args.max_len)
    val_dset = ProbSequenceDataset(
        os.path.join(args.data_dir, 'val.npz'), max_len=args.max_len)

    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dset)} sequences")
    print(f"  Val:   {len(val_dset)} sequences")

    model = ScoreNet(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ScoreNet parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for seqs, targets, lengths in tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            seqs = seqs.to(device)
            targets = targets.to(device)

            refined = model(seqs, lengths)
            mask = targets >= 0
            loss = combined_loss(refined[mask], targets[mask], alpha=args.alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                'optimizer_state_dict': optimizer.state_dict(),
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
