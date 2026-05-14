"""Train ScoreNet for every run discovered by the benchmark pipeline.

For each run directory that contains a log.txt:
  1. Parse log.txt and select best checkpoint by val_roc_auc
  2. Load the frozen LaBraM model
  3. Extract probabilities from train.h5 and val.h5
  4. Train ScoreNet (log-dice loss, cosine LR, val-based early stopping)
  5. Save scorenet_best.pth inside the run directory

The saved checkpoint is placed where benchmark_runs.py's find_scorenet()
looks first: {run_dir}/scorenet_best.pth

Usage:
    python train_scorenet_all.py \
        --checkpoints_root /path/to/checkpoints \
        --data_path /path/to/TUSZ_patient_id_19_channels \
        --dataset TUSZ \
        --device cuda
"""

import argparse
import os
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from einops import rearrange

import utils
from evaluate_checkpoint import (
    CHBMITDataset,
    get_ch_names_for_dataset,
    load_model,
    compute_event_metrics,
)
from benchmark_runs import (
    discover_runs,
    parse_log,
    select_best_epoch,
    find_checkpoint,
    detect_adversarial,
    extract_hyperparams,
)
from scorenet import ScoreNet, build_toeplitz, log_dice_loss, hard_constraints


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataset, input_chans, device, batch_size=2048):
    """Run model inference, return (y_prob, y_true) numpy arrays."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=16, pin_memory=True)
    all_probs, all_targets = [], []

    for samples, targets in tqdm(loader, desc="  inference", leave=False):
        samples = samples.to(device) / 100.0
        samples = rearrange(samples, "B N (A T) -> B N A T", T=200)
        with torch.amp.autocast("cuda"):
            output = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(output).squeeze()
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

    return np.array(all_probs), np.array(all_targets)


# ── Toeplitz dataset ─────────────────────────────────────────────────────────

def build_toeplitz_items(probs, labels, w, max_len):
    """Chunk a flat probability sequence into Toeplitz items for ScoreNet."""
    items = []
    n = len(probs)
    if max_len and n > max_len:
        for start in range(0, n, max_len):
            end = min(start + max_len, n)
            Z = build_toeplitz(probs[start:end].astype(np.float32), w)
            items.append((Z, labels[start:end].astype(np.float32), end - start))
    else:
        Z = build_toeplitz(probs.astype(np.float32), w)
        items.append((Z, labels.astype(np.float32), n))
    return items


class ToeplitzDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def scorenet_collate(batch):
    Zs, labels, ns = zip(*batch)
    Z_cat = torch.cat([torch.from_numpy(z) for z in Zs], dim=1)
    labels_cat = torch.cat([torch.from_numpy(l) for l in labels], dim=0)
    return Z_cat, labels_cat, list(ns)


# ── ScoreNet training ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_scorenet_on_val(model, val_probs, val_labels, device, w, threshold=0.5):
    """Run ScoreNet on the flat val probability array and return event F1."""
    Z = build_toeplitz(val_probs.astype(np.float32), w)
    Z_t = torch.from_numpy(Z).to(device)
    refined = model(Z_t, [len(val_probs)]).cpu().numpy()
    preds = (refined >= threshold).astype(int)
    preds = hard_constraints(preds, min_dur_sec=10)
    evt = compute_event_metrics(val_labels.astype(int), preds)
    return evt["F1"]


def train_scorenet(train_items, val_probs, val_labels, args, device):
    """Train ScoreNet, selecting the best epoch by val event F1."""
    train_ds = ToeplitzDataset(train_items)
    train_loader = DataLoader(
        train_ds, batch_size=args.sn_batch_size, shuffle=True,
        collate_fn=scorenet_collate, num_workers=0, pin_memory=True,
    )

    model = ScoreNet(w=args.sn_w, gamma=args.sn_gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sn_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.sn_epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.sn_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for Z, labels, n_samples in train_loader:
            Z = Z.to(device)
            labels = labels.to(device)
            yhat = model(Z, n_samples)
            loss = log_dice_loss(yhat, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_f1 = evaluate_scorenet_on_val(
            model, val_probs, val_labels, device, args.sn_w)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{args.sn_epochs}  "
                  f"loss={avg_loss:.4f}  val_evt_F1={val_f1:.4f}  "
                  f"best={best_val_f1:.4f}@{best_epoch}")

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_val_f1, best_epoch


# ── Per-run pipeline ─────────────────────────────────────────────────────────

def process_run(run_name, run_dir, args, input_chans, train_dset, val_dset, device):
    """Extract probs, train ScoreNet, save checkpoint for one run."""
    print(f"\n{'='*70}")
    print(f"  Run: {run_name}")
    print(f"{'='*70}")

    save_path = os.path.join(run_dir, "scorenet_best.pth")
    if os.path.isfile(save_path) and not args.force:
        print(f"  [SKIP] scorenet_best.pth already exists (use --force to overwrite)")
        return

    log_entries = parse_log(os.path.join(run_dir, "log.txt"))
    if not log_entries:
        print("  [SKIP] Empty or unparseable log.txt")
        return

    best_val, _, _ = select_best_epoch(log_entries)
    if best_val is None:
        print("  [SKIP] No val_roc_auc found in log.txt")
        return

    best_epoch = best_val["epoch"]
    print(f"  Best val epoch: {best_epoch}  "
          f"(val_roc_auc={best_val.get('val_roc_auc', '?'):.4f})")

    ckpt_path = find_checkpoint(run_dir, best_epoch)
    if ckpt_path is None:
        print(f"  [SKIP] No checkpoint found for epoch {best_epoch}")
        return
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    hparams = extract_hyperparams(ckpt_path)
    is_adversarial = detect_adversarial(ckpt_path)

    model_args = SimpleNamespace(
        model="labram_base_patch200_200",
        checkpoint=ckpt_path,
        adversarial=is_adversarial,
        adv_hidden_dim=hparams.get("adv_hidden_dim", 512),
        device=str(device),
    )
    model = load_model(model_args, device)

    print("  Extracting train probabilities...")
    train_probs, train_labels = run_inference(
        model, train_dset, input_chans, device, args.batch_size)
    print(f"    train: {len(train_probs)} samples, "
          f"{train_labels.sum():.0f} positive ({train_labels.mean()*100:.1f}%)")

    print("  Extracting val probabilities...")
    val_probs, val_labels = run_inference(
        model, val_dset, input_chans, device, args.batch_size)
    print(f"    val: {len(val_probs)} samples, "
          f"{val_labels.sum():.0f} positive ({val_labels.mean()*100:.1f}%)")

    del model
    torch.cuda.empty_cache()

    print(f"  Building Toeplitz items (w={args.sn_w}, max_len={args.sn_max_len})...")
    train_items = build_toeplitz_items(
        train_probs, train_labels, args.sn_w, args.sn_max_len)
    print(f"    {len(train_items)} training chunks")

    print(f"  Training ScoreNet ({args.sn_epochs} epochs, lr={args.sn_lr})...")
    sn_model, best_val_f1, best_ep = train_scorenet(
        train_items, val_probs, val_labels, args, device)

    print(f"  Best ScoreNet: epoch {best_ep}, val_evt_F1={best_val_f1:.4f}")

    torch.save({
        "model_state_dict": sn_model.state_dict(),
        "args": {"w": args.sn_w, "gamma": args.sn_gamma},
        "epoch": best_ep,
        "val_f1": best_val_f1,
    }, save_path)
    print(f"  Saved: {save_path}")

    del sn_model
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ScoreNet for all benchmark runs"
    )
    parser.add_argument("--checkpoints_root", required=True, type=str,
                        help="Root folder containing run subdirectories")
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to folder with train.h5, val.h5")
    parser.add_argument("--dataset", default="TUSZ", type=str,
                        choices=["TUSZ", "CHBMIT"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=2048, type=int,
                        help="Batch size for probability extraction")

    parser.add_argument("--sn_epochs", default=200, type=int)
    parser.add_argument("--sn_lr", default=1e-2, type=float)
    parser.add_argument("--sn_w", default=6, type=int)
    parser.add_argument("--sn_gamma", default=0.5, type=float)
    parser.add_argument("--sn_max_len", default=5000, type=int,
                        help="Max chunk length for Toeplitz items")
    parser.add_argument("--sn_batch_size", default=32, type=int)

    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scorenet_best.pth")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Only process these run names (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    print(f"Loading train dataset from {args.data_path}/train.h5 ...")
    train_dset = CHBMITDataset(os.path.join(args.data_path, "train.h5"))
    print(f"  {len(train_dset)} samples")

    print(f"Loading val dataset from {args.data_path}/val.h5 ...")
    val_dset = CHBMITDataset(os.path.join(args.data_path, "val.h5"))
    print(f"  {len(val_dset)} samples")

    all_runs = discover_runs(args.checkpoints_root)
    if args.runs:
        all_runs = [(name, path) for name, path in all_runs if name in args.runs]
    print(f"\nFound {len(all_runs)} run(s):")
    for name, _ in all_runs:
        print(f"  - {name}")

    for run_name, run_dir in all_runs:
        try:
            process_run(run_name, run_dir, args, input_chans,
                        train_dset, val_dset, device)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_name}' failed: {e}")
            traceback.print_exc()
            continue

    print("\nDone. Run benchmark_runs.py to evaluate with the trained ScoreNet models.")


if __name__ == "__main__":
    main()
