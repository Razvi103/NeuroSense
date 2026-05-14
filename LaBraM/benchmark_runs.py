"""Centralized benchmarking script for TUSZ / CHB-MIT seizure detection runs.

Scans a checkpoints root folder, parses each run's log.txt, selects the best
checkpoint by val_roc_auc, tunes postprocessing on val, and evaluates on test,
producing a unified comparison table (CSV + terminal).

Usage:
    python benchmark_runs.py \
        --checkpoints_root /path/to/checkpoints \
        --data_path /path/to/TUSZ_patient_id_19_channels \
        --dataset TUSZ \
        --device cuda \
        --output results_summary.csv
"""

import argparse
import itertools
import json
import os
import glob
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import utils
from evaluate_checkpoint import (
    CHBMITDataset,
    post_process_probs,
    get_ch_names_for_dataset,
    load_model,
    load_scorenet,
    scorenet_postprocess,
    compute_event_metrics,
    compute_szcore_event_metrics,
)


# ── Log parsing ──────────────────────────────────────────────────────────────

def discover_runs(checkpoints_root):
    """Return sorted list of (run_name, run_dir) for dirs containing log.txt."""
    runs = []
    root = Path(checkpoints_root)
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "log.txt").exists():
            runs.append((entry.name, str(entry)))
    return runs


def parse_log(log_path):
    """Parse a JSONL log.txt and return list of per-epoch dicts."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def select_best_epoch(log_entries, metric="val_roc_auc"):
    """Pick the epoch with the highest *metric* value.

    Returns (best_entry, best_test_entry_or_None, val_test_match_bool).
    """
    valid = [e for e in log_entries if metric in e]
    if not valid:
        return None, None, False

    best = max(valid, key=lambda e: e[metric])

    test_metric = metric.replace("val_", "test_")
    test_valid = [e for e in log_entries if test_metric in e]
    best_test = max(test_valid, key=lambda e: e[test_metric]) if test_valid else None

    match = (best_test is not None and best["epoch"] == best_test["epoch"])
    return best, best_test, match


def find_checkpoint(run_dir, epoch):
    """Locate the .pth file for a given epoch, with fallbacks."""
    candidates = [
        os.path.join(run_dir, f"checkpoint-{epoch}.pth"),
        os.path.join(run_dir, "checkpoint.pth"),
        os.path.join(run_dir, "checkpoint-best.pth"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def extract_hyperparams(ckpt_path):
    """Load a checkpoint and pull hyperparameters from saved args."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", None)
    hparams = {}
    if saved_args is not None:
        ns = saved_args if isinstance(saved_args, dict) else vars(saved_args)
        for key in [
            "lr", "pos_weight", "adv_lambda", "adv_gamma", "adv_hidden_dim",
            "epochs", "warmup_epochs", "batch_size", "drop_path", "seed",
            "weight_decay", "layer_decay", "clip_grad", "adversarial",
            "intermediate_layers", "dataset",
        ]:
            hparams[key] = ns.get(key, "N/A")
    return hparams


def detect_adversarial(ckpt_path):
    """Check whether a checkpoint contains adversarial wrapper keys."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    return any(k.startswith("patient_discriminator") for k in state.keys())


def find_scorenet(run_dir, checkpoints_root):
    """Look for a ScoreNet checkpoint in the run dir or sibling dirs."""
    patterns = [
        os.path.join(run_dir, "scorenet_best.pth"),
        os.path.join(run_dir, "scorenet*.pth"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return sorted(matches)[0]

    run_name = os.path.basename(run_dir)
    sibling_patterns = [
        os.path.join(checkpoints_root, f"{run_name}_scorenet_data", "scorenet_best.pth"),
        os.path.join(checkpoints_root, f"*{run_name}*scorenet*", "scorenet_best.pth"),
    ]
    for pat in sibling_patterns:
        matches = glob.glob(pat)
        if matches:
            return sorted(matches)[0]
    return None


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataset, input_chans, device, batch_size=2048):
    """Run model inference on a dataset, return (y_prob, y_true) numpy arrays."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
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


# ── Postprocessing tuning ────────────────────────────────────────────────────

PP_GRID = {
    "t_high": [0.3, 0.4, 0.5, 0.6, 0.7],
    "t_low": [0.2, 0.3, 0.4],
    "smooth": [3, 5, 10, 15],
    "min_dur": [5, 7, 10],
}

SN_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def tune_postprocessing(y_prob, y_true):
    """Grid-search hand-tuned PP params on val, select by event F1."""
    keys, values = zip(*PP_GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_f1 = -1.0
    best_params = None
    best_metrics = None

    for p in combos:
        if p["t_low"] >= p["t_high"]:
            continue
        y_pred = post_process_probs(
            y_prob, p["t_high"], p["t_low"], p["smooth"], p["min_dur"]
        )
        evt = compute_event_metrics(y_true, y_pred)
        if evt["F1"] > best_f1:
            best_f1 = evt["F1"]
            best_params = p.copy()
            best_metrics = evt

    return best_params, best_metrics


def tune_scorenet_threshold(y_prob, y_true, sn_model, device, min_dur=10):
    """Sweep ScoreNet thresholds on val, select by event F1."""
    best_f1 = -1.0
    best_thr = 0.5
    best_metrics = None

    for thr in SN_THRESHOLDS:
        y_pred, _ = scorenet_postprocess(
            y_prob, sn_model, device, threshold=thr, min_dur_sec=min_dur
        )
        evt = compute_event_metrics(y_true, y_pred)
        if evt["F1"] > best_f1:
            best_f1 = evt["F1"]
            best_thr = thr
            best_metrics = evt

    return best_thr, best_metrics


# ── Full metrics on test ─────────────────────────────────────────────────────

def compute_full_metrics(y_prob, y_true, y_pred):
    """Compute point-wise, event, and SzCORE metrics for a given y_pred."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pw_f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    evt = compute_event_metrics(y_true, y_pred)
    szcore = compute_szcore_event_metrics(y_true, y_pred)

    return {
        "pw_f1": pw_f1,
        "sens": sens,
        "spec": spec,
        "prec": prec,
        "roc_auc": roc,
        "auprc": auprc,
        "evt_f1": evt["F1"],
        "evt_prec": evt["Precision"],
        "evt_recall": evt["Recall"],
        "far_hr": evt["FAR/hr"],
        "evt_tp": evt["TP"],
        "evt_fp": evt["FP"],
        "evt_fn": evt["FN"],
        "szcore_f1": szcore["F1"],
        "szcore_sens": szcore["Sensitivity"],
        "szcore_prec": szcore["Precision"],
        "szcore_far_hr": szcore["FAR/hr"],
    }


def _prefixed(metrics, prefix):
    """Return a dict with all metric keys prefixed, e.g. 'pp_test_pw_f1'."""
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _print_metrics(metrics, label):
    """Print a compact metrics summary for one evaluation mode."""
    print(f"  [{label}]  AUC={metrics['roc_auc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
          f"PW-F1={metrics['pw_f1']:.4f}  Sens={metrics['sens']:.4f}  Spec={metrics['spec']:.4f}")
    print(f"  {'':>{len(label)+4}}Evt-F1={metrics['evt_f1']:.4f}  Evt-P={metrics['evt_prec']:.4f}  "
          f"Evt-R={metrics['evt_recall']:.4f}  FAR/hr={metrics['far_hr']:.2f}  "
          f"(TP={metrics['evt_tp']} FP={metrics['evt_fp']} FN={metrics['evt_fn']})")
    print(f"  {'':>{len(label)+4}}SzCORE-F1={metrics['szcore_f1']:.4f}  "
          f"SzCORE-Sens={metrics['szcore_sens']:.4f}  SzCORE-Prec={metrics['szcore_prec']:.4f}  "
          f"SzCORE-FAR/hr={metrics['szcore_far_hr']:.2f}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_run(run_name, run_dir, args, input_chans, val_dset, test_dset, device):
    """Full pipeline for a single run: parse log -> load model -> tune on val -> eval on test."""
    print(f"\n{'='*70}")
    print(f"  Run: {run_name}")
    print(f"{'='*70}")

    # 1. Parse log
    log_entries = parse_log(os.path.join(run_dir, "log.txt"))
    if not log_entries:
        print("  [SKIP] Empty or unparseable log.txt")
        return None

    best_val, best_test, val_test_match = select_best_epoch(log_entries)
    if best_val is None:
        print("  [SKIP] No val_roc_auc found in log.txt")
        return None

    best_epoch = best_val["epoch"]
    print(f"  Best val epoch: {best_epoch}  (val_roc_auc={best_val.get('val_roc_auc', '?'):.4f})")
    if best_test:
        print(f"  Best test epoch: {best_test['epoch']}  (test_roc_auc={best_test.get('test_roc_auc', '?'):.4f})")
        if val_test_match:
            print("  >> Val-best and test-best epochs MATCH")
        else:
            print("  >> Val-best and test-best epochs DIFFER")

    # 2. Find checkpoint
    ckpt_path = find_checkpoint(run_dir, best_epoch)
    if ckpt_path is None:
        print(f"  [SKIP] No checkpoint found for epoch {best_epoch}")
        return None
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    # 3. Extract hyperparams
    hparams = extract_hyperparams(ckpt_path)
    is_adversarial = detect_adversarial(ckpt_path)
    hparams["adversarial"] = is_adversarial
    hp_str = ", ".join(f"{k}={v}" for k, v in hparams.items() if v != "N/A")
    print(f"  Hyperparams: {hp_str}")

    # 4. Load model
    model_args = SimpleNamespace(
        model="labram_base_patch200_200",
        checkpoint=ckpt_path,
        adversarial=is_adversarial,
        adv_hidden_dim=hparams.get("adv_hidden_dim", 512),
        device=str(device),
    )
    model = load_model(model_args, device)

    # 5. Inference on val
    print("  Running inference on val...")
    y_prob_val, y_true_val = run_inference(
        model, val_dset, input_chans, device, args.batch_size
    )

    # 6. Inference on test
    print("  Running inference on test...")
    y_prob_test, y_true_test = run_inference(
        model, test_dset, input_chans, device, args.batch_size
    )

    # 7. Free model memory
    del model
    torch.cuda.empty_cache()

    # 8. Tune hand-tuned PP on val
    print("  Tuning postprocessing on val...")
    best_pp, val_pp_metrics = tune_postprocessing(y_prob_val, y_true_val)
    if best_pp is None:
        print("  [WARN] PP tuning failed, using defaults")
        best_pp = {"t_high": 0.5, "t_low": 0.3, "smooth": 5, "min_dur": 5}
    print(f"  Best PP (val): t_high={best_pp['t_high']}, t_low={best_pp['t_low']}, "
          f"smooth={best_pp['smooth']}, min_dur={best_pp['min_dur']}  "
          f"(val evt_F1={val_pp_metrics['F1']:.4f})" if val_pp_metrics else "")

    # 9. Evaluate on test: raw threshold
    print("  Evaluating on test (raw threshold=0.5)...")
    y_pred_raw = (y_prob_test >= 0.5).astype(int)
    raw_metrics = compute_full_metrics(y_prob_test, y_true_test, y_pred_raw)
    _print_metrics(raw_metrics, "Raw thr=0.5")

    # 10. Evaluate on test: hand-tuned PP with frozen val params
    print("  Evaluating on test (hand-tuned PP)...")
    y_pred_pp = post_process_probs(
        y_prob_test, best_pp["t_high"], best_pp["t_low"],
        best_pp["smooth"], best_pp["min_dur"]
    )
    pp_metrics = compute_full_metrics(y_prob_test, y_true_test, y_pred_pp)
    _print_metrics(pp_metrics, "Hand-tuned PP")

    # 11. ScoreNet (optional)
    sn_metrics = None
    sn_thr = None
    sn_path = find_scorenet(run_dir, args.checkpoints_root)
    if sn_path:
        print(f"  Found ScoreNet: {sn_path}")
        try:
            sn_model = load_scorenet(sn_path, device)
            print("  Tuning ScoreNet threshold on val...")
            sn_thr, val_sn_metrics = tune_scorenet_threshold(
                y_prob_val, y_true_val, sn_model, device
            )
            print(f"  Best SN threshold (val): {sn_thr} (val evt_F1={val_sn_metrics['F1']:.4f})")

            print(f"  Evaluating on test (ScoreNet thr={sn_thr})...")
            y_pred_sn, _ = scorenet_postprocess(
                y_prob_test, sn_model, device, threshold=sn_thr, min_dur_sec=10
            )
            sn_metrics = compute_full_metrics(y_prob_test, y_true_test, y_pred_sn)
            _print_metrics(sn_metrics, f"ScoreNet thr={sn_thr}")
            del sn_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [WARN] ScoreNet evaluation failed: {e}")
    else:
        print("  No ScoreNet checkpoint found")

    # 12. Assemble result row — all metrics for every eval mode
    row = {
        "run": run_name,
        "adversarial": is_adversarial,
        "lr": hparams.get("lr", "N/A"),
        "pos_weight": hparams.get("pos_weight", "N/A"),
        "adv_lambda": hparams.get("adv_lambda", "N/A") if is_adversarial else "N/A",
        "adv_gamma": hparams.get("adv_gamma", "N/A") if is_adversarial else "N/A",
        "epochs": hparams.get("epochs", "N/A"),
        "seed": hparams.get("seed", "N/A"),
        "best_val_epoch": best_epoch,
        "val_test_match": val_test_match,
        "best_test_epoch": best_test["epoch"] if best_test else "N/A",
        "val_roc_auc": best_val.get("val_roc_auc", None),
        "val_f1": best_val.get("val_f1", None),
        "val_balanced_acc": best_val.get("val_balanced_accuracy", None),
        "log_test_roc_auc": best_val.get("test_roc_auc", None),
        "log_test_f1": best_val.get("test_f1", None),
    }

    row.update(_prefixed(raw_metrics, "raw_test_"))

    row.update({
        "pp_t_high": best_pp["t_high"],
        "pp_t_low": best_pp["t_low"],
        "pp_smooth": best_pp["smooth"],
        "pp_min_dur": best_pp["min_dur"],
    })
    row.update(_prefixed(pp_metrics, "pp_test_"))

    row["sn_checkpoint"] = os.path.basename(sn_path) if sn_path else "N/A"
    row["sn_threshold"] = sn_thr if sn_thr is not None else "N/A"
    _ALL_METRIC_KEYS = [
        "pw_f1", "sens", "spec", "prec", "roc_auc", "auprc",
        "evt_f1", "evt_prec", "evt_recall", "far_hr", "evt_tp", "evt_fp", "evt_fn",
        "szcore_f1", "szcore_sens", "szcore_prec", "szcore_far_hr",
    ]
    if sn_metrics:
        row.update(_prefixed(sn_metrics, "sn_test_"))
    else:
        for k in _ALL_METRIC_KEYS:
            row[f"sn_test_{k}"] = "N/A"

    return row


def print_summary(df):
    """Pretty-print a condensed summary to the terminal."""
    print(f"\n{'='*100}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*100}\n")

    cols = [
        "run", "adversarial", "best_val_epoch", "val_roc_auc",
        "raw_test_roc_auc", "raw_test_pw_f1",
        "pp_test_pw_f1", "pp_test_evt_f1", "pp_test_far_hr", "pp_test_szcore_f1",
        "sn_test_evt_f1", "sn_test_far_hr",
        "val_test_match",
    ]
    present = [c for c in cols if c in df.columns]
    summary = df[present].copy()

    float_cols = summary.select_dtypes(include=["float64", "float32"]).columns
    for c in float_cols:
        summary[c] = summary[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(summary.to_string(index=False))
    print()

    print("Detailed postprocessing params per run:")
    for _, r in df.iterrows():
        pp_str = (f"  {r['run']}: t_high={r.get('pp_t_high','?')}, "
                  f"t_low={r.get('pp_t_low','?')}, smooth={r.get('pp_smooth','?')}, "
                  f"min_dur={r.get('pp_min_dur','?')}")
        print(pp_str)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Centralized benchmarking for TUSZ / CHB-MIT seizure detection runs"
    )
    parser.add_argument("--checkpoints_root", required=True, type=str,
                        help="Root folder containing run subdirectories")
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to folder with train.h5, val.h5, test.h5")
    parser.add_argument("--dataset", default="TUSZ", type=str, choices=["TUSZ", "CHBMIT"],
                        help="Dataset type (determines channel names)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--output", default="results_summary.csv", type=str,
                        help="Output CSV path")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Only process these run names (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ch_names = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names)

    # Load datasets once
    print(f"Loading val dataset from {args.data_path}/val.h5 ...")
    val_dset = CHBMITDataset(os.path.join(args.data_path, "val.h5"))
    print(f"  {len(val_dset)} samples")

    print(f"Loading test dataset from {args.data_path}/test.h5 ...")
    test_dset = CHBMITDataset(os.path.join(args.data_path, "test.h5"))
    print(f"  {len(test_dset)} samples")

    # Discover runs
    all_runs = discover_runs(args.checkpoints_root)
    if args.runs:
        all_runs = [(name, path) for name, path in all_runs if name in args.runs]
    print(f"\nFound {len(all_runs)} run(s) to evaluate:")
    for name, _ in all_runs:
        print(f"  - {name}")

    # Process each run
    results = []
    for run_name, run_dir in all_runs:
        try:
            row = process_run(
                run_name, run_dir, args, input_chans,
                val_dset, test_dset, device
            )
            if row is not None:
                results.append(row)
        except Exception as e:
            print(f"\n  [ERROR] Run '{run_name}' failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("\nNo runs produced results.")
        return

    df = pd.DataFrame(results)

    # Sort by test event F1 (PP) descending
    if "pp_test_evt_f1" in df.columns:
        numeric = pd.to_numeric(df["pp_test_evt_f1"], errors="coerce")
        df = df.iloc[numeric.argsort()[::-1]].reset_index(drop=True)

    print_summary(df)

    df.to_csv(args.output, index=False)
    print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
