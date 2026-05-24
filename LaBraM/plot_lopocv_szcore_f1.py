#!/usr/bin/env python3
"""
Generate a grouped bar chart of per-patient SzCORE Event F1 across four
CHB-MIT LOPOCV configurations (Baseline, Channel Attention, Single GRL,
Multilayer GRL).

Optionally runs eval_lopocv_full.py for each configuration first, then
reads the resulting JSON files and produces a publication-quality plot.

Usage:
    # Full pipeline: evaluate + plot
    python plot_lopocv_szcore_f1.py

    # Plot only (JSONs already exist)
    python plot_lopocv_szcore_f1.py --skip_eval

    # Force re-evaluation even if JSONs exist
    python plot_lopocv_szcore_f1.py --force
"""

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = OrderedDict([
    ("Baseline", {
        "subdir": "finetune_chbmit_baseline_lopocv",
        "extra_args": [],
    }),
    ("Channel Attention", {
        "subdir": "finetune_chbmit_channel_attention_lopocv",
        "extra_args": [],
    }),
    ("Single GRL", {
        "subdir": "finetune_chbmit_lopo_cv",
        "extra_args": [],
    }),
    ("Multilayer GRL", {
        "subdir": "finetune_chbmit_multilayer_lopocv",
        "extra_args": ["--intermediate_layers", "3,7"],
    }),
])

JSON_FILENAME = "lopocv_full_eval_mindur0.json"
STRATEGY = "hand_tuned"
METRIC_KEY = "szcore_evt_f1"


def get_args():
    p = argparse.ArgumentParser(
        description="Per-patient SzCORE Event F1 bar chart for LOPOCV ablations"
    )
    p.add_argument(
        "--checkpoint_base",
        default="/home/jovyan/extra-data/local-data-prev/checkpoints",
        help="Parent directory containing all LOPOCV checkpoint folders",
    )
    p.add_argument(
        "--data_dir",
        default="/home/jovyan/extra-data/CHBMIT_per_patient",
        help="Directory with per-patient H5 files (chb01.h5, ...)",
    )
    p.add_argument(
        "--output",
        default="lopocv_szcore_f1_per_patient.pdf",
        help="Output path for the plot (PDF/PNG/SVG)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluation even if JSON already exists",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation; only plot from existing JSON files",
    )
    return p.parse_args()


def run_evaluations(args):
    """Run eval_lopocv_full.py for each configuration with min_dur=0."""
    eval_script = os.path.join(os.path.dirname(__file__), "eval_lopocv_full.py")

    for name, cfg in CONFIGS.items():
        lopocv_dir = os.path.join(args.checkpoint_base, cfg["subdir"])
        json_path = os.path.join(lopocv_dir, JSON_FILENAME)

        if os.path.isfile(json_path) and not args.force:
            print(f"[{name}] JSON already exists, skipping: {json_path}")
            continue

        if not os.path.isdir(lopocv_dir):
            print(f"[{name}] WARNING: directory not found: {lopocv_dir}", file=sys.stderr)
            continue

        cmd = [
            sys.executable, eval_script,
            "--lopocv_dir", lopocv_dir,
            "--data_dir", args.data_dir,
            "--min_dur", "0",
            "--sn_min_dur", "0",
            "--output", json_path,
            *cfg["extra_args"],
        ]

        print(f"\n{'='*80}")
        print(f"[{name}] Running evaluation ...")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[{name}] eval_lopocv_full.py exited with code {result.returncode}",
                  file=sys.stderr)


def load_per_patient_f1(args):
    """Load per-patient SzCORE Event F1 from each config's JSON."""
    data = OrderedDict()

    for name, cfg in CONFIGS.items():
        lopocv_dir = os.path.join(args.checkpoint_base, cfg["subdir"])
        json_path = os.path.join(lopocv_dir, JSON_FILENAME)

        if not os.path.isfile(json_path):
            print(f"[{name}] JSON not found, skipping: {json_path}", file=sys.stderr)
            continue

        with open(json_path) as f:
            results = json.load(f)

        if STRATEGY not in results:
            print(f"[{name}] Strategy '{STRATEGY}' not in JSON, skipping", file=sys.stderr)
            continue

        per_fold = results[STRATEGY]["per_fold"]
        patient_f1 = {
            r["patient"]: r.get(METRIC_KEY, 0.0) for r in per_fold
        }
        data[name] = patient_f1

    return data


def plot_bar_chart(data, output_path):
    """Create a grouped bar chart of SzCORE Event F1 per patient."""
    all_patients = sorted(
        set().union(*(pf.keys() for pf in data.values())),
        key=lambda p: int(p.replace("chb", ""))
    )
    config_names = list(data.keys())
    n_patients = len(all_patients)
    n_configs = len(config_names)

    labels = [p.replace("chb", "chb") for p in all_patients]

    values = np.zeros((n_configs, n_patients))
    for i, name in enumerate(config_names):
        for j, patient in enumerate(all_patients):
            values[i, j] = data[name].get(patient, 0.0)

    bar_width = 0.8 / n_configs
    x = np.arange(n_patients)

    fig, ax = plt.subplots(figsize=(16, 6))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, name in enumerate(config_names):
        offset = (i - (n_configs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values[i], bar_width,
            label=name, color=colors[i], edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Patient", fontsize=13, labelpad=8)
    ax.set_ylabel("SzCORE Event F1", fontsize=13, labelpad=8)
    ax.set_title(
        "Per-Patient SzCORE Event F1  —  CHB-MIT LOPOCV  (Hand-Tuned PP, min_dur = 0)",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fontsize=11,
    )

    means = [np.nanmean(values[i]) for i in range(n_configs)]
    medians = [np.nanmedian(values[i]) for i in range(n_configs)]

    stats_text = "\n".join(
        f"{name}: mean={means[i]:.3f}, median={medians[i]:.3f}"
        for i, name in enumerate(config_names)
    )
    ax.text(
        0.01, 0.98, stats_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def main():
    args = get_args()

    if not args.skip_eval:
        run_evaluations(args)

    data = load_per_patient_f1(args)

    if not data:
        print("ERROR: No data loaded. Check paths and run evaluation first.",
              file=sys.stderr)
        sys.exit(1)

    plot_bar_chart(data, args.output)

    print("\nSummary:")
    for name, patient_f1 in data.items():
        vals = list(patient_f1.values())
        print(f"  {name:>20s}:  {len(vals)} patients,  "
              f"mean={np.mean(vals):.4f},  median={np.median(vals):.4f},  "
              f"std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
