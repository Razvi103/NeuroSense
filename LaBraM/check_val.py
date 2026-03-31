"""
Analyze and compare TUSZ train/val/test HDF5 splits.

Usage:
    python analyze_tusz_splits.py --data_dir ./datasets/TUSZ
"""

import argparse
import json
import h5py
import numpy as np
from collections import defaultdict


def analyze_split(h5_path, split_name):
    """Compute per-split and per-patient statistics."""
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels'][:]
        patient_ids = f['patient_ids'][:]

        patient_map_raw = f.attrs.get('patient_to_int', '{}')
        if isinstance(patient_map_raw, bytes):
            patient_map_raw = patient_map_raw.decode()
        patient_to_int = json.loads(patient_map_raw)
        int_to_patient = {v: k for k, v in patient_to_int.items()}

    total = len(labels)
    n_seizure = int(np.sum(labels == 1))
    n_background = total - n_seizure
    unique_pids = np.unique(patient_ids)

    per_patient = {}
    for pid in unique_pids:
        mask = patient_ids == pid
        pat_labels = labels[mask]
        pat_total = len(pat_labels)
        pat_seiz = int(np.sum(pat_labels == 1))
        pat_ratio = pat_seiz / pat_total if pat_total > 0 else 0.0

        events = get_events(pat_labels)
        n_seizure_events = len(events)
        event_durations = [e - s for s, e in events]

        per_patient[pid] = {
            'patient_str': int_to_patient.get(int(pid), f'?{pid}'),
            'total_windows': pat_total,
            'seizure_windows': pat_seiz,
            'background_windows': pat_total - pat_seiz,
            'seizure_ratio': pat_ratio,
            'has_seizure': pat_seiz > 0,
            'n_seizure_events': n_seizure_events,
            'event_durations': event_durations,
        }

    return {
        'split_name': split_name,
        'total_windows': total,
        'seizure_windows': n_seizure,
        'background_windows': n_background,
        'seizure_ratio': n_seizure / total if total > 0 else 0.0,
        'n_patients': len(unique_pids),
        'per_patient': per_patient,
    }


def get_events(labels):
    """Find contiguous blocks of 1s. Returns list of (start, end)."""
    padded = np.concatenate(([0], labels, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))


def print_split_summary(stats):
    name = stats['split_name']
    pp = stats['per_patient']

    patients_with_seizure = [p for p in pp.values() if p['has_seizure']]
    patients_without = [p for p in pp.values() if not p['has_seizure']]
    all_ratios = [p['seizure_ratio'] for p in pp.values()]
    seiz_ratios = [p['seizure_ratio'] for p in patients_with_seizure] if patients_with_seizure else [0]

    all_events = []
    all_durations = []
    for p in pp.values():
        all_events.append(p['n_seizure_events'])
        all_durations.extend(p['event_durations'])

    total_events = sum(all_events)

    print(f"\n{'='*65}")
    print(f"  {name.upper()}")
    print(f"{'='*65}")
    print(f"  Total windows:          {stats['total_windows']:>10,}")
    print(f"  Seizure windows:        {stats['seizure_windows']:>10,}  ({stats['seizure_ratio']*100:.2f}%)")
    print(f"  Background windows:     {stats['background_windows']:>10,}  ({(1-stats['seizure_ratio'])*100:.2f}%)")
    print(f"  Class imbalance ratio:  1:{stats['background_windows']//max(stats['seizure_windows'],1)}")
    print()
    print(f"  Patients:               {stats['n_patients']:>10}")
    print(f"    with seizures:        {len(patients_with_seizure):>10}  ({len(patients_with_seizure)/stats['n_patients']*100:.1f}%)")
    print(f"    without seizures:     {len(patients_without):>10}  ({len(patients_without)/stats['n_patients']*100:.1f}%)")
    print()
    print(f"  Seizure events (total): {total_events:>10}")
    if all_durations:
        durations_sec = np.array(all_durations)  # each window = 1s stride
        print(f"    mean duration:        {np.mean(durations_sec):>10.1f} windows (~seconds)")
        print(f"    median duration:      {np.median(durations_sec):>10.1f}")
        print(f"    min duration:         {np.min(durations_sec):>10}")
        print(f"    max duration:         {np.max(durations_sec):>10}")
        print(f"    std duration:         {np.std(durations_sec):>10.1f}")
    print()
    print(f"  Per-patient seizure ratio:")
    print(f"    mean:                 {np.mean(all_ratios)*100:>10.2f}%")
    print(f"    median:               {np.median(all_ratios)*100:>10.2f}%")
    print(f"    std:                  {np.std(all_ratios)*100:>10.2f}%")
    print(f"    min:                  {np.min(all_ratios)*100:>10.2f}%")
    print(f"    max:                  {np.max(all_ratios)*100:>10.2f}%")

    if patients_with_seizure:
        print(f"  Per-patient seizure ratio (seizure patients only):")
        print(f"    mean:                 {np.mean(seiz_ratios)*100:>10.2f}%")
        print(f"    median:               {np.median(seiz_ratios)*100:>10.2f}%")
        print(f"    max:                  {np.max(seiz_ratios)*100:>10.2f}%")


def print_comparison_table(all_stats):
    print(f"\n{'='*65}")
    print(f"  CROSS-SPLIT COMPARISON")
    print(f"{'='*65}")

    header = f"{'Metric':<30}"
    for s in all_stats:
        header += f" {s['split_name']:>10}"
    print(header)
    print("-" * (30 + 11 * len(all_stats)))

    def row(label, values, fmt=">10"):
        line = f"{label:<30}"
        for v in values:
            line += f" {v:{fmt}}"
        print(line)

    row("Total windows",
        [f"{s['total_windows']:,}" for s in all_stats], ">10")
    row("Seizure windows",
        [f"{s['seizure_windows']:,}" for s in all_stats], ">10")
    row("Seizure %",
        [f"{s['seizure_ratio']*100:.2f}%" for s in all_stats], ">10")
    row("Imbalance (1:N)",
        [f"1:{s['background_windows']//max(s['seizure_windows'],1)}" for s in all_stats], ">10")
    row("Patients",
        [str(s['n_patients']) for s in all_stats], ">10")
    row("Patients w/ seizure",
        [str(len([p for p in s['per_patient'].values() if p['has_seizure']])) for s in all_stats], ">10")

    all_events = []
    for s in all_stats:
        total_ev = sum(p['n_seizure_events'] for p in s['per_patient'].values())
        all_events.append(total_ev)
    row("Seizure events",
        [str(e) for e in all_events], ">10")

    for s in all_stats:
        durations = []
        for p in s['per_patient'].values():
            durations.extend(p['event_durations'])
        s['_all_durations'] = durations

    row("Mean event dur (win)",
        [f"{np.mean(s['_all_durations']):.1f}" if s['_all_durations'] else "N/A" for s in all_stats], ">10")
    row("Median event dur (win)",
        [f"{np.median(s['_all_durations']):.1f}" if s['_all_durations'] else "N/A" for s in all_stats], ">10")

    print()

    # Check for patient overlap
    print("  Patient overlap:")
    split_patients = {}
    for s in all_stats:
        pats = set(p['patient_str'] for p in s['per_patient'].values())
        split_patients[s['split_name']] = pats

    names = list(split_patients.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            overlap = split_patients[names[i]] & split_patients[names[j]]
            print(f"    {names[i]} ∩ {names[j]}: {len(overlap)} patients", end="")
            if overlap:
                print(f"  ⚠️  OVERLAP: {sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
            else:
                print(f"  ✓ no overlap")


def print_per_patient_detail(all_stats):
    print(f"\n{'='*65}")
    print(f"  PER-PATIENT DETAIL (sorted by seizure ratio)")
    print(f"{'='*65}")

    for s in all_stats:
        print(f"\n--- {s['split_name'].upper()} ---")
        patients = sorted(s['per_patient'].values(),
                          key=lambda p: p['seizure_ratio'], reverse=True)

        print(f"  {'Patient':<12} {'Total':>8} {'Seizure':>8} {'Ratio':>8} {'Events':>7} {'MeanDur':>8}")
        print(f"  {'-'*55}")
        for p in patients:
            mean_dur = np.mean(p['event_durations']) if p['event_durations'] else 0
            print(f"  {p['patient_str']:<12} {p['total_windows']:>8,} {p['seizure_windows']:>8,} "
                  f"{p['seizure_ratio']*100:>7.2f}% {p['n_seizure_events']:>7} {mean_dur:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze TUSZ HDF5 splits')
    parser.add_argument('--data_dir', default='./datasets/TUSZ', type=str,
                        help='Directory containing train.h5, val.h5, test.h5')
    parser.add_argument('--detail', action='store_true',
                        help='Print per-patient detail tables')
    args = parser.parse_args()

    splits = [
        ('train.h5', 'train'),
        ('val.h5', 'val'),
        ('test.h5', 'test'),
    ]

    all_stats = []
    for filename, name in splits:
        path = f"{args.data_dir}/{filename}"
        try:
            stats = analyze_split(path, name)
            all_stats.append(stats)
        except FileNotFoundError:
            print(f"⚠️  {path} not found, skipping")

    for stats in all_stats:
        print_split_summary(stats)

    if len(all_stats) > 1:
        print_comparison_table(all_stats)

    if args.detail:
        print_per_patient_detail(all_stats)
    else:
        print(f"\n  (Run with --detail to see per-patient breakdowns)")


if __name__ == '__main__':
    main()