"""
Preprocesses the TUH Seizure Corpus (TUSZ) v2.0.x into HDF5 files
compatible with the CHBMITDataset loader.

Expected TUSZ directory layout:
    <data_root>/
    └── edf/
        ├── train/
        │   └── 01_tcp_ar/
        │       └── <patient>/
        │           └── <session>/
        │               └── <recording>/
        │                   ├── *.edf
        │                   └── *.tse_bi
        ├── dev/
        └── eval/

Output: <output_dir>/{train,val,test}.h5
    data   -> (N, 23, 400)  float32   (23 unipolar channels, 2 s @ 200 Hz)
    labels -> (N,)           int64     (0 = background, 1 = seizure)

Usage:
    python make_TUSZ.py --data_root /path/to/tusz_v2.0.0 --output_dir ./datasets/TUSZ
"""

import os
import argparse
import numpy as np
import mne
import h5py
from tqdm import tqdm

WINDOW_SIZE = 2      # seconds
STRIDE = 1           # seconds
TARGET_FREQ = 200    # Hz
NUM_CHANNELS = 23

# 23 standard unipolar channels (same order as TUAB / TUEV)
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2',
]

# Aliases for channels that may appear under different names across TUSZ EDFs
CHANNEL_ALIASES = {
    'FP1': ['FP1'],
    'FP2': ['FP2'],
    'F3':  ['F3'],
    'F4':  ['F4'],
    'C3':  ['C3'],
    'C4':  ['C4'],
    'P3':  ['P3'],
    'P4':  ['P4'],
    'O1':  ['O1'],
    'O2':  ['O2'],
    'F7':  ['F7'],
    'F8':  ['F8'],
    'T3':  ['T3', 'T7'],
    'T4':  ['T4', 'T8'],
    'T5':  ['T5', 'P7'],
    'T6':  ['T6', 'P8'],
    'A1':  ['A1'],
    'A2':  ['A2'],
    'FZ':  ['FZ'],
    'CZ':  ['CZ'],
    'PZ':  ['PZ'],
    'T1':  ['T1', 'FT9'],
    'T2':  ['T2', 'FT10'],
}

# TUSZ split dirs -> output filenames
SPLIT_MAP = {
    'train': 'train.h5',
    'dev':   'val.h5',
    'eval':  'test.h5',
}


def normalize_tuh_channel_name(raw_name):
    """
    Strips the TUH-style prefix/suffix from a channel name.
    'EEG FP1-REF' -> 'FP1', 'EEG FP1-LE' -> 'FP1', 'EEG T3-REF' -> 'T3'
    """
    name = raw_name.strip().upper()
    if name.startswith('EEG '):
        name = name[4:]
    for suffix in ('-REF', '-LE', '-AR'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()


def get_channel_mapping(raw_channels, target_channels):
    """
    Maps each target channel to an index in raw_channels using
    normalized names and the alias table.  Returns (mapping, missing)
    where mapping[i] is the raw index for target_channels[i].
    """
    normalized = [normalize_tuh_channel_name(c) for c in raw_channels]
    mapping = []
    missing = []
    used = set()

    for tgt in target_channels:
        aliases = CHANNEL_ALIASES.get(tgt, [tgt])
        found = False
        for alias in aliases:
            for idx, norm in enumerate(normalized):
                if norm == alias and idx not in used:
                    mapping.append(idx)
                    used.add(idx)
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(tgt)

    return mapping, missing


def parse_tse_bi(tse_path):
    """
    Parses a .tse_bi annotation file and returns a list of
    (start_sec, end_sec) tuples for seizure ('seiz') intervals.

    File format:
        version = tse_v1.0.0
        0.0000 12.5600 bckg 1.0000
        12.5600 19.3200 seiz 1.0000
        ...
    """
    intervals = []
    with open(tse_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('version') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2].lower()
            except (ValueError, IndexError):
                continue
            if label == 'seiz':
                intervals.append((start, end))
    return intervals


def find_edf_tse_pairs(split_dir):
    """
    Recursively walks a TUSZ split directory to find all (.edf, .tse_bi) pairs.
    Returns a list of (edf_path, tse_path) tuples.
    """
    pairs = []
    for root, _dirs, files in os.walk(split_dir):
        edf_files = [f for f in files if f.lower().endswith('.edf')]
        for edf_name in edf_files:
            base = os.path.splitext(edf_name)[0]
            tse_name = base + '.tse_bi'
            tse_path = os.path.join(root, tse_name)
            edf_path = os.path.join(root, edf_name)
            if os.path.isfile(tse_path):
                pairs.append((edf_path, tse_path))
            else:
                print(f"WARNING: no .tse_bi for {edf_path}, skipping")
    return sorted(pairs)


def process_file(edf_path, seizure_intervals, writer_dict):
    """
    Loads a single EDF, selects/reorders 23 unipolar channels, filters,
    resamples, segments into 2 s windows (1 s stride), labels each window
    by midpoint, and appends to the open HDF5 datasets.
    """
    try:
        with mne.utils.use_log_level('ERROR'):
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Failed to read {os.path.basename(edf_path)}: {e}")
        return 0

    mapping, missing = get_channel_mapping(raw.ch_names, STANDARD_CHANNELS)
    if missing:
        print(f"Skipping {os.path.basename(edf_path)}: missing channels {missing}")
        return 0

    raw.pick(mapping)
    raw.reorder_channels([raw.ch_names[i] for i in range(len(mapping))])

    try:
        raw.notch_filter(60.0, verbose=False)
        raw.filter(0.1, 75.0, verbose=False)
        if raw.info['sfreq'] != TARGET_FREQ:
            raw.resample(TARGET_FREQ, verbose=False)
        data = raw.get_data() * 1e6  # V -> µV
    except Exception as e:
        print(f"Processing error in {os.path.basename(edf_path)}: {e}")
        return 0

    n_samples = data.shape[1]
    window_pts = int(WINDOW_SIZE * TARGET_FREQ)
    stride_pts = int(STRIDE * TARGET_FREQ)

    segments = []
    labels = []

    for start in range(0, n_samples - window_pts + 1, stride_pts):
        end = start + window_pts
        t_mid = (start + end) / 2 / TARGET_FREQ

        label = 0
        for (s_start, s_end) in seizure_intervals:
            if s_start <= t_mid <= s_end:
                label = 1
                break

        segments.append(data[:, start:end])
        labels.append(label)

    if segments:
        dset_data = writer_dict['data']
        dset_labels = writer_dict['labels']

        curr_len = dset_data.shape[0]
        add_len = len(segments)

        dset_data.resize(curr_len + add_len, axis=0)
        dset_labels.resize(curr_len + add_len, axis=0)

        dset_data[curr_len:] = np.array(segments, dtype=np.float32)
        dset_labels[curr_len:] = np.array(labels, dtype=np.int64)

    return len(segments)


def main():
    parser = argparse.ArgumentParser(description='Preprocess TUSZ v2.0 into HDF5')
    parser.add_argument('--data_root', required=True, type=str,
                        help='Root of the TUSZ dataset (contains edf/ directory)')
    parser.add_argument('--output_dir', default='./datasets/TUSZ', type=str,
                        help='Where to write train.h5, val.h5, test.h5')
    parser.add_argument('--window_size', default=WINDOW_SIZE, type=int,
                        help='Window size in seconds (default: 2)')
    parser.add_argument('--stride', default=STRIDE, type=int,
                        help='Stride in seconds (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    edf_root = os.path.join(args.data_root, 'edf')
    if not os.path.isdir(edf_root):
        edf_root = args.data_root
        print(f"No edf/ subdirectory found; using {edf_root} directly")

    window_pts = int(args.window_size * TARGET_FREQ)

    for split_name, h5_name in SPLIT_MAP.items():
        split_dir = os.path.join(edf_root, split_name)
        if not os.path.isdir(split_dir):
            print(f"Split directory not found: {split_dir}, skipping")
            continue

        pairs = find_edf_tse_pairs(split_dir)
        print(f"\n{'='*60}")
        print(f"Split: {split_name} -> {h5_name}  ({len(pairs)} EDF files)")
        print(f"{'='*60}")

        h5_path = os.path.join(args.output_dir, h5_name)
        total_segments = 0
        total_seizure = 0

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset(
                'data',
                shape=(0, NUM_CHANNELS, window_pts),
                maxshape=(None, NUM_CHANNELS, window_pts),
                dtype='float32',
                chunks=(64, NUM_CHANNELS, window_pts),
            )
            f.create_dataset(
                'labels',
                shape=(0,),
                maxshape=(None,),
                dtype='int64',
                chunks=(64,),
            )

            writer = {'data': f['data'], 'labels': f['labels']}

            for edf_path, tse_path in tqdm(pairs, desc=split_name):
                seizure_intervals = parse_tse_bi(tse_path)
                n = process_file(edf_path, seizure_intervals, writer)
                total_segments += n

            total_seizure = int(np.sum(f['labels'][:]))

        print(f"  Total segments: {total_segments}")
        print(f"  Seizure segments: {total_seizure}  "
              f"({100*total_seizure/max(total_segments,1):.2f}%)")
        print(f"  Saved to {h5_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
