"""
Preprocesses the TUH Seizure Corpus (TUSZ) v2.0.3+ into HDF5 files
compatible with the CHBMITDataset loader.

Expected TUSZ directory layout:
    <data_root>/
    └── edf/
        ├── train/
        │   └── <patient>/
        │       └── <session>/
        │           └── <montage>/
        │               ├── *.edf
        │               ├── *.csv       (per-channel annotations)
        │               └── *.csv_bi    (binary seizure/background)
        ├── dev/
        └── eval/

Output: <output_dir>/{train,val,test}.h5
    data          -> (N, 19, 400)  float32   (19 unipolar 10-20 channels, 2 s @ 200 Hz)
    labels        -> (N,)          int64     (0 = background, 1 = seizure)
    patient_ids   -> (N,)          int64     (integer patient ID for adversarial training)
    recording_ids -> (N,)          int64     (integer recording ID for per-recording metrics)

Usage:
    python make_TUSZ.py --data_root /path/to/tusz_v2.0.3 --output_dir ./datasets/TUSZ
    python make_TUSZ.py --data_root /path/to/tusz_v2.0.3 --output_dir ./datasets/TUSZ --num_workers 24
"""

import os
import csv
import json
import argparse
import numpy as np
import mne
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

WINDOW_SIZE = 2      # seconds
STRIDE = 1           # seconds
TARGET_FREQ = 200    # Hz
NUM_CHANNELS = 19

# 19 standard 10-20 unipolar channels (matches TUSZ_CH_NAMES in evaluate_checkpoint.py)
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]

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
    'FZ':  ['FZ'],
    'CZ':  ['CZ'],
    'PZ':  ['PZ'],
}

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


def parse_csv_bi(csv_bi_path):
    """
    Parses a TUSZ v2.0.3 .csv_bi annotation file and returns a list of
    (start_sec, end_sec) tuples for seizure ('seiz') intervals.
    """
    intervals = []
    with open(csv_bi_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if row[0].strip().lower() == 'channel':
                continue
            if len(row) < 4:
                continue
            try:
                start = float(row[1])
                end = float(row[2])
                label = row[3].strip().lower()
            except (ValueError, IndexError):
                continue
            if label == 'seiz':
                intervals.append((start, end))
    return intervals


def extract_patient_id(edf_path, split_dir):
    """
    Extracts the patient folder name from the TUSZ directory hierarchy.
    Path structure: .../split/<patient>/<session>/<montage>/<file>.edf
    Returns the <patient> string (e.g. 'aaaaagus').
    """
    rel = os.path.relpath(edf_path, split_dir)
    parts = rel.replace('\\', '/').split('/')
    return parts[0]


def find_edf_annotation_pairs(split_dir):
    """
    Recursively walks a TUSZ split directory to find all (.edf, .csv_bi) pairs.
    Returns a list of (edf_path, csv_bi_path, patient_str) tuples.
    """
    pairs = []
    for root, _dirs, files in os.walk(split_dir):
        edf_files = [f for f in files if f.lower().endswith('.edf')]
        for edf_name in edf_files:
            base = os.path.splitext(edf_name)[0]
            csv_bi_name = base + '.csv_bi'
            csv_bi_path = os.path.join(root, csv_bi_name)
            edf_path = os.path.join(root, edf_name)
            if os.path.isfile(csv_bi_path):
                patient_str = extract_patient_id(edf_path, split_dir)
                pairs.append((edf_path, csv_bi_path, patient_str))
            else:
                print(f"WARNING: no .csv_bi for {edf_path}, skipping")
    return sorted(pairs, key=lambda x: x[0])


def _process_single_edf(edf_path, csv_bi_path, patient_int_id, recording_int_id,
                         window_size, stride, target_freq):
    """
    Worker function: reads one EDF, filters, resamples, windows, and labels.
    Returns (segments, labels, patient_int_id, recording_int_id, n_windows)
    or None on failure.
    """
    window_pts = int(window_size * target_freq)
    stride_pts = int(stride * target_freq)

    try:
        with mne.utils.use_log_level('ERROR'):
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        return None, f"Failed to read {os.path.basename(edf_path)}: {e}"

    mapping, missing = get_channel_mapping(raw.ch_names, STANDARD_CHANNELS)
    if missing:
        return None, f"Skipping {os.path.basename(edf_path)}: missing channels {missing}"

    raw.pick(mapping)
    raw.reorder_channels([raw.ch_names[i] for i in range(len(mapping))])

    try:
        raw.notch_filter(60.0, verbose=False, n_jobs=1)
        raw.filter(0.1, 75.0, verbose=False, n_jobs=1)
        if raw.info['sfreq'] != target_freq:
            raw.resample(target_freq, verbose=False, n_jobs=1)
        data = raw.get_data() * 1e6  # V -> uV
    except Exception as e:
        return None, f"Processing error in {os.path.basename(edf_path)}: {e}"

    n_samples = data.shape[1]
    if n_samples < window_pts:
        return None, f"Recording too short ({n_samples} samples) in {os.path.basename(edf_path)}"

    # Vectorized windowing via stride_tricks
    n_windows = (n_samples - window_pts) // stride_pts + 1
    byte_stride = data.strides[1]
    segments = np.lib.stride_tricks.as_strided(
        data,
        shape=(NUM_CHANNELS, n_windows, window_pts),
        strides=(data.strides[0], stride_pts * byte_stride, byte_stride),
    ).copy()
    segments = segments.transpose(1, 0, 2).astype(np.float32)  # (n_windows, C, T)

    # Vectorized labeling: build seizure mask then sample at midpoints
    seizure_intervals = parse_csv_bi(csv_bi_path)
    starts = np.arange(n_windows) * stride_pts
    midpoints_sec = (starts + window_pts / 2) / target_freq

    labels = np.zeros(n_windows, dtype=np.int64)
    for s_start, s_end in seizure_intervals:
        labels |= ((midpoints_sec >= s_start) & (midpoints_sec <= s_end)).astype(np.int64)

    result = {
        'segments': segments,
        'labels': labels,
        'patient_id': patient_int_id,
        'recording_id': recording_int_id,
        'n_windows': n_windows,
    }
    return result, None


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
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of parallel workers (0 = auto, 1 = sequential)')
    args = parser.parse_args()

    if args.num_workers <= 0:
        args.num_workers = min(os.cpu_count() or 1, 24)

    os.makedirs(args.output_dir, exist_ok=True)

    edf_root = os.path.join(args.data_root, 'edf')
    if not os.path.isdir(edf_root):
        edf_root = args.data_root
        print(f"No edf/ subdirectory found; using {edf_root} directly")

    window_pts = int(args.window_size * TARGET_FREQ)

    # Collect all patient strings across splits for a global mapping
    all_patient_strs = set()
    split_pairs = {}
    for split_name in SPLIT_MAP:
        split_dir = os.path.join(edf_root, split_name)
        if not os.path.isdir(split_dir):
            continue
        pairs = find_edf_annotation_pairs(split_dir)
        split_pairs[split_name] = pairs
        for _, _, patient_str in pairs:
            all_patient_strs.add(patient_str)

    patient_to_int = {p: i for i, p in enumerate(sorted(all_patient_strs))}
    print(f"\nTotal unique patients across all splits: {len(patient_to_int)}")
    print(f"Using {args.num_workers} workers for parallel EDF processing")

    # Global recording ID counter (unique across all splits)
    recording_counter = 0

    for split_name, h5_name in SPLIT_MAP.items():
        if split_name not in split_pairs:
            print(f"Split directory not found for {split_name}, skipping")
            continue

        pairs = split_pairs[split_name]
        print(f"\n{'='*60}")
        print(f"Split: {split_name} -> {h5_name}  ({len(pairs)} EDF files)")
        print(f"{'='*60}")

        h5_path = os.path.join(args.output_dir, h5_name)

        # Assign recording IDs and patient IDs for each file
        file_jobs = []
        for edf_path, csv_bi_path, patient_str in pairs:
            pid = patient_to_int[patient_str]
            file_jobs.append((edf_path, csv_bi_path, pid, recording_counter))
            recording_counter += 1

        # Estimate total windows for pre-allocation (assume ~1 hr avg -> ~3600 windows)
        est_windows = len(file_jobs) * 3600

        # Phase 1: parallel EDF processing
        all_results = [None] * len(file_jobs)
        n_skipped = 0

        if args.num_workers == 1:
            for i, (edf_path, csv_bi_path, pid, rid) in enumerate(
                tqdm(file_jobs, desc=f"{split_name} (sequential)")
            ):
                result, err = _process_single_edf(
                    edf_path, csv_bi_path, pid, rid,
                    args.window_size, args.stride, TARGET_FREQ,
                )
                if err:
                    tqdm.write(err)
                    n_skipped += 1
                else:
                    all_results[i] = result
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                for i, (edf_path, csv_bi_path, pid, rid) in enumerate(file_jobs):
                    fut = executor.submit(
                        _process_single_edf,
                        edf_path, csv_bi_path, pid, rid,
                        args.window_size, args.stride, TARGET_FREQ,
                    )
                    futures[fut] = i

                pbar = tqdm(total=len(futures), desc=f"{split_name} (parallel)")
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        result, err = fut.result()
                    except Exception as exc:
                        tqdm.write(f"Worker exception: {exc}")
                        n_skipped += 1
                        pbar.update(1)
                        continue
                    if err:
                        tqdm.write(err)
                        n_skipped += 1
                    else:
                        all_results[idx] = result
                    pbar.update(1)
                pbar.close()

        # Count actual total windows
        valid_results = [r for r in all_results if r is not None]
        total_windows = sum(r['n_windows'] for r in valid_results)

        if total_windows == 0:
            print(f"  No valid windows produced, skipping H5 creation")
            continue

        # Phase 2: sequential H5 write with pre-allocation
        print(f"  Writing {total_windows} windows from {len(valid_results)} "
              f"recordings ({n_skipped} skipped) ...")

        with h5py.File(h5_path, 'w') as f:
            dset_data = f.create_dataset(
                'data',
                shape=(total_windows, NUM_CHANNELS, window_pts),
                dtype='float32',
                chunks=(512, NUM_CHANNELS, window_pts),
            )
            dset_labels = f.create_dataset(
                'labels',
                shape=(total_windows,),
                dtype='int64',
                chunks=(512,),
            )
            dset_pids = f.create_dataset(
                'patient_ids',
                shape=(total_windows,),
                dtype='int64',
                chunks=(512,),
            )
            dset_rids = f.create_dataset(
                'recording_ids',
                shape=(total_windows,),
                dtype='int64',
                chunks=(512,),
            )

            f.attrs['patient_to_int'] = json.dumps(patient_to_int)

            # Write results in submission order (preserves file ordering)
            offset = 0
            for r in all_results:
                if r is None:
                    continue
                n = r['n_windows']
                dset_data[offset:offset + n] = r['segments']
                dset_labels[offset:offset + n] = r['labels']
                dset_pids[offset:offset + n] = r['patient_id']
                dset_rids[offset:offset + n] = r['recording_id']
                offset += n

            total_seizure = int(np.sum(dset_labels[:]))
            unique_patients = len(set(dset_pids[:].tolist()))
            unique_recordings = len(set(dset_rids[:].tolist()))

        print(f"  Total segments: {total_windows}")
        print(f"  Seizure segments: {total_seizure}  "
              f"({100*total_seizure/max(total_windows,1):.2f}%)")
        print(f"  Unique patients: {unique_patients}")
        print(f"  Unique recordings: {unique_recordings}")
        print(f"  Saved to {h5_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
