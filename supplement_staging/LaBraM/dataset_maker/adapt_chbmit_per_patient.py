"""
Generate one HDF5 file per CHB-MIT patient for LOPOCV.

Reuses the same preprocessing pipeline as adapt_chbmit.py (channel mapping,
notch/bandpass filtering, resampling, 2-second windows with 1-second stride)
but writes each patient to its own file: output_dir/chb01.h5, chb02.h5, ...

Also writes patient_map.json with the global patient-to-int mapping and
per-patient metadata.

Usage:
    python dataset_maker/adapt_chbmit_per_patient.py \
        --data_root /path/to/CHB-MIT_Raw \
        --output_dir /path/to/CHBMIT_per_patient
"""

import os
import argparse
import glob
import json
import re

import h5py
import mne
import numpy as np
from tqdm import tqdm

WINDOW_SIZE = 2
STRIDE = 1
TARGET_FREQ = 200

Standard_Channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
]


def parse_summary_file(summary_path):
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    file_info = {}
    blocks = content.split('File Name: ')
    for block in blocks[1:]:
        lines = block.split('\n')
        filename = lines[0].strip()
        intervals = []
        for i, line in enumerate(lines):
            if "Start Time" in line and "Seizure" in line:
                try:
                    start_sec = int(re.search(r'(\d+)\s*seconds', line).group(1))
                    end_line = lines[i + 1]
                    end_sec = int(re.search(r'(\d+)\s*seconds', end_line).group(1))
                    intervals.append((start_sec, end_sec))
                except Exception:
                    pass
        file_info[filename] = intervals
    return file_info


def get_channel_mapping(raw_channels, target_channels):
    raw_upper = [c.upper().strip() for c in raw_channels]
    mapping = []
    missing = []
    used_indices = set()

    for tgt in target_channels:
        tgt_upper = tgt.upper().strip()
        candidates = []
        for idx, raw in enumerate(raw_upper):
            if raw == tgt_upper:
                candidates.append((0, idx))
            elif raw.startswith(tgt_upper):
                suffix = raw[len(tgt_upper):]
                if suffix.startswith('-') and suffix[1:].isdigit():
                    candidates.append((1, idx))
        candidates.sort(key=lambda x: x[0])
        found = False
        for _, idx in candidates:
            if idx not in used_indices:
                mapping.append(idx)
                used_indices.add(idx)
                found = True
                break
        if not found:
            missing.append(tgt)

    return mapping, missing


def process_file(edf_path, seizure_intervals, writer_dict, patient_int_id):
    try:
        with mne.utils.use_log_level('ERROR'):
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Failed to read {os.path.basename(edf_path)}: {e}")
        return 0

    mapping, missing = get_channel_mapping(raw.ch_names, Standard_Channels)
    if len(missing) > 0:
        return 0

    raw.pick(mapping)
    raw.reorder_channels([raw.ch_names[i] for i in range(len(mapping))])

    try:
        raw.notch_filter(60.0, verbose=False)
        raw.filter(0.1, 75.0, verbose=False)
        if raw.info['sfreq'] != TARGET_FREQ:
            raw.resample(TARGET_FREQ, verbose=False)
        data = raw.get_data() * 1e6
    except Exception as e:
        print(f"  Processing error in {os.path.basename(edf_path)}: {e}")
        return 0

    n_samples = data.shape[1]
    window_pts = int(WINDOW_SIZE * TARGET_FREQ)
    stride_pts = int(STRIDE * TARGET_FREQ)

    segments = []
    labels = []

    for start in range(0, n_samples - window_pts, stride_pts):
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
        dset_pids = writer_dict['patient_ids']

        curr_len = dset_data.shape[0]
        add_len = len(segments)

        dset_data.resize(curr_len + add_len, axis=0)
        dset_labels.resize(curr_len + add_len, axis=0)
        dset_pids.resize(curr_len + add_len, axis=0)

        dset_data[curr_len:] = np.array(segments, dtype=np.float32)
        dset_labels[curr_len:] = np.array(labels, dtype=np.int64)
        dset_pids[curr_len:] = np.full(add_len, patient_int_id, dtype=np.int64)

    return len(segments)


def main():
    parser = argparse.ArgumentParser(description='Generate per-patient H5 files for CHB-MIT LOPOCV')
    parser.add_argument('--data_root', type=str,
                        default='./data/CHB-MIT_Raw',
                        help='Root directory containing chbXX folders')
    parser.add_argument('--output_dir', type=str,
                        default='./data/CHBMIT_per_patient',
                        help='Output directory for per-patient H5 files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patient_dirs = sorted(glob.glob(os.path.join(args.data_root, 'chb*')))
    patient_map = {}

    print("Scanning and grouping by patient...")
    for p_dir in patient_dirs:
        if not os.path.isdir(p_dir):
            continue
        pid = os.path.basename(p_dir)
        summary_files = glob.glob(os.path.join(p_dir, '*-summary.txt'))
        if not summary_files:
            continue
        intervals_map = parse_summary_file(summary_files[0])
        edf_files = glob.glob(os.path.join(p_dir, '*.edf'))
        if pid not in patient_map:
            patient_map[pid] = []
        for edf in edf_files:
            fname = os.path.basename(edf)
            intervals = intervals_map.get(fname, [])
            patient_map[pid].append((edf, intervals))

    all_patients = sorted(patient_map.keys())
    patient_to_int = {pid: i for i, pid in enumerate(all_patients)}
    print(f"Total unique patients: {len(all_patients)}")

    window_pts = int(WINDOW_SIZE * TARGET_FREQ)
    meta = {"patient_to_int": patient_to_int, "patients": {}}

    for pid in all_patients:
        int_id = patient_to_int[pid]
        edf_list = patient_map[pid]
        h5_path = os.path.join(args.output_dir, f'{pid}.h5')
        print(f"\nProcessing {pid} ({len(edf_list)} files) -> {h5_path}")

        total_segments = 0
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', shape=(0, 23, window_pts),
                             maxshape=(None, 23, window_pts), dtype='float32',
                             chunks=(64, 23, window_pts))
            f.create_dataset('labels', shape=(0,), maxshape=(None,),
                             dtype='int64', chunks=(64,))
            f.create_dataset('patient_ids', shape=(0,), maxshape=(None,),
                             dtype='int64', chunks=(64,))
            f.attrs['patient_to_int'] = json.dumps(patient_to_int)
            f.attrs['patient_id'] = pid
            f.attrs['patient_int_id'] = int_id

            writer = {
                'data': f['data'],
                'labels': f['labels'],
                'patient_ids': f['patient_ids'],
            }
            for edf_path, intervals in tqdm(edf_list, desc=pid):
                n = process_file(edf_path, intervals, writer, int_id)
                total_segments += n

            total_seizure = int(np.sum(f['labels'][:])) if total_segments > 0 else 0

        seizure_pct = 100 * total_seizure / max(total_segments, 1)
        print(f"  Segments: {total_segments}  |  Seizure: {total_seizure} ({seizure_pct:.2f}%)")

        meta["patients"][pid] = {
            "int_id": int_id,
            "n_segments": total_segments,
            "n_seizure": total_seizure,
            "seizure_pct": round(seizure_pct, 2),
            "n_edf_files": len(edf_list),
        }

    meta_path = os.path.join(args.output_dir, 'patient_map.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved patient_map.json to {meta_path}")
    print("Done.")


if __name__ == '__main__':
    main()
