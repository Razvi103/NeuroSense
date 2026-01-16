import os
import glob
import re
import numpy as np
import mne
import h5py
import random
from tqdm import tqdm

DATA_ROOT = '/home/shadeform/work/chbmit' 
OUTPUT_DIR = './datasets/CHBMIT'
WINDOW_SIZE = 2     # Seconds
STRIDE = 1          # Seconds
TARGET_FREQ = 200   # Hz

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
                    end_line = lines[i+1]
                    end_sec = int(re.search(r'(\d+)\s*seconds', end_line).group(1))
                    intervals.append((start_sec, end_sec))
                except:
                    pass
        file_info[filename] = intervals
    return file_info

def get_channel_mapping(raw_channels, target_channels):
    """
    Intelligently maps target channels to raw channels, handling duplicates.
    """
    raw_upper = [c.upper().strip() for c in raw_channels]
    mapping = []
    missing = []
    used_indices = set()
    
    for tgt in target_channels:
        tgt_upper = tgt.upper().strip()
        candidates = []
        
        # Scan all raw channels for potential matches
        for idx, raw in enumerate(raw_upper):
            # Priority 1: Exact match
            if raw == tgt_upper:
                candidates.append((0, idx))
            # Priority 2: MNE Duplicate ( T8-P8-0 or T8-P8-1)
            elif raw.startswith(tgt_upper):
                suffix = raw[len(tgt_upper):]
                # Check if suffix is just a dash/number (duplicate indicator)
                if suffix.startswith('-') and suffix[1:].isdigit():
                    candidates.append((1, idx))
                    
        # Sort candidates: prefer exact matches first, then duplicates
        candidates.sort(key=lambda x: x[0])
        
        # Pick the first candidate that hasn't been used yet
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

def process_file(edf_path, seizure_intervals, writer_dict):
    try:
        # Load without verbose logs
        with mne.utils.use_log_level('ERROR'): 
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Failed to read {os.path.basename(edf_path)}: {e}")
        return

    # 1. Channel Selection
    mapping, missing = get_channel_mapping(raw.ch_names, Standard_Channels)
    
    if len(missing) > 0:

        return

    raw.pick(mapping)
    raw.reorder_channels([raw.ch_names[i] for i in range(len(mapping))])

    try:
        raw.notch_filter(60.0, verbose=False)
        raw.filter(0.1, 75.0, verbose=False)
        if raw.info['sfreq'] != TARGET_FREQ:
            raw.resample(TARGET_FREQ, verbose=False)
        data = raw.get_data() * 1e6 # Convert to uV
    except Exception as e:
        print(f"Processing error in {os.path.basename(edf_path)}: {e}")
        return

    # 3. Segmentation
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

    # 4. Write to HDF5
    if segments:
        dset_data = writer_dict['data']
        dset_labels = writer_dict['labels']
        
        curr_len = dset_data.shape[0]
        add_len = len(segments)
        
        dset_data.resize(curr_len + add_len, axis=0)
        dset_labels.resize(curr_len + add_len, axis=0)
        
        dset_data[curr_len:] = np.array(segments, dtype=np.float32)
        dset_labels[curr_len:] = np.array(labels, dtype=np.int64)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    patient_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, 'chb*')))
    patient_map = {}
    
    print("Scanning and grouping by Patient...")
    for p_dir in patient_dirs:
        if not os.path.isdir(p_dir): continue
        pid = os.path.basename(p_dir)
        
        summary_files = glob.glob(os.path.join(p_dir, '*-summary.txt'))
        if not summary_files: continue
        
        intervals_map = parse_summary_file(summary_files[0])
        edf_files = glob.glob(os.path.join(p_dir, '*.edf'))
        
        if pid not in patient_map: patient_map[pid] = []
        for edf in edf_files:
            fname = os.path.basename(edf)
            intervals = intervals_map.get(fname, [])
            patient_map[pid].append((edf, intervals))

    # --- SUBJECT INDEPENDENT SPLIT ---
    all_patients = list(patient_map.keys())
    # random.shuffle(all_patients) # Optional: shuffle patients
    
    n_patients = len(all_patients)
    n_train = int(n_patients * 0.8)
    n_val = int(n_patients * 0.1)
    
    train_pids = all_patients[:n_train]
    val_pids = all_patients[n_train:n_train+n_val]
    test_pids = all_patients[n_train+n_val:]
    
    print(f"Patients -> Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")
    
    splits = {'train': [], 'val': [], 'test': []}
    for pid in train_pids: splits['train'].extend(patient_map[pid])
    for pid in val_pids: splits['val'].extend(patient_map[pid])
    for pid in test_pids: splits['test'].extend(patient_map[pid])

    for split_name, files in splits.items():
        print(f"Processing {split_name} set ({len(files)} files)...")
        h5_path = os.path.join(OUTPUT_DIR, f'{split_name}.h5')
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', shape=(0, 23, int(WINDOW_SIZE*TARGET_FREQ)), 
                             maxshape=(None, 23, int(WINDOW_SIZE*TARGET_FREQ)), dtype='float32')
            f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype='int64')
            
            writer = {'data': f['data'], 'labels': f['labels']}
            for edf, intervals in tqdm(files):
                process_file(edf, intervals, writer)

if __name__ == '__main__':
    main()