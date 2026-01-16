import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
from tqdm import tqdm
import utils
from timm.models import create_model
import modeling_finetune

# --- 1. Dataset Class ---
class CHBMITDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = len(self.h5_file['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.h5_file['data'][idx]
        label = self.h5_file['labels'][idx]
        data = torch.from_numpy(data).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=1e4, neginf=-1e4)
        return data, label

def get_events(binary_arr):
    """Finds start and end indices of contiguous '1' blocks."""
    events = []
    if len(binary_arr) == 0:
        return events
    
    padded = np.concatenate(([0], binary_arr, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    for s, e in zip(starts, ends):
        events.append((s, e))
    return events

def post_process_probs(probs, t_high, t_low, smooth_window, min_duration):
    """Applies temporal smoothing, dual-thresholding, and duration filtering."""
    if smooth_window > 1:
        probs_smooth = pd.Series(probs).rolling(window=smooth_window, center=True).mean().fillna(0).values
    else:
        probs_smooth = probs

    # Dual Thresholding
    preds = np.zeros_like(probs_smooth, dtype=int)
    in_seizure = False
    
    for i, p in enumerate(probs_smooth):
        if not in_seizure:
            if p >= t_high:
                in_seizure = True
                preds[i] = 1
        else:
            if p >= t_low:
                preds[i] = 1
            else:
                in_seizure = False
    
    final_preds = preds.copy()
    events = get_events(final_preds)
    for s, e in events:
        duration = e - s
        if duration < min_duration:
            final_preds[s:e] = 0 

    return final_preds

def compute_event_metrics(y_true, y_pred, stride_sec=1.0):
    """Computes event-level metrics."""
    true_events = get_events(y_true)
    pred_events = get_events(y_pred)
    
    tp_events = 0
    fp_events = 0
    fn_events = 0
    
    for t_start, t_end in true_events:
        detected = False
        for p_start, p_end in pred_events:
            if max(t_start, p_start) < min(t_end, p_end):
                detected = True
                break
        if detected:
            tp_events += 1
        else:
            fn_events += 1
    
    for p_start, p_end in pred_events:
        is_true = False
        for t_start, t_end in true_events:
            if max(t_start, p_start) < min(t_end, p_end):
                is_true = True
                break
        if not is_true:
            fp_events += 1

    recall = tp_events / len(true_events) if len(true_events) > 0 else 0.0
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    total_hours = len(y_true) * stride_sec / 3600.0
    far_per_hour = fp_events / total_hours if total_hours > 0 else 0.0

    return {
        "Total Seizures": len(true_events),
        "TP": tp_events,
        "FN": fn_events,
        "FP": fp_events,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "FAR/hr": far_per_hour
    }

@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device)
    
    print(f"Loading Model: {args.model}")
    model = create_model(
        args.model, pretrained=False, num_classes=1, 
        drop_rate=0.0, drop_path_rate=0.1, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1
    )
    
    print(f"Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    clean_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()

    ch_names_raw = [
        'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
        'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
        'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
    ]
    input_chans = utils.get_input_chans(ch_names_raw)

    print("Loading Datasets...")
    try:
        test_dset = CHBMITDataset(args.data_path + '/test.h5')
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"\n{'='*20}\nEvaluating on Test Set ({len(test_dset)} samples)\n{'='*20}")
    loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    all_probs = []
    all_targets = []
    
    for samples, targets in tqdm(loader, desc="Inference"):
        samples = samples.to(device)
        samples = samples / 100.0
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        with torch.cuda.amp.autocast():
            output = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(output).squeeze()
        
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())
        
    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)

    print(f"\n--- Post-Processing ---")
    print(f"Params: T_High={args.t_high}, T_Low={args.t_low}, Smooth={args.smooth}s, MinDur={args.min_dur}s")
    
    y_pred_pp = post_process_probs(
        y_prob, 
        t_high=args.t_high, 
        t_low=args.t_low, 
        smooth_window=args.smooth, 
        min_duration=args.min_dur
    )
    
    print("\n--- Point-Wise Metrics ---")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_pp).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_pp, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision (PPV):      {precision:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"AUPRC:                {pr_auc:.4f}")
    print(f"AUROC:                {roc_auc:.4f}")
    print(f"Confusion Matrix:     [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    print("\n--- Event-Based Metrics ---")
    pp_evt = compute_event_metrics(y_true, y_pred_pp)
    
    print(f"Total Seizures (GT): {pp_evt['Total Seizures']}")
    print(f"Detected (TP):       {pp_evt['TP']} \t({pp_evt['Recall']*100:.1f}%)")
    print(f"Missed (FN):         {pp_evt['FN']}")
    print(f"False Alarms (FP):   {pp_evt['FP']}")
    print(f"False Alarms/Hr:     {pp_evt['FAR/hr']:.4f}")
    print(f"Event Precision:     {pp_evt['Precision']:.4f}")
    print(f"Event F1:            {pp_evt['F1']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-best.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Post-Processing Parameters
    parser.add_argument('--t_high', default=0.40, type=float, help='High threshold for seizure trigger')
    parser.add_argument('--t_low', default=0.20, type=float, help='Low threshold for seizure continuation')
    parser.add_argument('--smooth', default=5, type=int, help='Smoothing window size (seconds)')
    parser.add_argument('--min_dur', default=5, type=int, help='Minimum seizure duration (seconds)')
    
    args = parser.parse_args()
    run_eval(args)