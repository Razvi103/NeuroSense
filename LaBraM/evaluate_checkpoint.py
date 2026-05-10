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
from modeling_finetune import AdversarialNeuralTransformer
from scorenet import ScoreNet, hard_constraints, build_toeplitz

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
    """Computes event-level metrics (original any-overlap, no tolerances)."""
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


# ---------------------------------------------------------------------------
# SzCORE-compliant event-based scoring
# ---------------------------------------------------------------------------

def _apply_tolerances(events, pre_ictal_samples, post_ictal_samples, total_len):
    """Widen reference events by pre/post-ictal tolerance (in sample indices)."""
    widened = []
    for s, e in events:
        new_s = max(0, s - pre_ictal_samples)
        new_e = min(total_len, e + post_ictal_samples)
        widened.append((new_s, new_e))
    return widened


def _merge_close_events(events, gap_samples):
    """Merge events separated by fewer than *gap_samples* into one."""
    if not events:
        return []
    sorted_ev = sorted(events, key=lambda x: x[0])
    merged = [sorted_ev[0]]
    for s, e in sorted_ev[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < gap_samples:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _split_long_events(events, max_samples):
    """Split events longer than *max_samples* into consecutive chunks."""
    split = []
    for s, e in events:
        while e - s > max_samples:
            split.append((s, s + max_samples))
            s = s + max_samples
        if e > s:
            split.append((s, e))
    return split


def compute_szcore_event_metrics(
    y_true,
    y_pred,
    stride_sec=1.0,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
    merge_gap_sec=90.0,
    max_event_sec=300.0,
):
    """SzCORE-compliant event-based metrics.

    Applies the full SzCORE preprocessing pipeline before any-overlap matching:
      1. Convert raw binary arrays to event lists.
      2. Widen reference events by pre/post-ictal tolerances.
      3. Merge reference and predicted events closer than *merge_gap_sec*.
      4. Split events longer than *max_event_sec*.
      5. Any-overlap matching between reference and predicted events.

    Returns the same dict keys as ``compute_event_metrics`` plus ``FAR/day``.
    """
    total_len = len(y_true)

    pre_samples = int(round(pre_ictal_sec / stride_sec))
    post_samples = int(round(post_ictal_sec / stride_sec))
    gap_samples = int(round(merge_gap_sec / stride_sec))
    max_samples = int(round(max_event_sec / stride_sec))

    ref_events = get_events(y_true)
    hyp_events = get_events(y_pred)

    ref_events = _apply_tolerances(ref_events, pre_samples, post_samples, total_len)
    ref_events = _merge_close_events(ref_events, gap_samples)
    ref_events = _split_long_events(ref_events, max_samples)

    hyp_events = _merge_close_events(hyp_events, gap_samples)
    hyp_events = _split_long_events(hyp_events, max_samples)

    tp_events = 0
    fn_events = 0
    fp_events = 0

    for rs, re_ in ref_events:
        detected = any(max(rs, ps) < min(re_, pe) for ps, pe in hyp_events)
        if detected:
            tp_events += 1
        else:
            fn_events += 1

    for ps, pe in hyp_events:
        matched = any(max(ps, rs) < min(pe, re_) for rs, re_ in ref_events)
        if not matched:
            fp_events += 1

    sensitivity = tp_events / len(ref_events) if ref_events else 0.0
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    total_hours = total_len * stride_sec / 3600.0
    far_per_hour = fp_events / total_hours if total_hours > 0 else 0.0
    far_per_day = far_per_hour * 24.0

    return {
        "Total Seizures (ref)": len(ref_events),
        "TP": tp_events,
        "FN": fn_events,
        "FP": fp_events,
        "Sensitivity": sensitivity,
        "Precision": precision,
        "F1": f1,
        "FAR/hr": far_per_hour,
        "FAR/day": far_per_day,
    }

CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]
TUSZ_CH_NAMES = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]

def get_ch_names_for_dataset(dataset):
    if dataset == 'TUSZ':
        return TUSZ_CH_NAMES
    return CHBMIT_CH_NAMES


def load_model(args, device=None):
    """Load a baseline or adversarial model from a checkpoint.

    When ``--adversarial`` is set the backbone is wrapped in
    :class:`AdversarialNeuralTransformer` and ``num_patients`` is
    auto-detected from the checkpoint so the caller never needs to
    specify it.
    """
    if device is None:
        device = torch.device(getattr(args, 'device', 'cuda'))

    backbone = create_model(
        args.model, pretrained=False, num_classes=1,
        drop_rate=0.0, drop_path_rate=0.1, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True,
        init_values=0.1,
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    clean_state = {k.replace('module.', ''): v for k, v in model_state.items()}

    if getattr(args, 'adversarial', False):
        disc_key = [k for k in clean_state if k.startswith('patient_discriminator') and k.endswith('.weight')][-1]
        num_patients = clean_state[disc_key].shape[0]
        adv_hidden = getattr(args, 'adv_hidden_dim', 512)
        model = AdversarialNeuralTransformer(
            backbone, num_patients=num_patients, adv_hidden_dim=adv_hidden,
        )

        if 'seizure_head.0.weight' in clean_state:
            model.seizure_head = torch.nn.Sequential(
                torch.nn.Linear(backbone.embed_dim, backbone.embed_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(backbone.embed_dim, backbone.num_classes),
            )
            print("Detected 2-layer MLP seizure head checkpoint")
        else:
            print("Detected single-layer seizure head checkpoint")

        model.load_state_dict(clean_state, strict=False)
        print(f"Loaded adversarial model ({num_patients} patients)")
    else:
        backbone.load_state_dict(clean_state, strict=False)
        model = backbone

    model.to(device).eval()
    return model


def load_scorenet(ckpt_path, device):
    """Load a trained ScoreNet checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})
    net = ScoreNet(
        w=saved_args.get('w', 6),
        gamma=saved_args.get('gamma', 0.5),
    )
    net.load_state_dict(ckpt['model_state_dict'])
    net.to(device).eval()
    val_f1 = ckpt.get('val_f1', '?')
    f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
    print(f"Loaded ScoreNet from {ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')}, val_F1={f1_str})")
    return net


@torch.no_grad()
def scorenet_postprocess(y_prob, scorenet_model, device, threshold=0.5, min_dur_sec=10):
    """Run ScoreNet on a flat probability array and apply hard constraints."""
    w = scorenet_model.w
    Z = build_toeplitz(y_prob.astype(np.float32), w)
    Z_t = torch.from_numpy(Z).to(device)
    refined = scorenet_model(Z_t, [len(y_prob)]).cpu().numpy()
    preds = (refined >= threshold).astype(int)
    preds = hard_constraints(preds, min_dur_sec=min_dur_sec)
    return preds, refined


@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device)
    
    print(f"Loading Model: {args.model}")
    model = load_model(args, device)

    ch_names_raw = get_ch_names_for_dataset(args.dataset)
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

    scorenet_ckpt = getattr(args, 'scorenet_checkpoint', None)
    if scorenet_ckpt:
        print(f"\n--- ScoreNet Post-Processing ---")
        sn_model = load_scorenet(scorenet_ckpt, device)
        threshold = getattr(args, 'sn_threshold', 0.5)
        min_dur = getattr(args, 'sn_min_dur', 10)
        print(f"Params: threshold={threshold}, min_dur_sec={min_dur}")
        y_pred_pp, y_refined = scorenet_postprocess(
            y_prob, sn_model, device,
            threshold=threshold, min_dur_sec=min_dur)
    else:
        print(f"\n--- Hand-Tuned Post-Processing ---")
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

    print("\n--- Event-Based Metrics (strict, no tolerances) ---")
    pp_evt = compute_event_metrics(y_true, y_pred_pp)
    
    print(f"Total Seizures (GT): {pp_evt['Total Seizures']}")
    print(f"Detected (TP):       {pp_evt['TP']} \t({pp_evt['Recall']*100:.1f}%)")
    print(f"Missed (FN):         {pp_evt['FN']}")
    print(f"False Alarms (FP):   {pp_evt['FP']}")
    print(f"False Alarms/Hr:     {pp_evt['FAR/hr']:.4f}")
    print(f"Event Precision:     {pp_evt['Precision']:.4f}")
    print(f"Event F1:            {pp_evt['F1']:.4f}")

    print(f"\n--- SzCORE Event-Based Metrics ---")
    print(f"Params: pre_ictal={args.szcore_pre_ictal}s, "
          f"post_ictal={args.szcore_post_ictal}s, "
          f"merge_gap={args.szcore_merge_gap}s, "
          f"max_event={args.szcore_max_event}s")
    szcore_evt = compute_szcore_event_metrics(
        y_true, y_pred_pp,
        stride_sec=1.0,
        pre_ictal_sec=args.szcore_pre_ictal,
        post_ictal_sec=args.szcore_post_ictal,
        merge_gap_sec=args.szcore_merge_gap,
        max_event_sec=args.szcore_max_event,
    )
    print(f"Total Events (ref):  {szcore_evt['Total Seizures (ref)']}")
    print(f"Detected (TP):       {szcore_evt['TP']} \t({szcore_evt['Sensitivity']*100:.1f}%)")
    print(f"Missed (FN):         {szcore_evt['FN']}")
    print(f"False Alarms (FP):   {szcore_evt['FP']}")
    print(f"Sensitivity:         {szcore_evt['Sensitivity']:.4f}")
    print(f"Precision:           {szcore_evt['Precision']:.4f}")
    print(f"F1:                  {szcore_evt['F1']:.4f}")
    print(f"FAR/hr:              {szcore_evt['FAR/hr']:.4f}")
    print(f"FAR/day:             {szcore_evt['FAR/day']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-best.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str,
                        help='Dataset for channel names: CHBMIT | TUSZ')
    parser.add_argument('--adversarial', action='store_true',
                        help='Load an adversarial (GRL+Attention) checkpoint')
    
    # Hand-tuned post-processing parameters
    parser.add_argument('--t_high', default=0.40, type=float, help='High threshold for seizure trigger')
    parser.add_argument('--t_low', default=0.20, type=float, help='Low threshold for seizure continuation')
    parser.add_argument('--smooth', default=5, type=int, help='Smoothing window size (seconds)')
    parser.add_argument('--min_dur', default=5, type=int, help='Minimum seizure duration (seconds)')

    # ScoreNet post-processing (overrides hand-tuned if provided)
    parser.add_argument('--scorenet_checkpoint', default=None, type=str,
                        help='Path to trained ScoreNet .pth; uses learned postprocessing')
    parser.add_argument('--sn_threshold', default=0.5, type=float,
                        help='Threshold for ScoreNet refined probabilities')
    parser.add_argument('--sn_min_dur', default=10, type=int,
                        help='Min event duration in seconds (ACNS: >=10s)')

    # SzCORE event-based scoring parameters
    parser.add_argument('--szcore_pre_ictal', default=30.0, type=float,
                        help='Pre-ictal tolerance in seconds (default: 30)')
    parser.add_argument('--szcore_post_ictal', default=60.0, type=float,
                        help='Post-ictal tolerance in seconds (default: 60)')
    parser.add_argument('--szcore_merge_gap', default=90.0, type=float,
                        help='Merge events closer than this gap in seconds (default: 90)')
    parser.add_argument('--szcore_max_event', default=300.0, type=float,
                        help='Split events longer than this in seconds (default: 300)')

    args = parser.parse_args()
    run_eval(args)