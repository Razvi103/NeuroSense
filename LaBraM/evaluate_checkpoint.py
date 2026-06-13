import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
from tqdm import tqdm
import utils
from timm.models import create_model
from modeling_finetune import AdversarialNeuralTransformer
from scorenet import ScoreNet, hard_constraints, build_toeplitz
from timescoring.annotations import Annotation
from timescoring.scoring import EventScoring

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

#rule based post-processing
def post_process_probs(probs, t_high, t_low, smooth_window, min_duration):
    if smooth_window > 1:
        probs_smooth = pd.Series(probs).rolling(window=smooth_window, center=True).mean().fillna(0).values
    else:
        probs_smooth = probs

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

def compute_szcore_event_metrics(
    y_true,
    y_pred,
    stride_sec=1.0,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
    merge_gap_sec=90.0,
    max_event_sec=300.0,
):

    fs = 1.0 / stride_sec
    params = EventScoring.Parameters(
        toleranceStart=pre_ictal_sec,
        toleranceEnd=post_ictal_sec,
        minOverlap=0,
        maxEventDuration=max_event_sec,
        minDurationBetweenEvents=merge_gap_sec,
    )
    ref = Annotation(np.asarray(y_true, dtype=bool), fs)
    hyp = Annotation(np.asarray(y_pred, dtype=bool), fs)
    scores = EventScoring(ref, hyp, params)

    return {
        "Total Seizures (ref)": scores.refTrue,
        "TP": scores.tp,
        "FN": scores.refTrue - scores.tp,
        "FP": scores.fp,
        "Sensitivity": scores.sensitivity,
        "Precision": scores.precision,
        "F1": scores.f1,
        "FAR/hr": scores.fpRate / 24.0,
        "FAR/day": scores.fpRate,
    }

def load_scorenet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})
    net = ScoreNet(
        w=saved_args.get('w', 6),
        gamma=saved_args.get('gamma', 0.5),
    ).to(device).eval()
    net.load_state_dict(ckpt['model_state_dict'])
    return net


@torch.no_grad()
def scorenet_postprocess(y_prob, scorenet_model, device, threshold=0.5, min_dur_sec=10):
    w = scorenet_model.w
    Z = build_toeplitz(y_prob.astype(np.float32), w)
    Z_t = torch.from_numpy(Z).to(device)
    refined = scorenet_model(Z_t, [len(y_prob)]).cpu().numpy()
    preds = (refined >= threshold).astype(int)
    preds = hard_constraints(preds, min_dur_sec=min_dur_sec)
    return preds, refined


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

        model.load_state_dict(clean_state, strict=False)
        print(f"Loaded adversarial model ({num_patients} patients)")
    else:
        backbone.load_state_dict(clean_state, strict=False)
        model = backbone

    model.to(device).eval()
    return model


@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device)
    
    model = load_model(args, device)

    ch_names_raw = get_ch_names_for_dataset(args.dataset)
    input_chans = utils.get_input_chans(ch_names_raw)

    test_dset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    all_probs = []
    all_targets = []
    
    for samples, targets in tqdm(loader, desc="Inference"):
        samples = samples.to(device)
        samples = samples / 100.0
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        with torch.amp.autocast('cuda'):
            output = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(output).squeeze()
        
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())
        
    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)

    y_pred_pp = post_process_probs(
        y_prob,
        t_high=args.t_high,
        t_low=args.t_low,
        smooth_window=args.smooth,
        min_duration=args.min_dur
    )
    
    print("point-wise metrics:")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_pp).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_pp, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"recall:): {sensitivity}")
    print(f"specificity:          {specificity}")
    print(f"precision:      {precision}")
    print(f"F1:             {f1}")
    print(f"AUPRC:                {pr_auc}")
    print(f"AUROC:                {roc_auc}")
    print(f"cm:     [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    print(f"Event-based metrics:")
    szcore_evt = compute_szcore_event_metrics(
        y_true, y_pred_pp,
        stride_sec=1.0,
        pre_ictal_sec=args.szcore_pre_ictal,
        post_ictal_sec=args.szcore_post_ictal,
        merge_gap_sec=args.szcore_merge_gap,
        max_event_sec=args.szcore_max_event,
    )
    print(f"Total Events (ref):  {szcore_evt['Total Seizures']}")
    print(f"Detected (TP):       {szcore_evt['TP']} \t({szcore_evt['Sensitivity']*100}%)")
    print(f"missed:         {szcore_evt['FN']}")
    print(f"false alarms:   {szcore_evt['FP']}")
    print(f"sensitivity:         {szcore_evt['Sensitivity']}")
    print(f"precision:           {szcore_evt['Precision']}")
    print(f"F1:                  {szcore_evt['F1']}")
    print(f"far/hr:              {szcore_evt['FAR/hr']}")
    print(f"far/day:             {szcore_evt['FAR/day']}")

    if args.scorenet_checkpoint:
        sn_model = load_scorenet(args.scorenet_checkpoint, device)
        y_pred_sn, y_refined = scorenet_postprocess(
            y_prob, sn_model, device,
            threshold=args.sn_threshold, min_dur_sec=args.sn_min_dur)

        print(f"\nScoreNet post-processing (threshold={args.sn_threshold}, min_dur={args.sn_min_dur}):")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_sn).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred_sn, zero_division=0)
        roc_auc_sn = roc_auc_score(y_true, y_refined)
        pr_auc_sn = average_precision_score(y_true, y_refined)

        print(f"recall:              {sensitivity}")
        print(f"specificity:         {specificity}")
        print(f"precision:           {precision}")
        print(f"F1:                  {f1}")
        print(f"AUPRC:               {pr_auc_sn}")
        print(f"AUROC:               {roc_auc_sn}")
        print(f"cm:     [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

        szcore_sn = compute_szcore_event_metrics(
            y_true, y_pred_sn,
            stride_sec=1.0,
            pre_ictal_sec=args.szcore_pre_ictal,
            post_ictal_sec=args.szcore_post_ictal,
            merge_gap_sec=args.szcore_merge_gap,
            max_event_sec=args.szcore_max_event,
        )
        print(f"Total Events (ref):  {szcore_sn['Total Seizures (ref)']}")
        print(f"Detected (TP):       {szcore_sn['TP']}")
        print(f"missed:              {szcore_sn['FN']}")
        print(f"false alarms:        {szcore_sn['FP']}")
        print(f"sensitivity:         {szcore_sn['Sensitivity']}")
        print(f"precision:           {szcore_sn['Precision']}")
        print(f"F1:                  {szcore_sn['F1']}")
        print(f"far/hr:              {szcore_sn['FAR/hr']}")
        print(f"far/day:             {szcore_sn['FAR/day']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-best.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='CHBMIT', type=str)
    parser.add_argument('--adversarial', action='store_true')
    
    # post processing params
    parser.add_argument('--t_high', default=0.40, type=float)
    parser.add_argument('--t_low', default=0.20, type=float)
    parser.add_argument('--smooth', default=5, type=int)
    parser.add_argument('--min_dur', default=5, type=int)

    # szcore params
    parser.add_argument('--szcore_pre_ictal', default=30.0, type=float)
    parser.add_argument('--szcore_post_ictal', default=60.0, type=float)
    parser.add_argument('--szcore_merge_gap', default=90.0, type=float)
    parser.add_argument('--szcore_max_event', default=300.0, type=float)

    parser.add_argument('--scorenet_checkpoint', default=None, type=str)
    parser.add_argument('--sn_threshold', default=0.5, type=float)
    parser.add_argument('--sn_min_dur', default=10, type=int)

    args = parser.parse_args()
    run_eval(args)