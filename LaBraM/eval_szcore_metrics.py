import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.models import create_model
import utils
from einops import rearrange
from evaluate_checkpoint import CHBMITDataset, post_process_probs # Use your existing file

def compute_epoch_metrics(y_true, y_pred, fs=1, epoch_sec=10):
    """Divides the timeline into fixed epoch windows and applies max pooling."""
    n_samples = len(y_true)
    samples_per_epoch = int(fs * epoch_sec)
    n_epochs = n_samples // samples_per_epoch
    
    y_true_trunc = y_true[:n_epochs * samples_per_epoch]
    y_pred_trunc = y_pred[:n_epochs * samples_per_epoch]
    
    y_true_epoch = y_true_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    y_pred_epoch = y_pred_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    
    return y_true_epoch, y_pred_epoch

@torch.no_grad()
def run_rigorous_eval(args):
    device = torch.device(args.device)
    
    print(f"Loading Model: {args.model}")
    model = create_model(args.model, pretrained=False, num_classes=1, 
                         drop_rate=0.0, drop_path_rate=0.1, use_mean_pooling=True,
                         qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    clean_state = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()

    ch_names = ['F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1', 'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2', 'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6']
    input_chans = utils.get_input_chans(ch_names)

    print("Loading Test Set...")
    dataset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8)

    all_probs = []
    all_targets = []
    
    for samples, targets in tqdm(loader):
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
    
    y_pred_pp = post_process_probs(y_prob, t_high=0.4, t_low=0.2, smooth_window=5, min_duration=5)
    
    print("\n--- Epoch-Based Evaluation ---")
    
    yt_epoch, yp_epoch = compute_epoch_metrics(y_true, y_pred_pp, epoch_sec=10)
    
    epoch_f1 = f1_score(yt_epoch, yp_epoch)
    kappa = cohen_kappa_score(yt_epoch, yp_epoch)
    tn, fp, fn, tp = confusion_matrix(yt_epoch, yp_epoch).ravel()
    
    print(f"Epoch (10s) F1 Score: {epoch_f1:.4f}")
    print(f"Cohen's Kappa:        {kappa:.4f}")
    print(f"Epoch Specificity:    {tn / (tn+fp):.4f}")
    print(f"Epoch Sensitivity:    {tp / (tp+fn):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-9.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    
    run_rigorous_eval(args)