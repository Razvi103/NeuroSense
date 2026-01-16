import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from timm.models import create_model
import utils
import modeling_finetune
from evaluate_checkpoint import CHBMITDataset, post_process_probs # Re-use your existing classes

@torch.no_grad()
def visualize(args):
    device = torch.device(args.device)
    
    print(f"Loading Model & Checkpoint...")
    model = create_model(
        args.model, pretrained=False, num_classes=1, 
        drop_rate=0.0, drop_path_rate=0.1, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    clean_state = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()

    ch_names_raw = ['F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1', 'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2', 'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6']
    input_chans = utils.get_input_chans(ch_names_raw)

    print("Running Inference on Test Set...")
    dataset = CHBMITDataset(args.data_path + '/test.h5')
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=16)
    
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
    if args.smooth > 1:
        y_smooth = pd.Series(y_prob).rolling(window=args.smooth, center=True).mean().fillna(0).values
    else:
        y_smooth = y_prob
    
    y_pred_final = post_process_probs(y_prob, args.t_high, args.t_low, args.smooth, args.min_dur)
    diffs = np.diff(np.concatenate(([0], y_true, [0])))
    seizure_starts = np.where(diffs == 1)[0]
    
    print(f"\nFound {len(seizure_starts)} seizures in the test set.")
    
    if len(seizure_starts) == 0:
        print("No seizures found in ground truth. Skipping plots.")
        return

    print("Generating plots...")
    num_to_plot = min(5, len(seizure_starts))
    
    for i in range(num_to_plot):
        start_idx = seizure_starts[i]
        window_margin = 180
        plot_start = max(0, start_idx - window_margin)
        offset = 0
        while start_idx + offset < len(y_true) and y_true[start_idx + offset] == 1:
            offset += 1
        seizure_end = start_idx + offset
        plot_end = min(len(y_true), seizure_end + window_margin)
        
        timeline = np.arange(plot_start, plot_end)
        segment_true = y_true[plot_start:plot_end]
        segment_prob = y_prob[plot_start:plot_end]
        segment_smooth = y_smooth[plot_start:plot_end]
        segment_pred = y_pred_final[plot_start:plot_end]
        
        plt.figure(figsize=(14, 6))
        plt.fill_between(timeline, 0, 1, where=(segment_true==1), 
                         color='green', alpha=0.3, label='Ground Truth Seizure')
        plt.plot(timeline, segment_prob, color='blue', alpha=0.3, linewidth=1, label='Raw Model Prob')
        plt.plot(timeline, segment_smooth, color='darkorange', linewidth=2.5, label=f'Smoothed ({args.smooth}s)')
        plt.axhline(y=args.t_high, color='red', linestyle='--', alpha=0.7, label=f'Start Thresh ({args.t_high})')
        plt.axhline(y=args.t_low, color='darkred', linestyle=':', alpha=0.7, label=f'Stop Thresh ({args.t_low})')
        plt.plot(timeline, segment_pred * 1.05, color='red', linewidth=2, label='Final Alarm (Predicted)')
        
        plt.title(f'Seizure Event #{i+1} Detection', fontsize=14)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Seizure Probability', fontsize=12)
        plt.ylim(-0.05, 1.1)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        filename = f'seizure_viz_{i+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-9.pth', type=str) # Use your best ckpt
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Visualization Params (Match your best eval settings)
    parser.add_argument('--t_high', default=0.4, type=float)
    parser.add_argument('--t_low', default=0.2, type=float)
    parser.add_argument('--smooth', default=5, type=int)
    parser.add_argument('--min_dur', default=5, type=int)
    
    args = parser.parse_args()
    visualize(args)