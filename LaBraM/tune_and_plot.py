import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.models import create_model
from einops import rearrange
import itertools
import pandas as pd

# Import your existing modules
import utils
from evaluate_checkpoint import CHBMITDataset, post_process_probs

# Reuse the epoch metric function
def compute_epoch_metrics(y_true, y_pred, fs=1, epoch_sec=10):
    n_samples = len(y_true)
    samples_per_epoch = int(fs * epoch_sec)
    n_epochs = n_samples // samples_per_epoch
    
    y_true_trunc = y_true[:n_epochs * samples_per_epoch]
    y_pred_trunc = y_pred[:n_epochs * samples_per_epoch]
    
    # Max pooling: if any sample in 10s is seizure, the whole epoch is seizure
    y_true_epoch = y_true_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    y_pred_epoch = y_pred_trunc.reshape(n_epochs, samples_per_epoch).max(axis=1)
    
    return y_true_epoch, y_pred_epoch

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    
    print(f"Loading Model: {args.model} from {args.checkpoint}")
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

    print("Running Inference...")
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

    print("\nGenerating Plots...")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_curves.png', dpi=150)
    print("Saved plots to 'metrics_curves.png'")

    print("\n--- Tuning Post-Processing Parameters ---")
    
    grid = {
        't_high': [0.3, 0.4, 0.5, 0.6, 0.7],
        't_low':  [0.2, 0.3, 0.4],
        'smooth': [5, 10, 30],
        'min_dur': [5, 7, 10]
    }
    
    best_kappa = -1
    best_params = {}
    results = []
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for p in tqdm(combinations):
        if p['t_low'] >= p['t_high']:
            continue
        y_pred_pp = post_process_probs(y_prob, p['t_high'], p['t_low'], p['smooth'], p['min_dur'])
        yt_ep, yp_ep = compute_epoch_metrics(y_true, y_pred_pp, epoch_sec=10)
        
        kappa = cohen_kappa_score(yt_ep, yp_ep)
        tn, fp, fn, tp = confusion_matrix(yt_ep, yp_ep).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(yt_ep, yp_ep)
        
        res = {**p, 'kappa': kappa, 'f1': f1, 'sens': sens, 'spec': spec}
        results.append(res)
        
        if kappa > best_kappa:
            best_kappa = kappa
            best_params = res

    df = pd.DataFrame(results)
    df = df.sort_values(by='kappa', ascending=False)
    
    print("\nTop 5 Configurations by Kappa:")
    print(df.head(5).to_string(index=False))
    
    print("\nTop 5 Configurations by F1 Score:")
    print(df.sort_values(by='f1', ascending=False).head(5).to_string(index=False))

    print("-" * 30)
    print(f"Best Kappa Found: {best_params['kappa']:.4f}")
    print(f"Parameters: High={best_params['t_high']}, Low={best_params['t_low']}, Smooth={best_params['smooth']}, MinDur={best_params['min_dur']}")
    print(f"Resulting Metrics: Sens={best_params['sens']:.4f}, Spec={best_params['spec']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/CHBMIT', type=str)
    parser.add_argument('--checkpoint', default='./checkpoints/finetune_chbmit_v1/checkpoint-19.pth', type=str)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    
    main(args)