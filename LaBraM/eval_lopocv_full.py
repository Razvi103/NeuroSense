import argparse
import glob
import json
import math
import os
import re
import time
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, average_precision_score,
)
from timm.models import create_model
from torch.utils.data import DataLoader

import modeling_finetune
from modeling_finetune import AdversarialNeuralTransformer
from dataset_maker.dataset_chbmit import MultiPatientAdversarialDataset
import utils
from scorenet import ScoreNet, build_toeplitz, hard_constraints
from evaluate_checkpoint import (
    post_process_probs,
    compute_szcore_event_metrics,
)


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6'
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lopocv_dir', type=str, required=True)
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--folds', default='', type=str)

    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--drop_path', default=0.2, type=float)
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--intermediate_layers', default='', type=str)

    # post-processing
    p.add_argument('--smooth', default=5, type=int)
    p.add_argument('--t_high', default=0.5, type=float)
    p.add_argument('--t_low', default=0.3, type=float)
    p.add_argument('--min_dur', default=10, type=int)

    p.add_argument('--sn_threshold', default=0.5, type=float)
    p.add_argument('--sn_min_dur', default=10, type=int)

    # szcore parameters
    p.add_argument('--pre_ictal', default=30.0, type=float)
    p.add_argument('--post_ictal', default=60.0, type=float)
    p.add_argument('--merge_gap', default=90.0, type=float)
    p.add_argument('--max_event', default=300.0, type=float)

    p.add_argument('--batch_size', default=2048, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='', type=str)

    return p.parse_args()


def load_fold_model(ckpt_path, args, device):
    backbone = create_model(
        args.model, pretrained=False, num_classes=1,
        drop_rate=0.0, drop_path_rate=args.drop_path, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True,
        init_values=0.1,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    clean = {k.replace('module.', ''): v for k, v in state.items()}

    disc_keys = [k for k in clean
                 if k.startswith('patient_discriminator') and k.endswith('.weight')]

    if disc_keys:
        num_patients = clean[disc_keys[-1]].shape[0]
        il = args.intermediate_layers
        intermediate = tuple(int(x) for x in il.split(',') if x.strip()) if il else ()

        model = AdversarialNeuralTransformer(
            backbone, num_patients=num_patients,
            adv_hidden_dim=args.adv_hidden_dim,
            intermediate_layers=intermediate,
        )

        if 'seizure_head.0.weight' in clean:
            model.seizure_head = torch.nn.Sequential(
                torch.nn.Linear(backbone.embed_dim, backbone.embed_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(backbone.embed_dim, backbone.num_classes),
            )

        model.load_state_dict(clean, strict=False)
    else:
        backbone.load_state_dict(clean, strict=False)
        model = backbone

    return model.to(device).eval()


@torch.no_grad()
def extract_test_probs(model, h5_path, device, batch_size, num_workers):
    input_chans = utils.get_input_chans(CHBMIT_CH_NAMES)
    ds = MultiPatientAdversarialDataset([h5_path])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_probs, all_labels = [], []
    for batch in loader:
        samples = batch[0].float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        with torch.amp.autocast('cuda'):
            logits = model(samples, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch[1].numpy())
    ds.close()
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_pointwise(y_true, y_pred, y_prob):
    r = {}
    if len(np.unique(y_true)) > 1:
        r['auc'] = float(roc_auc_score(y_true, y_prob))
        r['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        r['auc'] = r['auprc'] = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    r['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    return r


def evaluate_strategy(y_true, y_pred, y_prob, args):
    pw = compute_pointwise(y_true, y_pred, y_prob)

    szcore = compute_szcore_event_metrics(
        y_true, y_pred,
        pre_ictal_sec=args.pre_ictal,
        post_ictal_sec=args.post_ictal,
        merge_gap_sec=args.merge_gap,
        max_event_sec=args.max_event,
    )
    szcore_out = {
        'szcore_evt_f1': szcore['F1'],
        'szcore_evt_recall': szcore['Sensitivity'],
        'szcore_evt_precision': szcore['Precision'],
        'szcore_far_hr': szcore['FAR/hr'],
        'szcore_far_day': szcore['FAR/day'],
        'szcore_tp': szcore['TP'],
        'szcore_fn': szcore['FN'],
        'szcore_fp': szcore['FP'],
        'n_seizures': szcore['Total Seizures (ref)'],
    }

    return {**pw, **szcore_out}


def main():
    args = get_args()
    cudnn.benchmark = True
    device = torch.device(args.device)

    patient_h5s = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in sorted(glob.glob(os.path.join(args.data_dir, 'chb*.h5')))
    }

    folds = []
    for entry in sorted(os.listdir(args.lopocv_dir)):
        m = re.match(r'fold_(\d+)_(chb\d+)', entry)
        if not m:
            continue
        fd = os.path.join(args.lopocv_dir, entry)
        ckpt = os.path.join(fd, 'checkpoint-best.pth')
        if os.path.isfile(ckpt):
            folds.append((int(m.group(1)), m.group(2), fd, ckpt))

    if args.folds:
        sel = set(int(x) for x in args.folds.split(','))
        folds = [f for f in folds if f[0] in sel]

    strategies = ['raw', 'rule_based', 'scorenet']
    results_by_strategy = {s: [] for s in strategies}

    start = time.time()

    for fold_idx, test_pid, fold_dir, ckpt_path in folds:

        cache_path = os.path.join(fold_dir, 'test_probs.npz')
        if os.path.isfile(cache_path):
            cached = np.load(cache_path)
            probs, labels = cached['probs'], cached['labels']
        else:
            labram = load_fold_model(ckpt_path, args, device)
            probs, labels = extract_test_probs(
                labram, patient_h5s[test_pid], device,
                args.batch_size, args.num_workers)
            del labram
            torch.cuda.empty_cache()
            np.savez_compressed(cache_path, probs=probs, labels=labels)

        y_true = labels.astype(int)

        y_pred_raw = (probs >= 0.5).astype(int)
        raw_metrics = evaluate_strategy(y_true, y_pred_raw, probs, args)
        raw_metrics['fold'] = fold_idx
        raw_metrics['patient'] = test_pid
        results_by_strategy['raw'].append(raw_metrics)
        print(f"raw F1={raw_metrics['f1']}  "
              f"event F1={raw_metrics['szcore_evt_f1']}  "
              f"event precision={raw_metrics['szcore_evt_precision']}"
              )

        y_pred_ht = post_process_probs(
            probs, t_high=args.t_high, t_low=args.t_low,
            smooth_window=args.smooth, min_duration=args.min_dur,
        )
        ht_metrics = evaluate_strategy(y_true, y_pred_ht, probs, args)
        ht_metrics['fold'] = fold_idx
        ht_metrics['patient'] = test_pid
        results_by_strategy['rule_based'].append(ht_metrics)
        print(f"rule_based: F1={ht_metrics['f1']}  "
              f"ScEvF1={ht_metrics['szcore_evt_f1']}  "
              f"ScEvPrc={ht_metrics['szcore_evt_precision']}"
              )

        sn_path = os.path.join(fold_dir, 'scorenet_best.pth')
        if os.path.isfile(sn_path):
            sn_ckpt = torch.load(sn_path, map_location='cpu', weights_only=False)
            sn_model = ScoreNet(
                w=sn_ckpt.get('w', 6), gamma=sn_ckpt.get('gamma', 0.5),
            ).to(device).eval()
            sn_model.load_state_dict(sn_ckpt['model_state_dict'])

            w = sn_model.w
            Z = build_toeplitz(probs.astype(np.float32), w)
            Z_t = torch.from_numpy(Z).to(device)
            with torch.no_grad():
                refined = sn_model(Z_t, [len(probs)]).cpu().numpy()
            y_pred_sn = (refined >= args.sn_threshold).astype(int)
            y_pred_sn = hard_constraints(y_pred_sn, min_dur_sec=args.sn_min_dur)

            sn_metrics = evaluate_strategy(y_true, y_pred_sn, probs, args)
            sn_metrics['fold'] = fold_idx
            sn_metrics['patient'] = test_pid
            results_by_strategy['scorenet'].append(sn_metrics)
            print(f"scorenet:   F1={sn_metrics['f1']}  "
                  f"ScEvF1={sn_metrics['szcore_evt_f1']}  "
                  f"ScEvPrc={sn_metrics['szcore_evt_precision']}"
                  )

            del sn_model
            torch.cuda.empty_cache()
        else:
            print(f"scorenet:   no checkpoint found, skipping")


if __name__ == '__main__':
    main()
