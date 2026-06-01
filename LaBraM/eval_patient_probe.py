"""
Post-hoc patient-identifiability probe for frozen seizure-detector checkpoints.

Self-contained script: loads checkpoints, extracts internal features, trains a
fresh linear classifier on patient IDs, and writes JSON results.

Usage:
    python eval_patient_probe.py \\
        --data_path /path/to/TUSZ \\
        --dataset TUSZ \\
        --run "Baseline:baseline:/path/to/baseline/checkpoint-best.pth" \\
        --run "B+A:adversarial:/path/to/ba/checkpoint-best.pth" \\
        --run "B+A+GRL:adversarial:/path/to/grl/checkpoint-best.pth" \\
        --run "B+A+GRL+multi:adversarial:/path/to/multi/checkpoint-best.pth:3,7" \\
        --layers final,block_3,block_7 \\
        --output patient_probe_results.json

Each ``--run`` argument: ``NAME:TYPE:CHECKPOINT[:INTERMEDIATE_LAYERS]``
  TYPE = baseline | adversarial

TUSZ train / dev / test are **patient-disjoint** and use **per-split patient ID
mappings**.  The probe therefore runs entirely within one HDF5 split (default
``train.h5``): extract features, then 80/20 **window** split stratified by
patient so every patient in that split appears in both probe-train and
probe-eval.  Do not train on train and evaluate on val/test.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timm.models import create_model
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import modeling_finetune  # noqa: F401 — registers timm models
from modeling_finetune import AdversarialNeuralTransformer
import utils


CHBMIT_CH_NAMES = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8',
]
TUSZ_CH_NAMES = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5DatasetWithPIDs(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = len(self.h5_file['labels'])
        self.has_pids = 'patient_ids' in self.h5_file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.from_numpy(self.h5_file['data'][idx]).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=1e4, neginf=-1e4)
        label = int(self.h5_file['labels'][idx])
        pid = int(self.h5_file['patient_ids'][idx]) if self.has_pids else -1
        return data, label, pid

    def close(self):
        self.h5_file.close()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path, model_name, adversarial, adv_hidden_dim,
                    intermediate_layers, device):
    backbone = create_model(
        model_name, pretrained=False, num_classes=1,
        drop_rate=0.0, drop_path_rate=0.1, use_mean_pooling=True,
        qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True,
        init_values=0.1,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    clean = {k.replace('module.', ''): v for k, v in state.items()}

    if adversarial:
        disc_key = [k for k in clean
                    if k.startswith('patient_discriminator') and k.endswith('.weight')][-1]
        num_patients = clean[disc_key].shape[0]
        il = tuple(int(x) for x in intermediate_layers.split(',') if x.strip()) \
            if intermediate_layers else ()
        model = AdversarialNeuralTransformer(
            backbone, num_patients=num_patients,
            adv_hidden_dim=adv_hidden_dim, intermediate_layers=il,
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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _parse_layer(layer):
    if layer in ('final', 'final_mean'):
        return None
    if layer.startswith('block_'):
        return int(layer.split('_', 1)[1])
    raise ValueError(f"Unknown layer {layer!r}; use 'final' or 'block_N'")


def _forward_patch_tokens_to_block(backbone, x, input_chans, block_idx, token_fc_norm):
    """Partial backbone forward; returns patch tokens (B, n_patches, D)."""
    batch_size, n, a, t = x.shape
    input_time_window = a if t == backbone.patch_size else t
    x = backbone.patch_embed(x)

    cls_tokens = backbone.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    pos_embed_used = backbone.pos_embed[:, input_chans] if input_chans is not None else backbone.pos_embed
    if backbone.pos_embed is not None:
        pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
            batch_size, -1, input_time_window, -1,
        ).flatten(1, 2)
        pos_embed = torch.cat(
            (pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1,
        )
        x = x + pos_embed
    if backbone.time_embed is not None:
        nc = n if t == backbone.patch_size else a
        time_embed = backbone.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
            batch_size, nc, -1, -1,
        ).flatten(1, 2)
        x[:, 1:, :] += time_embed

    x = backbone.pos_drop(x)
    last = (len(backbone.blocks) - 1) if block_idx is None else block_idx
    for i, blk in enumerate(backbone.blocks):
        x = blk(x, rel_pos_bias=None)
        if i == last:
            break

    x = backbone.norm(x)
    patch_tokens = x[:, 1:, :]
    if token_fc_norm and backbone.fc_norm is not None:
        patch_tokens = backbone.fc_norm(patch_tokens)
    return patch_tokens


def extract_probe_features(model, x, input_chans, layer='final'):
    """Return (B, D) features for the patient probe.

    Adversarial: channel attention + fc_norm at every depth.
    Baseline: mean pool + fc_norm (final or partial block).
    """
    block_idx = _parse_layer(layer)
    _, n_channels, n_time, _ = x.shape

    if isinstance(model, AdversarialNeuralTransformer):
        patch_tokens = _forward_patch_tokens_to_block(
            model.backbone, x, input_chans, block_idx, token_fc_norm=True,
        )
        pooled, _ = model.channel_attention(patch_tokens, n_channels, n_time)
        return model.fc_norm(pooled)

    if block_idx is None:
        return model.forward_features(x, input_chans=input_chans)

    patch_tokens = _forward_patch_tokens_to_block(
        model, x, input_chans, block_idx, token_fc_norm=False,
    )
    return model.fc_norm(patch_tokens.mean(dim=1))


@torch.no_grad()
def extract_features_from_h5(model, h5_path, input_chans, device, batch_size, layer,
                             max_windows_per_patient=0, seed=42):
    dset = H5DatasetWithPIDs(h5_path)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_feats, all_pids = [], []
    for data, _labels, pids in tqdm(loader, desc=f"{os.path.basename(h5_path)} [{layer}]"):
        data = data.to(device) / 100.0
        data = rearrange(data, 'B N (A T) -> B N A T', T=200)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            feats = extract_probe_features(model, data, input_chans, layer=layer)
        all_feats.append(feats.float().cpu().numpy())
        all_pids.append(pids.numpy())

    dset.close()
    X = np.concatenate(all_feats)
    y = np.concatenate(all_pids)

    if max_windows_per_patient > 0:
        rng = np.random.default_rng(seed)
        keep = []
        for pid in np.unique(y):
            idx = np.where(y == pid)[0]
            if len(idx) > max_windows_per_patient:
                idx = rng.choice(idx, max_windows_per_patient, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        X, y = X[keep], y[keep]

    return X, y


# ---------------------------------------------------------------------------
# Probe train / eval (GPU)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _balanced_class_weights(y, n_classes, device):
    """Inverse-frequency weights matching sklearn class_weight='balanced'."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_probe(probe_type, in_dim, n_classes):
    if probe_type == 'logistic':
        return LinearProbe(in_dim, n_classes)
    if probe_type == 'mlp':
        return MLPProbe(in_dim, n_classes)
    raise ValueError(f"Unknown probe type: {probe_type}")


def train_and_eval_probe(
    X, y, device, probe_type='logistic', test_size=0.2, seed=42,
    probe_epochs=50, probe_lr=1e-2, probe_batch_size=8192,
):
    unique_ids = np.unique(y)
    n_patients = len(unique_ids)
    if n_patients < 2:
        raise ValueError(f"Need >= 2 patients, got {n_patients}")

    id_to_class = {pid: i for i, pid in enumerate(unique_ids)}
    y_cls = np.array([id_to_class[pid] for pid in y], dtype=np.int64)

    # Window-level split stratified by patient — every patient must appear in
    # probe-train, otherwise held-out patients have zero training examples and
    # accuracy is trivially 0.
    indices = np.arange(len(y_cls))
    train_idx, eval_idx = train_test_split(
        indices, test_size=test_size, stratify=y_cls, random_state=seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_eval = scaler.transform(X[eval_idx]).astype(np.float32)
    y_train = y_cls[train_idx]
    y_eval = y_cls[eval_idx]

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_eval_t = torch.from_numpy(X_eval).to(device)

    probe = _build_probe(probe_type, X_train.shape[1], n_patients).to(device)
    class_weights = _balanced_class_weights(y_train, n_patients, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=probe_batch_size,
        shuffle=True,
    )

    probe.train()
    for epoch in tqdm(range(probe_epochs), desc='probe train', leave=False):
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(X_eval_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    all_labels = np.arange(n_patients)
    return {
        'n_patients': int(n_patients),
        'n_train_windows': int(len(train_idx)),
        'n_eval_windows': int(len(eval_idx)),
        'n_train_patients': int(len(np.unique(y_cls[train_idx]))),
        'n_eval_patients': int(len(np.unique(y_cls[eval_idx]))),
        'chance_accuracy': float(1.0 / n_patients),
        'balanced_accuracy': float(balanced_accuracy_score(
            y_eval, y_pred, labels=all_labels)),
        'macro_f1': float(f1_score(
            y_eval, y_pred, average='macro', zero_division=0, labels=all_labels)),
        'top1_accuracy': float((y_pred == y_eval).mean()),
        'probe_epochs': int(probe_epochs),
        'probe_device': str(device),
        'eval_protocol': 'window-level 80/20 split stratified by patient, within one H5 split',
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    model_type: str
    checkpoint: str
    intermediate_layers: str = ''


def parse_run_spec(spec):
    parts = spec.split(':')
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            f"Expected NAME:TYPE:CHECKPOINT[:INTERMEDIATE], got {spec!r}"
        )
    name, model_type, checkpoint = parts[0], parts[1], parts[2]
    if model_type not in ('baseline', 'adversarial'):
        raise argparse.ArgumentTypeError(f"TYPE must be baseline or adversarial, got {model_type!r}")
    return RunConfig(name, model_type, checkpoint, parts[3] if len(parts) > 3 else '')


def main():
    p = argparse.ArgumentParser(description='Patient-identifiability probe')
    p.add_argument('--data_path', required=True)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                   help='Which HDF5 split to use (patients are disjoint across splits; '
                        'probe train+eval stay within this file only)')
    p.add_argument('--dataset', default='TUSZ', choices=['TUSZ', 'CHBMIT'])
    p.add_argument('--run', action='append', required=True, type=parse_run_spec)
    p.add_argument('--layers', default='final')
    p.add_argument('--output', default='patient_probe_results.json')
    p.add_argument('--cache_dir', default='')
    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--batch_size', default=2048, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--probe', default='logistic', choices=['logistic', 'mlp'])
    p.add_argument('--probe_epochs', default=50, type=int)
    p.add_argument('--probe_lr', default=1e-2, type=float)
    p.add_argument('--probe_batch_size', default=8192, type=int)
    p.add_argument('--probe_test_size', default=0.2, type=float)
    p.add_argument('--max_windows_per_patient', default=0, type=int)
    p.add_argument('--seed', default=42, type=int)
    args = p.parse_args()

    layers = [s.strip() for s in args.layers.split(',') if s.strip()]
    device = torch.device(args.device)
    ch_names = TUSZ_CH_NAMES if args.dataset == 'TUSZ' else CHBMIT_CH_NAMES
    input_chans = utils.get_input_chans(ch_names)
    train_h5 = os.path.join(args.data_path, f'{args.split}.h5')
    if not os.path.isfile(train_h5):
        raise FileNotFoundError(f"Split file not found: {train_h5}")

    results = {}
    for run in args.run:
        results[run.name] = {}
        print(f"\nLoading [{run.name}] ← {run.checkpoint}")
        model = load_checkpoint(
            run.checkpoint, args.model, run.model_type == 'adversarial',
            args.adv_hidden_dim, run.intermediate_layers, device,
        )

        for layer in layers:
            print(f"  layer={layer}")
            safe = run.name.replace(' ', '_').replace('+', 'p')
            cache = os.path.join(args.cache_dir, f"{safe}_{layer}.npz") if args.cache_dir else ''

            try:
                if cache and os.path.isfile(cache):
                    d = np.load(cache)
                    X, y = d['features'], d['patient_ids']
                else:
                    X, y = extract_features_from_h5(
                        model, train_h5, input_chans, device, args.batch_size, layer,
                        args.max_windows_per_patient, args.seed,
                    )
                    if cache:
                        os.makedirs(args.cache_dir, exist_ok=True)
                        np.savez_compressed(cache, features=X, patient_ids=y)

                if (y < 0).any():
                    raise ValueError(f"patient_ids missing in {args.split}.h5")

                metrics = train_and_eval_probe(
                    X, y, device, args.probe, args.probe_test_size, args.seed,
                    args.probe_epochs, args.probe_lr, args.probe_batch_size,
                )
                metrics.update({
                    'layer': layer,
                    'split': args.split,
                    'checkpoint': run.checkpoint,
                    'readout': 'channel_attention+fc_norm' if run.model_type == 'adversarial'
                               else 'mean_pool+fc_norm',
                })
                results[run.name][layer] = metrics
                print(f"    balanced_acc={metrics['balanced_accuracy']:.4f}  "
                      f"(chance={metrics['chance_accuracy']:.4f})")
            except Exception as exc:
                results[run.name][layer] = {'error': str(exc)}
                print(f"    ERROR: {exc}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out = {'settings': vars(args), 'results': results}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == '__main__':
    main()
