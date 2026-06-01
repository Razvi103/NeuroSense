"""
Post-hoc patient-identifiability probe (fast refactor).

Measures how linearly decodable patient identity is from frozen backbone
features at multiple depths, comparing Baseline vs single-GRL vs multi-GRL.
The headline comparison is block_3 / block_7 between single-GRL and multi-GRL:
single-layer alignment should leave intermediate layers patient-discriminative,
multi-layer alignment should suppress them.

Key differences from the original script (all either speed-neutral or
methodology-positive):

  1. SINGLE FORWARD PASS, THREE TAPS. One pass to the last block taps
     block_3, block_7 and final via the existing block loop. The readout is
     identical to the original (backbone.norm -> mean-pool patch tokens ->
     fc_norm) at every depth, so features are numerically equivalent; we just
     stop recomputing the lower blocks three times and stop re-reading HDF5
     three times.

  2. INDEX SELECTION BEFORE EXTRACTION. Per patient we read only `labels`
     (never `data`), do a temporal 80/20 block split with a +/- gap buffer,
     keep background (non-seizure) windows for the patient probe, and subsample
     per patient per side to a fixed cap. Only the selected windows are ever
     forward-passed. Seizure windows are also kept and tagged (cheap, ~0.3% of
     data) so a seizure-decodability probe can be added later from cache with
     no re-extraction; it is NOT computed here.

  3. TEMPORAL BLOCK SPLIT (no recording IDs needed). Windows are stored in
     chronological order within each per-patient H5. An 80% cut falls inside a
     single recording, so all but one recording lands entirely on one side
     (approximate recording-level holdout); the gap buffer removes the overlap
     at that one seam. The probe must therefore recognise patient identity
     ACROSS sessions, a stricter test than a random split.

  4. The window selection depends only on labels, not weights, so it is
     computed once per fold and reused across all three variants -> the three
     variants are always scored on the same windows.

CHB-MIT LOPOCV usage:
    python eval_patient_probe_fast.py \\
        --lopocv \\
        --data_dir /path/to/CHBMIT_per_patient \\
        --dataset CHBMIT \\
        --num_workers 8 \\
        --run "Baseline:baseline:/path/to/finetune_chbmit_baseline_lopocv" \\
        --run "Single_GRL:adversarial:/path/to/finetune_chbmit_lopo_cv" \\
        --run "Multi_GRL:adversarial:/path/to/finetune_chbmit_multilayer_lopocv:3,7" \\
        --layers final,block_3,block_7 \\
        --cache_dir ./probe_cache \\
        --output patient_probe_lopocv_results.json

TUSZ (single H5, patient-disjoint splits) is supported as a secondary path;
it groups by patient_id and applies the same temporal split assuming each
patient's rows are stored in chronological order (verify for your TUSZ build).
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from timm.models import create_model
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import modeling_finetune  # noqa: F401 - registers timm models
from modeling_finetune import AdversarialNeuralTransformer
import utils

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


CHBMIT_CH_NAMES = [
    'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'C3', 'O1',
    'F4', 'C4', 'C4', 'O2', 'F8', 'T4', 'T6', 'O2',
    'CZ', 'PZ', 'T5', 'FT9', 'FT10', 'T4', 'T6',
]
TUSZ_CH_NAMES = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]

LAYER_DEFAULT = 'final,block_3,block_7'


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class H5SubsetDataset(Dataset):
    """Reads only the given (sorted) window indices from one H5 file.

    Opens the file lazily per worker so num_workers > 0 is fork-safe. Returns
    (data, position) where position is the row in `indices`, used to keep the
    extracted features aligned with the precomputed per-window metadata.
    """

    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self._h5 = None

    def _f(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        data = torch.from_numpy(self._f()['data'][idx]).float()
        data = torch.nan_to_num(data, nan=0.0, posinf=1e4, neginf=-1e4)
        return data, i

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


# ---------------------------------------------------------------------------
# Checkpoint loading (unchanged from the original)
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
# Single-pass multi-layer feature extraction
# ---------------------------------------------------------------------------

def _parse_layer(layer):
    if layer in ('final', 'final_mean'):
        return None
    if layer.startswith('block_'):
        return int(layer.split('_', 1)[1])
    raise ValueError(f"Unknown layer {layer!r}; use 'final' or 'block_N'")


@torch.no_grad()
def extract_all_layers_features(backbone, x, input_chans, layer_specs):
    """One forward pass; returns {layer_name: (B, D)} for all requested depths.

    Readout per depth = backbone.norm -> mean-pool patch tokens -> fc_norm,
    matching the original per-layer extraction exactly.
    """
    want_final = False
    block_taps = {}            # block_idx -> layer_name
    for name in layer_specs:
        bi = _parse_layer(name)
        if bi is None:
            want_final = True
        else:
            block_taps[bi] = name

    # ---- embedding prep (verbatim from the original partial forward) ----
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

    def readout(h):
        tokens = backbone.norm(h)[:, 1:, :]
        return backbone.fc_norm(tokens.mean(dim=1))

    out = {}
    last = len(backbone.blocks) - 1
    for i, blk in enumerate(backbone.blocks):
        x = blk(x, rel_pos_bias=None)
        if i in block_taps:
            out[block_taps[i]] = readout(x)
        if i == last and want_final:
            out['final'] = readout(x)
    return out


@torch.no_grad()
def extract_selected_features(model, h5_path, sel_idx, input_chans, device,
                              batch_size, num_workers, layer_specs, desc='extract'):
    backbone = model.backbone if isinstance(model, AdversarialNeuralTransformer) else model
    ds = H5SubsetDataset(h5_path, sel_idx)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == 'cuda',
    )
    feats = {name: [] for name in layer_specs}
    for data, _pos in tqdm(loader, desc=desc, leave=False):
        data = data.to(device) / 100.0
        data = rearrange(data, 'B N (A T) -> B N A T', T=200)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            out = extract_all_layers_features(backbone, data, input_chans, layer_specs)
        for name in layer_specs:
            feats[name].append(out[name].float().cpu().numpy())
    ds.close()
    return {name: (np.concatenate(v) if v else np.empty((0,))) for name, v in feats.items()}


# ---------------------------------------------------------------------------
# Temporal-block window selection (background subsample + kept seizures)
# ---------------------------------------------------------------------------

def build_unit_selection(labels, split_frac, gap, cap_train_bg, cap_eval_bg, rng):
    """Given one unit's chronologically-ordered labels, return selected window
    indices (sorted) plus aligned split ('train'/'eval') and seizure (0/1) tags.

    Background windows feed the patient probe (subsampled per side); seizure
    windows are kept in full and tagged for later use.
    """
    labels = np.asarray(labels)
    n = len(labels)
    split_idx = int(split_frac * n)
    train_end = max(0, split_idx - gap)
    eval_start = min(n, split_idx + gap)

    idx = np.arange(n)
    is_bg = labels == 0
    is_sz = labels != 0
    in_train = idx < train_end
    in_eval = idx >= eval_start

    train_bg = idx[in_train & is_bg]
    eval_bg = idx[in_eval & is_bg]
    train_sz = idx[in_train & is_sz]
    eval_sz = idx[in_eval & is_sz]

    if cap_train_bg > 0 and len(train_bg) > cap_train_bg:
        train_bg = rng.choice(train_bg, cap_train_bg, replace=False)
    if cap_eval_bg > 0 and len(eval_bg) > cap_eval_bg:
        eval_bg = rng.choice(eval_bg, cap_eval_bg, replace=False)

    parts = [train_bg, eval_bg, train_sz, eval_sz]
    split_parts = ['train', 'eval', 'train', 'eval']
    sz_parts = [0, 0, 1, 1]

    sel = np.concatenate(parts) if any(len(p) for p in parts) else np.array([], dtype=int)
    split = np.concatenate([[s] * len(p) for s, p in zip(split_parts, parts)]) \
        if len(sel) else np.array([], dtype='<U5')
    sz = np.concatenate([[v] * len(p) for v, p in zip(sz_parts, parts)]).astype(int) \
        if len(sel) else np.array([], dtype=int)

    order = np.argsort(sel)
    return sel[order], split[order], sz[order], len(train_bg), len(eval_bg)


# ---------------------------------------------------------------------------
# Patient probe (uses the precomputed temporal split; background only)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def _balanced_class_weights(y, n_classes, device):
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_patient_probe(feats, patient_id, seizure_label, split, device,
                      n_patients_total, probe_epochs=30, probe_lr=1e-2,
                      probe_batch_size=8192):
    """23-way patient probe on background windows, temporal train/eval split."""
    bg = seizure_label == 0
    tr = bg & (split == 'train')
    ev = bg & (split == 'eval')

    X_tr, y_tr_raw = feats[tr], patient_id[tr]
    X_ev, y_ev_raw = feats[ev], patient_id[ev]

    classes = np.unique(np.concatenate([y_tr_raw, y_ev_raw]))
    if len(classes) < 2:
        raise ValueError(f"Need >= 2 patients with background windows, got {len(classes)}")
    id_to_cls = {pid: i for i, pid in enumerate(classes)}
    y_tr = np.array([id_to_cls[p] for p in y_tr_raw], dtype=np.int64)
    y_ev = np.array([id_to_cls[p] for p in y_ev_raw], dtype=np.int64)
    k = len(classes)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_ev = scaler.transform(X_ev).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_ev_t = torch.from_numpy(X_ev).to(device)

    probe = LinearProbe(X_tr.shape[1], k).to(device)
    criterion = nn.CrossEntropyLoss(weight=_balanced_class_weights(y_tr, k, device))
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=probe_batch_size, shuffle=True)

    probe.train()
    for _ in range(probe_epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        y_pred = probe(X_ev_t).argmax(dim=1).cpu().numpy()

    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_ev, y_pred)),
        'macro_f1': float(f1_score(y_ev, y_pred, average='macro', zero_division=0)),
        'top1_accuracy': float((y_pred == y_ev).mean()),
        'chance_accuracy': float(1.0 / n_patients_total),
        'n_classes': int(k),
        'n_train_windows': int(len(y_tr)),
        'n_eval_windows': int(len(y_ev)),
    }


# ---------------------------------------------------------------------------
# LOPOCV driver
# ---------------------------------------------------------------------------

def discover_patient_h5s(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, 'chb*.h5')))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def discover_lopocv_folds(lopocv_dir, fold_filter=None):
    if not os.path.isdir(lopocv_dir):
        raise FileNotFoundError(f"LOPOCV directory not found: {lopocv_dir}")
    folds = []
    for entry in sorted(os.listdir(lopocv_dir)):
        m = re.match(r'fold_(\d+)_(chb\d+)', entry)
        if not m:
            continue
        ckpt = os.path.join(lopocv_dir, entry, 'checkpoint-best.pth')
        if os.path.isfile(ckpt):
            folds.append((int(m.group(1)), m.group(2), ckpt))
    if fold_filter is not None:
        folds = [f for f in folds if f[0] in fold_filter]
    return folds


def cache_is_valid(cache_path, ckpt_path):
    return (os.path.isfile(cache_path)
            and os.path.getmtime(cache_path) >= os.path.getmtime(ckpt_path))


def build_fold_selection(train_files, args, rng):
    """For each training-patient file, compute the shared window selection from
    labels only (no `data` read). Returns ordered list of per-file dicts and a
    global patient->class-id map.
    """
    selection = []
    pid_map = {pid: i for i, pid in enumerate(sorted(p for p, _ in train_files))}
    for pid, path in train_files:
        with h5py.File(path, 'r') as f:
            labels = f['labels'][:]
        sel, split, sz, n_tr, n_ev = build_unit_selection(
            labels, args.split_frac, args.gap_windows,
            args.cap_train_bg, args.cap_eval_bg, rng,
        )
        if n_tr == 0 or n_ev == 0:
            print(f"      warning: {pid} has background train={n_tr} eval={n_ev}; "
                  f"underused this fold")
        selection.append({
            'pid': pid, 'path': path, 'sel': sel, 'split': split,
            'seizure': sz, 'class_id': pid_map[pid],
        })
    return selection, pid_map


def get_or_extract(run, fold_idx, test_pid, ckpt, selection, layers, args, device, input_chans):
    safe = run.name.replace(' ', '_').replace('+', 'p')
    cache_path = ''
    if args.cache_dir:
        cache_path = os.path.join(args.cache_dir, safe, f"fold_{fold_idx:02d}_{test_pid}.npz")

    if cache_path and cache_is_valid(cache_path, ckpt):
        d = np.load(cache_path, allow_pickle=True)
        feats = {name: d[f"feat_{name}"] for name in layers}
        return feats, d['patient_id'], d['seizure_label'], d['split']

    model = load_checkpoint(ckpt, args.model, run.model_type == 'adversarial',
                            args.adv_hidden_dim, run.intermediate_layers, device)
    feat_acc = {name: [] for name in layers}
    pid_acc, sz_acc, split_acc = [], [], []
    for u in selection:
        if len(u['sel']) == 0:
            continue
        f = extract_selected_features(
            model, u['path'], u['sel'], input_chans, device,
            args.batch_size, args.num_workers, layers,
            desc=f"{run.name} f{fold_idx:02d} {u['pid']}",
        )
        for name in layers:
            feat_acc[name].append(f[name])
        pid_acc.append(np.full(len(u['sel']), u['class_id'], dtype=np.int64))
        sz_acc.append(u['seizure'])
        split_acc.append(u['split'])
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    feats = {name: np.concatenate(feat_acc[name]) for name in layers}
    patient_id = np.concatenate(pid_acc)
    seizure_label = np.concatenate(sz_acc)
    split = np.concatenate(split_acc)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path, patient_id=patient_id, seizure_label=seizure_label,
            split=split, **{f"feat_{name}": feats[name] for name in layers},
        )
    return feats, patient_id, seizure_label, split


def aggregate(per_fold, key='balanced_accuracy'):
    vals = [r[key] for r in per_fold if key in r and 'error' not in r]
    if not vals:
        return {}
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'median': float(np.median(vals)), 'min': float(np.min(vals)),
            'max': float(np.max(vals)), 'n': len(vals)}


def compute_paired_tests(results, layers):
    """Wilcoxon signed-rank between every pair of runs at each layer, on folds
    where both succeeded (paired by fold index)."""
    if not _HAVE_SCIPY:
        return {'note': 'scipy not available; run wilcoxon offline on per_fold values'}
    out = {}
    names = list(results.keys())
    for layer in layers:
        out[layer] = {}
        for a, b in itertools.combinations(names, 2):
            fa = {r['fold']: r['balanced_accuracy'] for r in results[a][layer]['per_fold'] if 'error' not in r}
            fb = {r['fold']: r['balanced_accuracy'] for r in results[b][layer]['per_fold'] if 'error' not in r}
            common = sorted(set(fa) & set(fb))
            if len(common) < 6:
                out[layer][f"{a}_vs_{b}"] = {'n_pairs': len(common), 'note': 'n<6: cannot reach p<0.05'}
                continue
            xa = [fa[f] for f in common]
            xb = [fb[f] for f in common]
            try:
                stat, p = wilcoxon(xa, xb)
                out[layer][f"{a}_vs_{b}"] = {
                    'n_pairs': len(common), 'statistic': float(stat), 'p_value': float(p),
                    'median_diff': float(np.median(np.array(xa) - np.array(xb))),
                }
            except Exception as exc:
                out[layer][f"{a}_vs_{b}"] = {'n_pairs': len(common), 'error': str(exc)}
    return out


def run_lopocv(args, layers, device, input_chans, patient_h5s):
    fold_filter = ({int(x) for x in args.folds.split(',') if x.strip()}
                   if args.folds else None)
    results = {run.name: {layer: {'per_fold': []} for layer in layers} for run in args.run}

    # All runs share the same fold list (one LOPOCV root per run; folds keyed by test patient).
    ref_folds = discover_lopocv_folds(args.run[0].checkpoint, fold_filter)
    print(f"{len(ref_folds)} folds")

    for fold_idx, test_pid, _ in ref_folds:
        if test_pid not in patient_h5s:
            print(f"fold {fold_idx:02d} {test_pid}: skipped (no H5)")
            continue
        print(f"\n=== fold {fold_idx:02d}  (held-out {test_pid}) ===")
        train_files = [(pid, p) for pid, p in sorted(patient_h5s.items()) if pid != test_pid]
        rng = np.random.default_rng(args.seed + fold_idx)  # shared across variants
        selection, _ = build_fold_selection(train_files, args, rng)

        for run in args.run:
            run_folds = {f[1]: f[2] for f in discover_lopocv_folds(run.checkpoint, fold_filter)}
            if test_pid not in run_folds:
                print(f"  [{run.name}] missing fold for {test_pid}; skipping")
                for layer in layers:
                    results[run.name][layer]['per_fold'].append(
                        {'fold': fold_idx, 'test_patient': test_pid, 'error': 'missing checkpoint'})
                continue
            ckpt = run_folds[test_pid]
            try:
                feats, pid, sz, split = get_or_extract(
                    run, fold_idx, test_pid, ckpt, selection, layers, args, device, input_chans)
                for layer in layers:
                    m = fit_patient_probe(feats[layer], pid, sz, split, device,
                                          n_patients_total=len(train_files),
                                          probe_epochs=args.probe_epochs,
                                          probe_lr=args.probe_lr,
                                          probe_batch_size=args.probe_batch_size)
                    m.update({'fold': fold_idx, 'test_patient': test_pid, 'layer': layer})
                    results[run.name][layer]['per_fold'].append(m)
                    print(f"  [{run.name}] {layer}: bal_acc={m['balanced_accuracy']:.4f} "
                          f"(chance={m['chance_accuracy']:.4f})")
            except Exception as exc:
                for layer in layers:
                    results[run.name][layer]['per_fold'].append(
                        {'fold': fold_idx, 'test_patient': test_pid, 'error': str(exc)})
                print(f"  [{run.name}] ERROR: {exc}")

    for run in args.run:
        for layer in layers:
            pf = results[run.name][layer]['per_fold']
            results[run.name][layer]['aggregate'] = aggregate(pf)
    return results


# ---------------------------------------------------------------------------
# Single-H5 driver (TUSZ; groups by patient_id, same temporal split per group)
# ---------------------------------------------------------------------------

def run_single_h5(args, layers, device, input_chans):
    h5_path = os.path.join(args.data_path, f'{args.split}.h5')
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels'][:]
        if 'patient_ids' not in f:
            raise ValueError("single-H5 mode needs patient_ids in the file")
        pids = f['patient_ids'][:]

    rng = np.random.default_rng(args.seed)
    selection, n_patients = [], len(np.unique(pids))
    for upid in np.unique(pids):
        gidx = np.where(pids == upid)[0]            # assumed chronological within patient
        sel, split, sz, n_tr, n_ev = build_unit_selection(
            labels[gidx], args.split_frac, args.gap_windows,
            args.cap_train_bg, args.cap_eval_bg, rng)
        if len(sel) == 0:
            continue
        selection.append({'pid': int(upid), 'path': h5_path, 'sel': gidx[sel],
                           'split': split, 'seizure': sz, 'class_id': int(upid)})

    results = {run.name: {layer: {} for layer in layers} for run in args.run}
    for run in args.run:
        feats, pid_acc, sz_acc, split_acc = ({name: [] for name in layers}, [], [], [])
        model = load_checkpoint(run.checkpoint, args.model, run.model_type == 'adversarial',
                                args.adv_hidden_dim, run.intermediate_layers, device)
        for u in selection:
            f = extract_selected_features(model, u['path'], u['sel'], input_chans, device,
                                          args.batch_size, args.num_workers, layers,
                                          desc=f"{run.name} pid{u['pid']}")
            for name in layers:
                feats[name].append(f[name])
            pid_acc.append(np.full(len(u['sel']), u['class_id'], dtype=np.int64))
            sz_acc.append(u['seizure'])
            split_acc.append(u['split'])
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        F = {name: np.concatenate(feats[name]) for name in layers}
        pid = np.concatenate(pid_acc)
        sz = np.concatenate(sz_acc)
        split = np.concatenate(split_acc)
        for layer in layers:
            try:
                results[run.name][layer] = fit_patient_probe(
                    F[layer], pid, sz, split, device, n_patients_total=n_patients,
                    probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
                    probe_batch_size=args.probe_batch_size)
                results[run.name][layer]['layer'] = layer
                print(f"[{run.name}] {layer}: bal_acc="
                      f"{results[run.name][layer]['balanced_accuracy']:.4f}")
            except Exception as exc:
                results[run.name][layer] = {'error': str(exc)}
    return results


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
        raise argparse.ArgumentTypeError(f"Expected NAME:TYPE:PATH[:INTERMEDIATE], got {spec!r}")
    if parts[1] not in ('baseline', 'adversarial'):
        raise argparse.ArgumentTypeError(f"TYPE must be baseline or adversarial, got {parts[1]!r}")
    return RunConfig(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else '')


def main():
    p = argparse.ArgumentParser(description='Fast patient-identifiability probe')
    p.add_argument('--data_path', default='')
    p.add_argument('--data_dir', default='')
    p.add_argument('--lopocv', action='store_true')
    p.add_argument('--folds', default='', type=str)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--dataset', default='CHBMIT', choices=['TUSZ', 'CHBMIT'])
    p.add_argument('--run', action='append', required=True, type=parse_run_spec)
    p.add_argument('--layers', default=LAYER_DEFAULT)
    p.add_argument('--output', default='patient_probe_results.json')
    p.add_argument('--cache_dir', default='')
    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--adv_hidden_dim', default=512, type=int)
    p.add_argument('--batch_size', default=2048, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--device', default='cuda')
    # temporal split + subsampling
    p.add_argument('--split_frac', default=0.8, type=float)
    p.add_argument('--gap_windows', default=30, type=int,
                   help='windows dropped each side of the split point (1 s stride => 30 = 30 s)')
    p.add_argument('--cap_train_bg', default=4000, type=int,
                   help='max background windows per patient on the train side (0 = no cap)')
    p.add_argument('--cap_eval_bg', default=1000, type=int)
    # probe
    p.add_argument('--probe_epochs', default=30, type=int)
    p.add_argument('--probe_lr', default=1e-2, type=float)
    p.add_argument('--probe_batch_size', default=8192, type=int)
    p.add_argument('--seed', default=42, type=int)
    args = p.parse_args()

    if args.lopocv and not args.data_dir:
        p.error('--data_dir is required with --lopocv')
    if not args.lopocv and not args.data_path:
        p.error('--data_path is required without --lopocv')

    layers = [s.strip() for s in args.layers.split(',') if s.strip()]
    device = torch.device(args.device)
    ch_names = TUSZ_CH_NAMES if args.dataset == 'TUSZ' else CHBMIT_CH_NAMES
    input_chans = utils.get_input_chans(ch_names)

    if args.lopocv:
        patient_h5s = discover_patient_h5s(args.data_dir)
        if not patient_h5s:
            raise FileNotFoundError(f"No chb*.h5 in {args.data_dir}")
        print(f"Discovered {len(patient_h5s)} patients")
        results = run_lopocv(args, layers, device, input_chans, patient_h5s)
        paired = compute_paired_tests(results, layers)
    else:
        results = run_single_h5(args, layers, device, input_chans)
        paired = {}

    settings = vars(args).copy()
    settings['run'] = [asdict(r) for r in args.run]
    out = {'settings': settings, 'results': results, 'paired_tests': paired}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()