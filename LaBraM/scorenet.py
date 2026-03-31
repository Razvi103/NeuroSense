"""
ScoreNet: a lightweight postprocessor that refines per-window seizure
probabilities produced by a frozen upstream detector.

Faithful reimplementation of:
    Boonyakitanont et al., "ScoreNet: A Neural Network-Based Post-Processing
    Model for Identifying Epileptic Seizure Onset and Offset in EEGs",
    IEEE TNSRE 2021.

Architecture (Equations 1-4 of the paper):
    candidate  c_i = sigmoid(a1^T z_i + b1)        -- 1D conv, size 2w+1
    score      s_i = tanh(a2^T z_i + b2)            -- 1D conv, size 2w+1
    output gate o_l = sigmoid(a3/N_l sum(s_j) + b3)  -- per group
    final      yhat_i = sigmoid(a4 * c_i * o_l + b4) -- per epoch

Total learnable parameters: 2*(2w+1) + 6 = 32 when w=6.

The forward pass is fully vectorised using scatter/segment ops — no Python
loops over time steps or groups.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """ScoreNet onset-offset detector (32 parameters for w=6)."""

    def __init__(self, w=6, gamma=0.5):
        super().__init__()
        self.w = w
        self.gamma = gamma
        filter_len = 2 * w + 1

        self.a1 = nn.Parameter(torch.ones(filter_len))
        self.b1 = nn.Parameter(torch.tensor(-6.0))
        self.a2 = nn.Parameter(torch.ones(filter_len))
        self.b2 = nn.Parameter(torch.tensor(-2.0))
        self.a3 = nn.Parameter(torch.tensor(3.0))
        self.b3 = nn.Parameter(torch.tensor(0.0))
        self.a4 = nn.Parameter(torch.tensor(3.0))
        self.b4 = nn.Parameter(torch.tensor(-1.0))

    def forward(self, Z, n_samples):
        """
        Parameters
        ----------
        Z : (filter_len, total_N) Toeplitz input matrix (all records concat)
        n_samples : list[int]  lengths of each record within Z

        Returns
        -------
        yhat : (total_N,)  refined seizure probabilities
        """
        total_N = Z.shape[1]
        device = Z.device

        candidate = torch.sigmoid(self.a1 @ Z + self.b1)   # (total_N,)
        score = torch.tanh(self.a2 @ Z + self.b2)           # (total_N,)

        binary = (candidate >= self.gamma).long()

        # Mark where a new group starts: value change OR record boundary
        value_change = torch.zeros(total_N, dtype=torch.bool, device=device)
        value_change[0] = True
        value_change[1:] = binary[1:] != binary[:-1]

        rec_starts = torch.zeros(total_N, dtype=torch.bool, device=device)
        offsets = torch.tensor(
            np.cumsum([0] + list(n_samples[:-1])),
            dtype=torch.long, device=device,
        )
        rec_starts[offsets] = True

        new_group = value_change | rec_starts
        group_ids = new_group.long().cumsum(0) - 1          # 0-based IDs
        n_groups = group_ids[-1].item() + 1

        # Per-group mean score via scatter
        ones = torch.ones(total_N, device=device)
        g_sum = torch.zeros(n_groups, device=device).scatter_add_(0, group_ids, score)
        g_cnt = torch.zeros(n_groups, device=device).scatter_add_(0, group_ids, ones)
        g_mean = g_sum / g_cnt

        # Output gate expanded to per-element
        o_l = torch.sigmoid(self.a3 * g_mean[group_ids] + self.b3)

        yhat = torch.sigmoid(self.a4 * candidate * o_l + self.b4)
        return yhat


# ---------------------------------------------------------------------------
#  Toeplitz matrix construction  (vectorised — no Python loop)
# ---------------------------------------------------------------------------

def build_toeplitz(z, w):
    """Build the (2w+1, N) Toeplitz input matrix from a 1-D prob array.

    Column i contains (z_{i-w}, ..., z_i, ..., z_{i+w}) with zero-padding
    at boundaries, matching the MATLAB reference implementation.
    """
    n = len(z)
    filt_len = 2 * w + 1
    padded = np.concatenate([np.zeros(w, dtype=z.dtype), z, np.zeros(w, dtype=z.dtype)])
    idx = np.arange(n)[None, :] + np.arange(filt_len)[:, None]   # (filt_len, n)
    return padded[idx]


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------

def log_dice_loss(yhat, y, eps=1e-7):
    """Log-dice loss from Eq. 7 of the ScoreNet paper.

    Uses log-transformed proxies for TP, FP, FN and ignores TN,
    naturally handling class imbalance.
    """
    yhat = yhat.clamp(eps, 1.0 - eps)
    log_1_minus_yhat = torch.log(1.0 - yhat)
    log_yhat = torch.log(yhat)

    intersec = (y * log_1_minus_yhat).sum()
    union = (2.0 * y * log_1_minus_yhat
             + (1.0 - y) * log_1_minus_yhat
             + y * log_yhat).sum()

    loss = 1.0 - 2.0 * intersec / (union + eps)
    return loss


def dice_loss(yhat, y, eps=1e-7):
    """Standard soft Dice loss."""
    yhat = yhat.clamp(eps, 1.0 - eps)
    intersection = (y * yhat).sum()
    return 1.0 - 2.0 * intersection / (y.sum() + yhat.sum() + eps)


def weighted_bce_loss(yhat, y, pos_weight=1.0, eps=1e-7):
    """Binary cross-entropy with per-class weighting."""
    yhat = yhat.clamp(eps, 1.0 - eps)
    w = torch.where(y == 1, pos_weight, 1.0)
    return -(w * (y * torch.log(yhat) + (1.0 - y) * torch.log(1.0 - yhat))).mean()


def combined_loss(yhat, y, pos_weight=17.0, alpha=0.3, eps=1e-7):
    """Dice + weighted BCE.  BCE provides strong per-window gradients even when
    the model is only moderately discriminative; Dice handles class imbalance
    once discrimination improves.

    Parameters
    ----------
    yhat : Tensor, values in (0, 1)
    y    : Tensor, binary labels {0, 1}
    pos_weight : float  — upweight for the seizure (minority) class in BCE
    alpha : float — weight of the Dice component (1-alpha for BCE)
    """
    return alpha * dice_loss(yhat, y, eps) + (1.0 - alpha) * weighted_bce_loss(yhat, y, pos_weight, eps)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class ProbSequenceDataset(Dataset):
    """Per-patient probability sequences with pre-built Toeplitz matrices.

    Each item is one patient's full recording (or a chunk of it),
    stored as a Toeplitz matrix ready for ScoreNet's forward pass.
    """

    def __init__(self, npz_path, w=6, max_len=None):
        data = np.load(npz_path)
        probs = data['probs'].astype(np.float32)
        labels = data['labels'].astype(np.float32)
        pids = data['patient_ids'].astype(np.int64)

        self.items = []

        for pid in np.unique(pids):
            mask = pids == pid
            p = probs[mask]
            lab = labels[mask]

            if max_len is not None and len(p) > max_len:
                for start in range(0, len(p), max_len):
                    end = min(start + max_len, len(p))
                    Z = build_toeplitz(p[start:end], w)
                    self.items.append((Z, lab[start:end], end - start))
            else:
                Z = build_toeplitz(p, w)
                self.items.append((Z, lab, len(p)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        Z, lab, n = self.items[idx]
        return (
            torch.from_numpy(Z),           # (filter_len, N)
            torch.from_numpy(lab),          # (N,)
            n,
        )


def collate_fn(batch):
    """Concatenate variable-length Toeplitz matrices along the time axis."""
    Zs, labels, ns = zip(*batch)
    Z_cat = torch.cat(Zs, dim=1)                # (filter_len, sum(N))
    labels_cat = torch.cat(labels, dim=0)        # (sum(N),)
    n_samples = list(ns)
    return Z_cat, labels_cat, n_samples


# ---------------------------------------------------------------------------
#  Hard constraints (applied after ScoreNet)
# ---------------------------------------------------------------------------

def hard_constraints(preds, min_dur_sec=10):
    """Apply minimum-duration filtering to a binary prediction array.

    Any detected event shorter than *min_dur_sec* windows (= seconds at
    1 s stride) is removed.  The default of 10 seconds follows the ACNS
    2021 Standardized Critical Care EEG Terminology (Hirsch et al., 2021)
    which defines an electrographic seizure as epileptiform discharges
    averaging >2.5 Hz for >= 10 s, or any evolving pattern lasting >= 10 s.
    """
    filtered = preds.copy()
    padded = np.concatenate(([0], filtered, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) < min_dur_sec:
            filtered[s:e] = 0
    return filtered
