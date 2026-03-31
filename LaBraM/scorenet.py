"""
ScoreNet: a lightweight BiLSTM postprocessor that refines per-window seizure
probabilities produced by a frozen upstream detector.

References
----------
Ilyas et al., "ScoreNet: A Neural Network-Based Post-Processing Model for
Identifying Epileptic Seizure Onset and Offset in EEGs", IEEE TNSRE 2022.

The log-dice loss (a differentiable F1 proxy) is taken from the same paper.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """2-layer BiLSTM that maps a probability sequence to a refined one."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths=None):
        """
        Parameters
        ----------
        x : (B, T, 1)  padded probability sequences
        lengths : (B,)  original lengths before padding (optional)

        Returns
        -------
        out : (B, T, 1)  refined probabilities in [0, 1]
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        return torch.sigmoid(self.fc(lstm_out))


# ---------------------------------------------------------------------------
#  Loss
# ---------------------------------------------------------------------------

def log_dice_loss(pred, target, smooth=1.0):
    """Differentiable F1 proxy (log-dice).

    Both *pred* and *target* should have the same shape and contain
    values in [0, 1].  Padding positions must be masked out before
    calling this function.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return -torch.log(dice)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class ProbSequenceDataset(Dataset):
    """Splits a flat probability array into per-patient chunks.

    Each item is a contiguous block of predictions belonging to a single
    patient.  Long recordings are further split into non-overlapping
    sub-sequences of at most *max_len* windows so that batching stays
    memory-friendly.
    """

    def __init__(self, npz_path, max_len=4096):
        data = np.load(npz_path)
        probs = data['probs'].astype(np.float32)
        labels = data['labels'].astype(np.float32)
        pids = data['patient_ids'].astype(np.int64)

        self.sequences = []
        self.targets = []

        unique_pids = np.unique(pids)
        for pid in unique_pids:
            mask = pids == pid
            p = probs[mask]
            l = labels[mask]
            for start in range(0, len(p), max_len):
                end = min(start + max_len, len(p))
                self.sequences.append(p[start:end])
                self.targets.append(l[start:end])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]).unsqueeze(-1),  # (T, 1)
            torch.from_numpy(self.targets[idx]).unsqueeze(-1),    # (T, 1)
        )


def collate_fn(batch):
    """Pad variable-length sequences and return lengths for packing."""
    seqs, targets = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1.0)
    return seqs_padded, targets_padded, lengths


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

    Parameters
    ----------
    preds : np.ndarray, shape (T,), dtype int
        Binary predictions (0 = background, 1 = seizure).
    min_dur_sec : int
        Minimum event duration in windows (1 window = 1 second).

    Returns
    -------
    filtered : np.ndarray, shape (T,), dtype int
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
