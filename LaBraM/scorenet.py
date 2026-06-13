import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ScoreNet(nn.Module):
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

        total_N = Z.shape[1]
        device = Z.device

        candidate = torch.sigmoid(self.a1 @ Z + self.b1)   
        score = torch.tanh(self.a2 @ Z + self.b2)          

        binary = (candidate >= self.gamma).long()

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
        group_ids = new_group.long().cumsum(0) - 1
        n_groups = group_ids[-1].item() + 1

        ones = torch.ones(total_N, device=device)
        g_sum = torch.zeros(n_groups, device=device).scatter_add_(0, group_ids, score)
        g_cnt = torch.zeros(n_groups, device=device).scatter_add_(0, group_ids, ones)
        g_mean = g_sum / g_cnt

        o_l = torch.sigmoid(self.a3 * g_mean[group_ids] + self.b3)

        yhat = torch.sigmoid(self.a4 * candidate * o_l + self.b4)
        return yhat

def build_toeplitz(z, w):
    n = len(z)
    filt_len = 2 * w + 1
    padded = np.concatenate([np.zeros(w, dtype=z.dtype), z, np.zeros(w, dtype=z.dtype)])
    idx = np.arange(n)[None, :] + np.arange(filt_len)[:, None]   # (filt_len, n)
    return padded[idx]

def log_dice_loss(yhat, y, eps=1e-7):
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
    yhat = yhat.clamp(eps, 1.0 - eps)
    intersection = (y * yhat).sum()
    return 1.0 - 2.0 * intersection / (y.sum() + yhat.sum() + eps)


def weighted_bce_loss(yhat, y, pos_weight=1.0, eps=1e-7):
    yhat = yhat.clamp(eps, 1.0 - eps)
    w = torch.where(y == 1, pos_weight, 1.0)
    return -(w * (y * torch.log(yhat) + (1.0 - y) * torch.log(1.0 - yhat))).mean()


def combined_loss(yhat, y, pos_weight=17.0, alpha=0.5, eps=1e-7):
    return alpha * dice_loss(yhat, y, eps) + (1.0 - alpha) * weighted_bce_loss(yhat, y, pos_weight, eps)

class ProbSequenceDataset(Dataset):

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
        return (torch.from_numpy(Z),torch.from_numpy(lab), n)


def collate_fn(batch):
    Zs, labels, ns = zip(*batch)
    Z_cat = torch.cat(Zs, dim=1)
    labels_cat = torch.cat(labels, dim=0)
    n_samples = list(ns)
    return Z_cat, labels_cat, n_samples


def hard_constraints(preds, min_dur_sec=10):
    filtered = preds.copy()
    padded = np.concatenate(([0], filtered, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) < min_dur_sec:
            filtered[s:e] = 0
    return filtered
