"""
Shared utilities: reproducibility, EMA, AverageMeter, Mixup/CutMix.
"""

import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self._model = copy.deepcopy(model).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, m_p in zip(self._model.parameters(), model.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * m_p)
        for ema_b, m_b in zip(self._model.buffers(), model.buffers()):
            ema_b.copy_(m_b)

    def get_model(self) -> nn.Module:
        return self._model


class AverageMeter:
    """Tracks a running mean."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=x.device)
    cut_ratio = math.sqrt(1.0 - lam)
    ch = int(H * cut_ratio); cw = int(W * cut_ratio)
    cx = np.random.randint(W);  cy = np.random.randint(H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, y, y[idx], lam_adj


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
