"""
Loss functions: FocalLoss 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.alpha = (
            torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        )

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction="none",
                                label_smoothing=self.label_smoothing)
        pt    = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha.to(logits.device)[targets] * focal
        return focal.mean()


# class CenterLoss(nn.Module):
#     def __init__(self, num_classes: int, feat_dim: int):
#         super().__init__()
#         self.num_classes = num_classes
#         self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)

#     def forward(self, features, labels):
#         return 0.5 * ((features - self.centers[labels]) ** 2).sum(dim=1).mean()

#     @torch.no_grad()
#     def update_centers(self, features, labels, lr=0.5, class_weights=None):
#         for c in range(self.num_classes):
#             mask = labels == c
#             if mask.sum() == 0:
#                 continue
#             diff = self.centers[c] - features[mask].mean(dim=0)
#             w = class_weights[c].item() if class_weights is not None else 1.0
#             self.centers[c] -= lr * w * diff
