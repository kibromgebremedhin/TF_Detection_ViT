"""
Model definitions used across all experiments.

DINOv2ECABinary   — final winning model (DINOv2 + ECA, used in train_figshare.py)
SelectorModel     — DINOv2 + any selector (used in selector_ablation.py)
BackboneModel     — any backbone + ECA  (used in backbone_experiment.py)
"""

import math
import torch
import torch.nn as nn
import timm

from src.selectors import create_selector, ECAGate


# =============================================================================
# Shared MLP head builder
# =============================================================================

def build_head(backbone_dim: int, head_hidden_dim: int, num_classes: int,
               drop_rate: float) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(backbone_dim),
        nn.Dropout(drop_rate),
        nn.Linear(backbone_dim, head_hidden_dim),
        nn.GELU(),
        nn.Dropout(drop_rate),
        nn.Linear(head_hidden_dim, num_classes),
    )
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return head


def forward_with_embedding(head: nn.Sequential, x: torch.Tensor):
    """Walk through head, capture 256-dim embedding after GELU (index 3)."""
    h = x
    emb = None
    for i, layer in enumerate(head):
        h = layer(h)
        if i == 3:
            emb = h
    return h, emb


# =============================================================================
# 1. Final model: DINOv2 ViT-B/14 + ECA  (used by train_figshare.py)
# =============================================================================

class DINOv2ECABinary(nn.Module):
    """
    DINOv2 ViT-B/14 backbone loaded via timm + ECAGate + MLP head.
    pretrained=True for training, False when loading a checkpoint.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "vit_base_patch14_dinov2",
        backbone_dim: int = 768,
        head_hidden_dim: int = 256,
        drop_rate: float = 0.1,
        img_size: int = 336,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            num_classes=0, global_pool="token", img_size=img_size,
        )
        self.backbone_dim     = backbone_dim
        self.feature_selector = ECAGate(backbone_dim)
        self.head             = build_head(backbone_dim, head_hidden_dim,
                                           num_classes, drop_rate)

    def forward(self, x, return_embedding=False):
        feat = self.feature_selector(self.backbone(x))
        if not return_embedding:
            return self.head(feat)
        return forward_with_embedding(self.head, feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def get_selector_params(self) -> int:
        return sum(p.numel() for p in self.feature_selector.parameters())


# =============================================================================
# 2. Selector ablation model: DINOv2 (torch.hub) + pluggable selector
#    Matches selector_ablation.py exactly.
# =============================================================================

class SelectorModel(nn.Module):
    """
    DINOv2 loaded via torch.hub + any selector from src.selectors.
    Used for Stage 2 ablation study.
    """

    def __init__(
        self,
        selector_name: str,
        num_classes: int = 2,
        backbone_name: str = "vit_base_patch14_dinov2",
        backbone_dim: int = 768,
        head_hidden_dim: int = 256,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0,
        global_pool="token",
        img_size=336,
    )
        # Set stochastic depth
        if drop_path_rate > 0 and hasattr(self.backbone, "blocks"):
            dpr = torch.linspace(0, drop_path_rate,
                                  len(self.backbone.blocks)).tolist()
            for i, blk in enumerate(self.backbone.blocks):
                if hasattr(blk, "drop_path"):
                    blk.drop_path.drop_prob = dpr[i]

        self.selector_name    = selector_name
        self.feature_selector = create_selector(selector_name, backbone_dim)
        self.head             = build_head(backbone_dim, head_hidden_dim,
                                           num_classes, drop_rate)

    def forward(self, x, return_embedding=False):
        feat = self.feature_selector(self.backbone(x))
        if not return_embedding:
            return self.head(feat)
        return forward_with_embedding(self.head, feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def get_selector_params(self) -> int:
        sel = self.feature_selector
        return sum(p.numel() for p in sel.parameters())


# =============================================================================
# 3. Backbone comparison model: any backbone (timm / open_clip) + ECA
#    Matches backbone_experiment.py exactly.
# =============================================================================

def _load_backbone(model_id: str, source: str):
    """Load a backbone as a plain feature extractor."""
    if source == "timm":
        return timm.create_model(model_id, pretrained=True, num_classes=0)
    elif source == "open_clip":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(model_id)
        if hasattr(clip_model.visual, "trunk"):
            return clip_model.visual.trunk
        backbone = clip_model.visual
        if hasattr(backbone, "proj"):
            backbone.proj = None
        return backbone
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'timm' or 'open_clip'.")


class BackboneModel(nn.Module):
    """
    Generic backbone + ECA + MLP head.
    Works for all six Stage 1 backbones (timm + open_clip).
    """

    def __init__(
        self,
        model_id: str,
        source: str,
        backbone_dim: int,
        img_size: int,
        num_classes: int = 2,
        head_hidden_dim: int = 256,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = _load_backbone(model_id, source)

        # Verify actual feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            actual = self.backbone(dummy).shape[-1]
        if actual != backbone_dim:
            print(f"    ⚠ backbone_dim mismatch: expected {backbone_dim}, "
                  f"got {actual}. Using {actual}.")
            backbone_dim = actual

        self.backbone_dim     = backbone_dim
        self.feature_selector = ECAGate(backbone_dim)
        self.head             = build_head(backbone_dim, head_hidden_dim,
                                           num_classes, drop_rate)

    def forward(self, x, return_embedding=False):
        feat = self.feature_selector(self.backbone(x))
        if not return_embedding:
            return self.head(feat)
        return forward_with_embedding(self.head, feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def get_selector_params(self) -> int:
        return sum(p.numel() for p in self.feature_selector.parameters())
