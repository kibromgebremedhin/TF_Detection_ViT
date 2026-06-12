"""
Baseline Experiment — DINOv2 ViT-B/14 without ECA selector.

Identical training protocol to train_figshare.py but the ECA gate
is replaced with an Identity module (no feature selection at all).
This provides the ablation baseline for Table 5 in the thesis.

Usage
-----
    python experiments/baseline_experiment.py \
        --output_dir results/baseline_no_eca
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_utils import train_one_fold
from src.utils import seed_everything
from src.model import build_head, forward_with_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "figshare"
BINARY_SPLIT = DATA_DIR / "split.json"

SEED        = 123
N_FOLDS     = 5
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "TF"]

HPARAMS = {
    "seed": SEED, "n_folds": N_FOLDS, "epochs": 60, "patience": 20,
    "warmup_epochs": 8, "backbone_lr": 5e-6, "head_lr": 1e-3,
    "batch_size": 8, "drop_rate": 0.1, "mixup_alpha": 0.3,
    "cutmix_alpha": 1.0, "mix_prob": 0.5,
    "focal_alpha": [0.35, 0.65], "focal_gamma": 2.0, "label_smoothing": 0.05,
    "ema_decay": 0.999, "weight_decay": 0.01, "grad_clip": 1.0,
    "center_loss_lambda": 0.005, "center_lr": 0.5, "center_class_weights": True,
    "backbone": "vit_base_patch14_dinov2", "backbone_dim": 768,
    "head_hidden_dim": 256, "img_size": 336, "val_fraction": 0.1,
}


# =============================================================================
# Baseline model: DINOv2 (timm) + Identity + MLP head
# =============================================================================

class DINOv2Baseline(nn.Module):
    """
    DINOv2 ViT-B/14 with no feature selector (Identity gate).
    Identical head and training protocol to DINOv2ECABinary.
    Provides the no-selector ablation baseline.
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
        self.feature_selector = nn.Identity()   # ← no ECA, no gating
        self.head             = build_head(backbone_dim, head_hidden_dim,
                                           num_classes, drop_rate)

    def forward(self, x, return_embedding=False):
        feat = self.feature_selector(self.backbone(x))  # identity pass-through
        if not return_embedding:
            return self.head(feat)
        return forward_with_embedding(self.head, feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def get_selector_params(self) -> int:
        return 0   # Identity has no parameters


# =============================================================================
# CLI + main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline: DINOv2 ViT-B/14 without ECA selector"
    )
    p.add_argument("--output_dir", default="results/baseline_no_eca")
    p.add_argument("--folds", default=None,
                   help="Comma-separated folds to run, e.g. 1,2")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert BINARY_SPLIT.exists(), (
        f"Split not found: {BINARY_SPLIT}\n"
        f"Run: python preprocessing/prepare_figshare_split.py"
    )
    with open(BINARY_SPLIT) as f:
        split = json.load(f)

    all_fns    = np.array(
        split["train_filenames"] + split["val_filenames"] + split["test_filenames"]
    )
    all_labels = np.array(
        split["train_labels"] + split["val_labels"] + split["test_labels"]
    )

    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fi, (tv_idx, te_idx) in enumerate(skf.split(all_fns, all_labels), 1):
        tv_f = all_fns[tv_idx]; tv_l = all_labels[tv_idx]
        te_f = all_fns[te_idx].tolist(); te_l = all_labels[te_idx].tolist()
        n_val = max(1, int(len(tv_f) * HPARAMS["val_fraction"]))
        sss   = StratifiedShuffleSplit(1, test_size=n_val, random_state=SEED + fi)
        tr_s, va_s = next(sss.split(tv_f, tv_l))
        folds.append((tv_f[tr_s].tolist(), tv_l[tr_s].tolist(),
                      tv_f[va_s].tolist(), tv_l[va_s].tolist(),
                      te_f, te_l))

    run_fold_ids = (
        [int(x) for x in args.folds.split(",")]
        if args.folds else list(range(1, N_FOLDS + 1))
    )
    output_dir = Path(args.output_dir)

    print(f"\n{'='*65}")
    print(f"  BASELINE: DINOv2 ViT-B/14 (no ECA selector)")
    print(f"  Device: {device}   Folds: {run_fold_ids}")
    print(f"  Selector params: 0  (Identity)")
    print(f"{'='*65}")

    seed_everything(SEED)
    fold_results = []

    for fold_idx in run_fold_ids:
        tr_f, tr_l, va_f, va_l, te_f, te_l = folds[fold_idx - 1]
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} | "
              f"train={len(tr_f)} val={len(va_f)} test={len(te_f)} ---")

        seed_everything(SEED + fold_idx)
        model = DINOv2Baseline(
            num_classes=NUM_CLASSES,
            backbone_name=HPARAMS["backbone"],
            backbone_dim=HPARAMS["backbone_dim"],
            head_hidden_dim=HPARAMS["head_hidden_dim"],
            drop_rate=HPARAMS["drop_rate"],
            img_size=HPARAMS["img_size"],
            pretrained=True,
        )

        result = train_one_fold(
            model=model,
            fold_idx=fold_idx,
            tr_f=tr_f, tr_l=tr_l,
            va_f=va_f, va_l=va_l,
            te_f=te_f, te_l=te_l,
            data_dir=str(DATA_DIR),
            fold_dir=output_dir / f"fold_{fold_idx}",
            hparams=HPARAMS,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            device=device,
            experiment_name="baseline",
            extra_hparams={"selector": "none", "backbone": HPARAMS["backbone"]},
        )
        result.update({"selector": "none", "backbone": HPARAMS["backbone"]})
        fold_results.append(result)

    if len(fold_results) == N_FOLDS:
        accs  = [r["test_accuracy"]  for r in fold_results]
        f1s   = [r["test_f1_macro"]  for r in fold_results]
        aucs  = [r["test_auc"]       for r in fold_results]
        sens  = [r["sensitivity"]    for r in fold_results]
        specs = [r["specificity"]    for r in fold_results]
        summary = {
            "selector": "none (baseline)",
            "selector_params": 0,
            "accuracy_mean":    float(np.mean(accs)), "accuracy_std":    float(np.std(accs)),
            "f1_macro_mean":    float(np.mean(f1s)),  "f1_macro_std":    float(np.std(f1s)),
            "auc_mean":         float(np.mean(aucs)), "auc_std":         float(np.std(aucs)),
            "sensitivity_mean": float(np.mean(sens)), "sensitivity_std": float(np.std(sens)),
            "specificity_mean": float(np.mean(specs)),"specificity_std": float(np.std(specs)),
            "accuracy_per_fold": accs, "per_fold_results": fold_results,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*65}")
        print(f"  BASELINE (no ECA) — 5-FOLD RESULTS")
        print(f"  Accuracy   : {100*summary['accuracy_mean']:.2f}% ± {100*summary['accuracy_std']:.2f}%")
        print(f"  F1 Macro   : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
        print(f"  AUC-ROC    : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"  Sensitivity: {100*summary['sensitivity_mean']:.2f}% ± {100*summary['sensitivity_std']:.2f}%")
        print(f"  Specificity: {100*summary['specificity_mean']:.2f}% ± {100*summary['specificity_std']:.2f}%")
        print(f"{'='*65}")
        print(f"✅ Summary → {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
