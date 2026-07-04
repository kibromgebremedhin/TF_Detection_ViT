"""
Stage 1 — Backbone Architecture Comparison.
Evaluates 5 alternative backbones (all paired with ECA selector).

Usage
-----
    python experiments/backbone_experiment.py --backbone eva02_base
    python experiments/backbone_experiment.py --backbone convnext_tiny
    python experiments/backbone_experiment.py --backbone biomedclip
    python experiments/backbone_experiment.py --backbone convnextv2_base
    python experiments/backbone_experiment.py --backbone mobilenetv3_large
    python experiments/backbone_experiment.py --list
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import BackboneModel
from src.train_utils import train_one_fold
from src.utils import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "figshare"
BINARY_SPLIT = DATA_DIR / "split.json"

SEED        = 123
N_FOLDS     = 5
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "TF"]

# ── Backbone registry ─────────────────────────────────────────────────────────
BACKBONE_REGISTRY = {
    "convnextv2_base": {
        "model_id":    "convnextv2_base.fcmae_ft_in22k_in1k",
        "source":      "timm",
        "dim":         1024,
        "img_size":    336,
        "params_M":    88.7,
        "pretraining": "FCMAE + IN-22k + IN-1k",
        "notes":       "CNN-based, masked autoencoder pre-training",
    },
    "eva02_base": {
        "model_id":    "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        "source":      "timm",
        "dim":         768,
        "img_size":    448,
        "params_M":    87.1,
        "pretraining": "MIM + IN-22k + IN-1k",
        "notes":       "EVA-style masked image modeling ViT-B/14",
    },
    "biomedclip": {
        "model_id":    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "source":      "open_clip",
        "dim":         768,
        "img_size":    224,
        "params_M":    86.0,
        "pretraining": "PMC-15M (biomedical image-text pairs)",
        "notes":       "Domain-specific biomedical CLIP ViT-B/16",
    },
    "convnext_tiny": {
        "model_id":    "convnext_tiny.in12k_ft_in1k",
        "source":      "timm",
        "dim":         768,
        "img_size":    336,
        "params_M":    28.6,
        "pretraining": "IN-12k + IN-1k",
        "notes":       "Lightweight CNN baseline",
    },
    "mobilenetv3_large": {
        "model_id":    "mobilenetv3_large_100.ra_in1k",
        "source":      "timm",
        "dim":         960,
        "img_size":    336,
        "params_M":    5.4,
        "pretraining": "IN-1k",
        "notes":       "Mobile-efficient baseline, smallest model",
    },
}


def get_hparams(backbone_key: str) -> dict:
    cfg = BACKBONE_REGISTRY[backbone_key]
    return {
        "seed": SEED, "n_folds": N_FOLDS, "epochs": 60, "patience": 20,
        "warmup_epochs": 8, "backbone_lr": 5e-6, "head_lr": 1e-3,
        "batch_size": cfg.get("batch_size", 8), "drop_rate": 0.1,
        "mixup_alpha": 0.3, "cutmix_alpha": 1.0, "mix_prob": 0.5,
        "focal_alpha": [0.35, 0.65], "focal_gamma": 2.0, "label_smoothing": 0.05,
        "ema_decay": 0.999, "weight_decay": 0.01, "grad_clip": 1.0,
        "center_loss_lambda": 0.005, "center_lr": 0.5, "center_class_weights": True,
        "backbone":         cfg["model_id"],
        "backbone_source":  cfg["source"],
        "backbone_dim":     cfg["dim"],
        "head_hidden_dim":  256,
        "img_size":         cfg["img_size"],
        "val_fraction":     0.1,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="5-fold CV backbone comparison with ECA selector"
    )
    p.add_argument("--backbone",   type=str, default=None)
    p.add_argument("--folds",      type=str, default=None,
                   help="Comma-separated folds, e.g. 3,4,5")
    p.add_argument("--output_dir", default="results/backbone_comparison")
    p.add_argument("--list",       action="store_true")
    return p.parse_args()


def save_summary(backbone_key, fold_results, out_dir):
    accs  = [r["test_accuracy"] for r in fold_results]
    f1s   = [r["test_f1_macro"] for r in fold_results]
    aucs  = [r["test_auc"]      for r in fold_results]
    sens  = [r["sensitivity"]   for r in fold_results]
    specs = [r["specificity"]   for r in fold_results]

    summary = {
        "backbone":         backbone_key,
        "selector":         "ECA",
        "n_folds":          len(fold_results),
        "selector_params":  fold_results[0]["selector_params"],
        "accuracy_mean":    float(np.mean(accs)), "accuracy_std":    float(np.std(accs)),
        "f1_macro_mean":    float(np.mean(f1s)),  "f1_macro_std":    float(np.std(f1s)),
        "auc_mean":         float(np.mean(aucs)), "auc_std":         float(np.std(aucs)),
        "sensitivity_mean": float(np.mean(sens)), "sensitivity_std": float(np.std(sens)),
        "specificity_mean": float(np.mean(specs)),"specificity_std": float(np.std(specs)),
        "accuracy_per_fold": accs, "per_fold_results": fold_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cfg = BACKBONE_REGISTRY[backbone_key]
    print(f"\n{'='*70}")
    print(f"  {backbone_key} + ECA — 5-FOLD RESULTS")
    print(f"  {cfg['notes']}")
    print(f"  Accuracy   : {100*summary['accuracy_mean']:.2f}% ± {100*summary['accuracy_std']:.2f}%")
    print(f"  F1 Macro   : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"  AUC-ROC    : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  Sensitivity: {100*summary['sensitivity_mean']:.2f}% ± {100*summary['sensitivity_std']:.2f}%")
    print(f"  Specificity: {100*summary['specificity_mean']:.2f}% ± {100*summary['specificity_std']:.2f}%")
    print(f"  Per-fold   : {[f'{100*a:.1f}%' for a in accs]}")
    print(f"{'='*70}\n")
    print(f" Summary → {out_dir / 'summary.json'}")
    return summary


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.list:
        print(f"\n  {'Key':<20} {'Params':>8} {'Dim':>5} {'Img':>5} {'Source':<10} Notes")
        print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*5} {'-'*10} {'-'*40}")
        for key, cfg in BACKBONE_REGISTRY.items():
            print(f"  {key:<20} {cfg['params_M']:>6.1f}M {cfg['dim']:>5} "
                  f"{cfg['img_size']:>5} {cfg['source']:<10} {cfg['notes']}")
        return

    if not args.backbone:
        print("Error: --backbone required. Use --list to see options.")
        sys.exit(1)

    backbone_key = args.backbone
    assert backbone_key in BACKBONE_REGISTRY, \
        f"Unknown backbone '{backbone_key}'. Run --list for options."

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

    # Recompute same folds as selector_ablation
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    hparams = get_hparams(backbone_key)
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fi, (tv_idx, te_idx) in enumerate(skf.split(all_fns, all_labels), 1):
        tv_f = all_fns[tv_idx]; tv_l = all_labels[tv_idx]
        te_f = all_fns[te_idx].tolist(); te_l = all_labels[te_idx].tolist()
        n_val = max(1, int(len(tv_f) * hparams["val_fraction"]))
        sss   = StratifiedShuffleSplit(1, test_size=n_val, random_state=SEED + fi)
        tr_s, va_s = next(sss.split(tv_f, tv_l))
        folds.append((tv_f[tr_s].tolist(), tv_l[tr_s].tolist(),
                      tv_f[va_s].tolist(), tv_l[va_s].tolist(),
                      te_f, te_l))

    run_fold_ids = (
        [int(x) for x in args.folds.split(",")]
        if args.folds else list(range(1, N_FOLDS + 1))
    )
    output_dir = Path(args.output_dir) / backbone_key

    cfg_info = BACKBONE_REGISTRY[backbone_key]
    print(f"\n{'='*70}")
    print(f"  BACKBONE: {backbone_key} + ECA")
    print(f"  {cfg_info['model_id']}  ({cfg_info['dim']}-dim, {cfg_info['params_M']}M)")
    print(f"  Device: {device}   Folds: {run_fold_ids}")
    print(f"{'='*70}")

    seed_everything(SEED)
    fold_results = []

    for fold_idx in run_fold_ids:
        tr_f, tr_l, va_f, va_l, te_f, te_l = folds[fold_idx - 1]
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} | "
              f"train={len(tr_f)} val={len(va_f)} test={len(te_f)} ---")

        seed_everything(SEED + fold_idx)
        model = BackboneModel(
            model_id=cfg_info["model_id"],
            source=cfg_info["source"],
            backbone_dim=cfg_info["dim"],
            img_size=cfg_info["img_size"],
            num_classes=NUM_CLASSES,
            head_hidden_dim=256,
            drop_rate=0.1,
        )

        result = train_one_fold(
            model=model,
            fold_idx=fold_idx,
            tr_f=tr_f, tr_l=tr_l,
            va_f=va_f, va_l=va_l,
            te_f=te_f, te_l=te_l,
            data_dir=str(DATA_DIR),
            fold_dir=output_dir / f"fold_{fold_idx}",
            hparams=hparams,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            device=device,
            experiment_name=backbone_key,
            extra_hparams={
                "backbone": backbone_key,
                "selector": "ECA",
                "experiment_group": "backbone_comparison",
            },
        )
        result.update({"backbone": backbone_key, "selector": "ECA"})
        fold_results.append(result)

        # Check if all 5 folds now exist (resume-friendly)
        all_results = []
        for fi in range(1, N_FOLDS + 1):
            rp = output_dir / f"fold_{fi}" / "results.json"
            if rp.exists():
                with open(rp) as f2:
                    all_results.append(json.load(f2))
        if len(all_results) == N_FOLDS:
            save_summary(backbone_key, all_results, output_dir)

    if len(fold_results) == N_FOLDS and len(run_fold_ids) == N_FOLDS:
        save_summary(backbone_key, fold_results, output_dir)


if __name__ == "__main__":
    main()
