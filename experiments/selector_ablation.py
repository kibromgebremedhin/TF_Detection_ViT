"""
Stage 2 — Feature Selector Ablation Study.
Compares SE-Gate, L0 Hard Concrete, ECA-Net, ECA+STG on DINOv2 ViT-B/14.

Usage
-----
    # Run all 4 selectors
    python experiments/selector_ablation.py

    # Run a single selector
    python experiments/selector_ablation.py --selector eca

    # List available selectors
    python experiments/selector_ablation.py --list
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import SelectorModel
from src.selectors import SELECTOR_REGISTRY
from src.train_utils import train_one_fold
from src.utils import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "figshare"
BINARY_SPLIT = DATA_DIR / "split.json"
SEED         = 123
N_FOLDS      = 5
NUM_CLASSES  = 2
CLASS_NAMES  = ["Normal", "TF"]

HPARAMS = {
    "seed": SEED, "n_folds": N_FOLDS, "epochs": 60, "patience": 20,
    "warmup_epochs": 8, "backbone_lr": 5e-6, "head_lr": 1e-3,
    "batch_size": 8, "drop_rate": 0.1, "mixup_alpha": 0.3,
    "cutmix_alpha": 1.0, "mix_prob": 0.5,
    "focal_alpha": [0.35, 0.65], "focal_gamma": 2.0, "label_smoothing": 0.05,
    "ema_decay": 0.999, "weight_decay": 0.01, "grad_clip": 1.0,
    "center_loss_lambda": 0.005, "center_lr": 0.5, "center_class_weights": True,
    "backbone": "dinov2_vitb14", "backbone_dim": 768,
    "head_hidden_dim": 256, "img_size": 336, "val_fraction": 0.1,
}

SELECTOR_NAMES = ["se_gate", "l0", "eca", "eca_stg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selector", default=None,
                   help="Run a single selector (default: all)")
    p.add_argument("--folds", default=None,
                   help="Comma-separated folds, e.g. 1,2")
    p.add_argument("--output_dir", default="results/selector_ablation")
    p.add_argument("--list", action="store_true")
    return p.parse_args()


def save_selector_summary(selector_name, fold_results, out_dir):
    accs  = [r["test_accuracy"] for r in fold_results]
    f1s   = [r["test_f1_macro"] for r in fold_results]
    aucs  = [r["test_auc"]      for r in fold_results]
    sens  = [r["sensitivity"]   for r in fold_results]
    specs = [r["specificity"]   for r in fold_results]
    summary = {
        "selector": selector_name,
        "selector_params": fold_results[0]["selector_params"],
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
    print(f"  {selector_name}: {100*summary['accuracy_mean']:.2f}% ± "
          f"{100*summary['accuracy_std']:.2f}%  (AUC {summary['auc_mean']:.4f})")
    return summary


def save_comparison(all_summaries, output_dir):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  5-FOLD ABLATION — FINAL COMPARISON")
    print(f"{'Selector':<12} {'Params':>8} {'Accuracy':>22} {'F1':>22} {'AUC':>20}")
    for s in all_summaries:
        print(f"  {s['selector']:<12} {s['selector_params']:>8} "
              f"{100*s['accuracy_mean']:>7.2f}%±{100*s['accuracy_std']:.2f}%  "
              f"{s['f1_macro_mean']:>7.4f}±{s['f1_macro_std']:.4f}  "
              f"{s['auc_mean']:>7.4f}±{s['auc_std']:.4f}")
    best = max(all_summaries, key=lambda s: s["auc_mean"])
    print(f"\n  🏆 Best AUC: {best['selector']}  {best['auc_mean']:.4f}")
    print(sep)
    comparison = {"selectors": {s["selector"]: s for s in all_summaries}}
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f" Comparison → {output_dir / 'comparison.json'}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.list:
        print("\nAvailable selectors:")
        for name in SELECTOR_NAMES:
            print(f"  {name}")
        return

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

    # Recompute folds identically to selector_ablation.py
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

    run_selectors = [args.selector] if args.selector else SELECTOR_NAMES
    run_fold_ids  = (
        [int(x) for x in args.folds.split(",")]
        if args.folds else list(range(1, N_FOLDS + 1))
    )
    output_dir = Path(args.output_dir)

    print(f"\n{'='*65}")
    print(f"  Selector Ablation | Device: {device} | Selectors: {run_selectors}")
    print(f"  Total images: {len(all_fns)} "
          f"(Normal={(all_labels==0).sum()} TF={(all_labels==1).sum()})")
    print(f"{'='*65}")

    seed_everything(SEED)
    all_summaries = []

    for sel_name in run_selectors:
        print(f"\n{'━'*65}")
        print(f"  SELECTOR: {sel_name}")
        print(f"{'━'*65}")

        sel_dir      = output_dir / sel_name
        fold_results = []

        for fold_idx in run_fold_ids:
            tr_f, tr_l, va_f, va_l, te_f, te_l = folds[fold_idx - 1]
            print(f"\n  --- Fold {fold_idx}/{N_FOLDS} | "
                  f"train={len(tr_f)} val={len(va_f)} test={len(te_f)} ---")

            seed_everything(SEED + fold_idx)
            model = SelectorModel(
                selector_name=sel_name,
                num_classes=NUM_CLASSES,
                backbone_name=HPARAMS["backbone"],
                backbone_dim=HPARAMS["backbone_dim"],
                head_hidden_dim=HPARAMS["head_hidden_dim"],
                drop_rate=HPARAMS["drop_rate"],
                drop_path_rate=0.1,
            )

            result = train_one_fold(
                model=model,
                fold_idx=fold_idx,
                tr_f=tr_f, tr_l=tr_l,
                va_f=va_f, va_l=va_l,
                te_f=te_f, te_l=te_l,
                data_dir=str(DATA_DIR),
                fold_dir=sel_dir / f"fold_{fold_idx}",
                hparams=HPARAMS,
                num_classes=NUM_CLASSES,
                class_names=CLASS_NAMES,
                device=device,
                experiment_name=sel_name,
                extra_hparams={"selector": sel_name},
            )
            result["selector"] = sel_name
            fold_results.append(result)

        if len(fold_results) == N_FOLDS:
            summary = save_selector_summary(sel_name, fold_results, sel_dir)
            all_summaries.append(summary)

    if len(all_summaries) == len(run_selectors):
        save_comparison(all_summaries, output_dir)


if __name__ == "__main__":
    main()
