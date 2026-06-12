"""
Final model training: DINOv2 ViT-B/14 + ECA, 5-fold CV on Figshare.

Usage
-----
    python experiments/train_figshare.py \
        --data_dir   data/figshare \
        --split_json data/figshare/split.json \
        --output_dir results/figshare_dinov2_eca
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import DINOv2ECABinary
from src.train_utils import train_one_fold
from src.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/figshare")
    p.add_argument("--split_json", default="data/figshare/split.json")
    p.add_argument("--output_dir", default="results/figshare_dinov2_eca")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--folds",      default=None,
                   help="Comma-separated folds to run, e.g. 1,2")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config)     as f: cfg = yaml.safe_load(f)
    with open(args.split_json) as f: split = json.load(f)

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map   = split["label_map"]
    class_names = split["class_names"]
    num_classes = len(label_map)
    run_folds   = (
        [int(x) for x in args.folds.split(",")]
        if args.folds else list(range(1, split["n_folds"] + 1))
    )

    print(f"\n{'='*65}")
    print(f"  DINOv2+ECA — Figshare Training")
    print(f"  Device  : {device}   Task : {split['task']}")
    print(f"  Folds   : {run_folds}")
    print(f"{'='*65}\n")

    seed_everything(cfg["seed"])
    fold_results = []

    for fold_idx in run_folds:
        fi = split["folds"][fold_idx - 1]
        print(f"\n{'─'*55}")
        print(f"  FOLD {fold_idx}  train={fi['n_train']}  val={fi['n_val']}  test={fi['n_test']}")
        print(f"{'─'*55}")

        seed_everything(cfg["seed"] + fold_idx)
        model = DINOv2ECABinary(
            num_classes=num_classes,
            backbone_name=cfg["backbone"],
            backbone_dim=cfg["backbone_dim"],
            head_hidden_dim=cfg["head_hidden_dim"],
            drop_rate=cfg["drop_rate"],
            img_size=cfg["img_size"],
            pretrained=True,
        )

        result = train_one_fold(
            model=model,
            fold_idx=fold_idx,
            tr_f=fi["train"]["paths"], tr_l=fi["train"]["labels"],
            va_f=fi["val"]["paths"],   va_l=fi["val"]["labels"],
            te_f=fi["test"]["paths"],  te_l=fi["test"]["labels"],
            data_dir=args.data_dir,
            fold_dir=output_dir / f"fold_{fold_idx}",
            hparams=cfg,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            experiment_name="dinov2_eca",
            extra_hparams={"selector": "ECA", "backbone": cfg["backbone"]},
        )
        result.update({"backbone": cfg["backbone"], "selector": "ECA", "fold": fold_idx})
        fold_results.append(result)

    if len(fold_results) == split["n_folds"]:
        _save_summary(fold_results, output_dir, cfg)


def _save_summary(fold_results, output_dir, cfg):
    from src.train_utils import _summary_stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns

    accs  = [r["test_accuracy"]  for r in fold_results]
    f1s   = [r["test_f1_macro"]  for r in fold_results]
    aucs  = [r["test_auc"]       for r in fold_results]
    sens  = [r["sensitivity"]    for r in fold_results]
    specs = [r["specificity"]    for r in fold_results]

    summary = {
        "backbone": cfg["backbone"], "selector": "ECA",
        "accuracy_mean":    float(np.mean(accs)), "accuracy_std":    float(np.std(accs)),
        "f1_macro_mean":    float(np.mean(f1s)),  "f1_macro_std":    float(np.std(f1s)),
        "auc_mean":         float(np.mean(aucs)), "auc_std":         float(np.std(aucs)),
        "sensitivity_mean": float(np.mean(sens)), "sensitivity_std": float(np.std(sens)),
        "specificity_mean": float(np.mean(specs)),"specificity_std": float(np.std(specs)),
        "accuracy_per_fold": accs, "per_fold_results": fold_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DINOv2+ECA — 5-FOLD RESULTS")
    print(f"  Accuracy   : {100*summary['accuracy_mean']:.2f}% ± {100*summary['accuracy_std']:.2f}%")
    print(f"  F1 Macro   : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"  AUC-ROC    : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  Sensitivity: {100*summary['sensitivity_mean']:.2f}% ± {100*summary['sensitivity_std']:.2f}%")
    print(f"  Specificity: {100*summary['specificity_mean']:.2f}% ± {100*summary['specificity_std']:.2f}%")
    print(f"{'='*65}\n")
    print(f"✅ Summary → {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
