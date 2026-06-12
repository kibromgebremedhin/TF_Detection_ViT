"""
Generate data/figshare/split.json for all experiments.

Reads tfti.csv, derives binary Normal/TF labels, checks all images exist,
and writes the 5-fold split used by every experiment script.

The split.json format has two equivalent sections:
  1. Flat arrays (train_filenames, val_filenames, test_filenames, + labels)
     → used by backbone_experiment.py and selector_ablation.py
  2. Per-fold dicts
     → used by train_figshare.py

Usage
-----
    python preprocessing/prepare_figshare_split.py \
        --data_dir  data/figshare \
        --csv       data/figshare/tfti.csv \
        --task      C2_BINARY_TF \
        --n_folds   5 \
        --seed      123
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LABEL_MAPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/figshare")
    p.add_argument("--csv",       default="data/figshare/tfti.csv")
    p.add_argument("--task",      default="C2_BINARY_TF",
                   choices=list(LABEL_MAPS.keys()))
    p.add_argument("--n_folds",   type=int, default=5)
    p.add_argument("--seed",      type=int, default=123)
    p.add_argument("--val_frac",  type=float, default=0.1)
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    assert {"key", "TF", "TI"}.issubset(df.columns)

    # ── Derive labels ──────────────────────────────────────────────────────
    label_map = LABEL_MAPS[args.task]
    rows = []
    for _, r in df.iterrows():
        tf, ti = int(r["TF"]), int(r["TI"])
        fn     = f"image{int(r['key'])}.jpg"
        if not (data_dir / fn).exists():
            continue
        if args.task == "C2_BINARY_TF":
            mapping = {(0,0):"Normal",(1,0):"TF",(0,1):None,(1,1):"TF"}
        elif args.task == "C2_BINARY_TI":
            mapping = {(0,0):"Normal",(1,0):None,(0,1):"TI",(1,1):"TI"}
        else:
            mapping = {(0,0):"Normal",(1,0):"TF",(0,1):"TI",(1,1):"TF"}
        lbl = mapping.get((tf, ti))
        if lbl is None:
            continue
        rows.append({"filename": fn, "label_str": lbl})

    result = pd.DataFrame(rows)
    result["label_int"] = result["label_str"].map(label_map)

    all_fns    = result["filename"].values
    all_labels = result["label_int"].values

    print(f"\nTask      : {args.task}   Label map: {label_map}")
    print(f"Images    : {len(all_fns)}")
    for lbl, cnt in zip(*np.unique(all_labels, return_counts=True)):
        names = {v: k for k, v in label_map.items()}
        print(f"  {names[lbl]}: {cnt}")

    # ── 5-fold split ──────────────────────────────────────────────────────
    skf   = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                             random_state=args.seed)
    folds = []

    all_tr_fns, all_va_fns, all_te_fns = [], [], []
    all_tr_lbl, all_va_lbl, all_te_lbl = [], [], []

    for fold_idx, (tv_idx, te_idx) in enumerate(skf.split(all_fns, all_labels), 1):
        tv_fns = all_fns[tv_idx];  tv_lbl = all_labels[tv_idx]
        te_fns = all_fns[te_idx];  te_lbl = all_labels[te_idx]

        n_val = max(1, int(len(tv_fns) * args.val_frac))
        sss   = StratifiedShuffleSplit(n_splits=1, test_size=n_val,
                                       random_state=args.seed + fold_idx)
        tr_sub, va_sub = next(sss.split(tv_fns, tv_lbl))

        tr_fns = tv_fns[tr_sub].tolist(); tr_lbl = tv_lbl[tr_sub].tolist()
        va_fns = tv_fns[va_sub].tolist(); va_lbl = tv_lbl[va_sub].tolist()
        te_fns = te_fns.tolist();         te_lbl = te_lbl.tolist()

        print(f"  Fold {fold_idx}: train={len(tr_fns)}  "
              f"val={len(va_fns)}  test={len(te_fns)}")

        folds.append({
            "fold":    fold_idx,
            "train":   {"paths": tr_fns, "labels": tr_lbl,
                        "filenames": tr_fns},
            "val":     {"paths": va_fns, "labels": va_lbl,
                        "filenames": va_fns},
            "test":    {"paths": te_fns, "labels": te_lbl,
                        "filenames": te_fns},
            "n_train": len(tr_fns),
            "n_val":   len(va_fns),
            "n_test":  len(te_fns),
        })

        all_tr_fns += tr_fns; all_tr_lbl += tr_lbl
        all_va_fns += va_fns; all_va_lbl += va_lbl
        all_te_fns += te_fns; all_te_lbl += te_lbl

    # ── Save ──────────────────────────────────────────────────────────────
    split_data = {
        # Per-fold format (train_figshare.py)
        "task":         args.task,
        "label_map":    label_map,
        "class_names":  list(label_map.keys()),
        "n_folds":      args.n_folds,
        "seed":         args.seed,
        "val_frac":     args.val_frac,
        "total_images": len(all_fns),
        "folds":        folds,
        # Flat format (backbone_experiment.py & selector_ablation.py)
        "train_filenames": all_tr_fns,
        "train_labels":    all_tr_lbl,
        "val_filenames":   all_va_fns,
        "val_labels":      all_va_lbl,
        "test_filenames":  all_te_fns,
        "test_labels":     all_te_lbl,
    }

    out = data_dir / "split.json"
    with open(out, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"\n✅ Split saved → {out}")


if __name__ == "__main__":
    main()
