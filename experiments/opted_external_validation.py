"""
External Validation on the OPTED Dataset.
Evaluates the five Figshare fold checkpoints on OPTED test split.

Usage
-----
    python experiments/opted_external_validation.py \
        --checkpoint_dir results/figshare_dinov2_eca \
        --opted_csv      data/opted/metadata_opted.csv \
        --opted_img_dir  data/opted \
        --output_dir     results/opted_validation \
        --split          test
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, RocCurveDisplay,
)
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import get_val_transform
from src.model import DINOv2ECABinary

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="results/figshare_dinov2_eca")
    p.add_argument("--opted_csv",      default="data/opted/metadata_opted.csv")
    p.add_argument("--opted_img_dir",  default="data/opted")
    p.add_argument("--output_dir",     default="results/opted_validation")
    p.add_argument("--split",          default="test",
                   choices=["test", "val", "train", "all"])
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--n_folds",        type=int, default=5)
    p.add_argument("--img_size",       type=int, default=336)
    p.add_argument("--device",         default=None)
    return p.parse_args()


class OPTEDDataset(Dataset):
    LABEL_MAP   = {"Normal": 0, "TF": 1}
    CLASS_NAMES = ["Normal", "TF"]

    def __init__(self, metadata_csv, image_dir, transform,
                 use_splits=None, exclude_ti=True):
        df = pd.read_csv(metadata_csv)
        assert {"filename", "label"}.issubset(df.columns)
        if use_splits is not None and "split" in df.columns:
            df = df[df["split"].isin(use_splits)].copy()
            print(f"  Split filter {use_splits}: {len(df)} rows")
        if exclude_ti:
            before = len(df)
            df = df[df["label"] != "TI"].copy()
            if len(df) < before:
                print(f"  Excluded {before-len(df)} TI rows")
        df["label_int"] = df["label"].map(self.LABEL_MAP)
        image_dir = Path(image_dir)
        df["full_path"] = df.apply(
            lambda r: str(image_dir / r["label"] / r["filename"]), axis=1
        )
        missing = df[~df["full_path"].apply(os.path.exists)]
        if len(missing):
            print(f"  WARNING: {len(missing)} images missing.")
            df = df[df["full_path"].apply(os.path.exists)].copy()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        print(f"\n  OPTED: {len(self.df)} images")
        for lbl, cnt in self.df["label"].value_counts().items():
            print(f"    {lbl}: {cnt}  ({100*cnt/len(self.df):.1f}%)")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["full_path"]).convert("RGB"))
        return self.transform(image=img)["image"], int(row["label_int"])

    @property
    def labels(self):    return self.df["label_int"].tolist()
    @property
    def filenames(self): return self.df["filename"].tolist()


def load_fold_model(ckpt_path, device, img_size=336):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = DINOv2ECABinary(
        num_classes=2, backbone_name="vit_base_patch14_dinov2",
        backbone_dim=768, head_hidden_dim=256, drop_rate=0.1,
        img_size=img_size, pretrained=False,
    ).to(device)
    full  = ckpt["model_state_dict"]
    keys  = set(model.state_dict().keys())
    filt  = {k: v for k, v in full.items() if k in keys}
    dropped = set(full.keys()) - keys
    if dropped: print(f"   Dropped: {dropped}")
    model.load_state_dict(filt, strict=True)
    model.eval()
    return model, {k: ckpt.get(k) for k in ["epoch","val_acc","val_f1"]}


@torch.no_grad()
def run_inference(model, loader, device):
    all_probs, all_labels = [], []
    for imgs, labs in tqdm(loader, desc="  Inferring", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            with autocast("cuda"): logits = model(imgs)
        else:
            logits = model(imgs)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labs.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels):
    preds = np.argmax(probs, axis=1)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    try:    auc = roc_auc_score(labels, probs[:, 1])
    except: auc = float("nan")
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0.0
    else:
        sens = spec = float("nan")
    return dict(accuracy=float(acc), f1_macro=float(f1), auc=float(auc),
                sensitivity=float(sens), specificity=float(spec),
                confusion_matrix=cm.tolist(),
                n_samples=int(len(labels)), n_positive=int(labels.sum()))


def print_results(title, m, meta=None):
    sep = "=" * 62
    print(f"\n{sep}\n  {title}")
    if meta: print(f"  Ckpt epoch={meta['epoch']} val_f1={meta['val_f1']:.4f}")
    print(sep)
    print(f"  Accuracy   : {100*m['accuracy']:.2f}%")
    print(f"  F1 Macro   : {m['f1_macro']:.4f}")
    print(f"  AUC-ROC    : {m['auc']:.4f}")
    print(f"  Sensitivity: {100*m['sensitivity']:.2f}%")
    print(f"  Specificity: {100*m['specificity']:.2f}%")
    print(f"  N          : {m['n_samples']}  (TF={m['n_positive']})")
    print(sep)


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_splits  = None if args.split == "all" else [args.split]

    print(f"\n{'='*62}")
    print(f"  DINOv2+ECA — OPTED External Validation")
    print(f"  Device: {device}   Splits: {use_splits or 'ALL'}")
    print(f"{'='*62}\n")

    transform = get_val_transform(args.img_size)
    ds        = OPTEDDataset(args.opted_csv, args.opted_img_dir,
                              transform, use_splits, exclude_ti=True)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=(device.type=="cuda"))

    fold_metrics, fold_probs_list, gt_labels = [], [], None

    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = Path(args.checkpoint_dir) / f"fold_{fold_idx}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[WARNING] Not found: {ckpt_path} — skipping"); continue

        print(f"\n── Fold {fold_idx}/{args.n_folds} ─────────────────────────────────────")
        model, meta = load_fold_model(ckpt_path, device, args.img_size)
        print(f"   Epoch={meta['epoch']} val_f1={meta['val_f1']:.4f}")

        probs, labels = run_inference(model, loader, device)
        if gt_labels is None: gt_labels = labels
        else: assert np.array_equal(gt_labels, labels)

        m = compute_metrics(probs, labels)
        m.update({"fold": fold_idx, "ckpt_epoch": meta["epoch"],
                  "ckpt_val_f1": meta["val_f1"]})
        fold_metrics.append(m); fold_probs_list.append(probs)
        print_results(f"Fold {fold_idx} — OPTED", m, meta)
        del model; torch.cuda.empty_cache()

    # Summary
    keys    = ["accuracy","f1_macro","auc","sensitivity","specificity"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if not math.isnan(m[k])]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))
    summary["per_fold"] = fold_metrics

    sep = "=" * 62
    print(f"\n{sep}\n  5-FOLD MEAN ± STD  (OPTED External Validation)\n{sep}")
    for k, lbl, pct in [
        ("accuracy","Accuracy   ",True),("f1_macro","F1 Macro   ",False),
        ("auc","AUC-ROC    ",False),("sensitivity","Sensitivity",True),
        ("specificity","Specificity",True),
    ]:
        m, s = summary[f"{k}_mean"], summary[f"{k}_std"]
        if pct: print(f"  {lbl}: {100*m:.2f}% ± {100*s:.2f}%")
        else:   print(f"  {lbl}: {m:.4f} ± {s:.4f}")
    print(sep)

    # Ensemble
    ens_p = np.mean(np.stack(fold_probs_list), axis=0)
    ens_m = compute_metrics(ens_p, gt_labels)
    ens_m["method"] = "mean_ensemble"
    print_results("5-Fold Mean Ensemble — OPTED", ens_m)

    # Save
    results = {
        "dataset": "OPTED", "model": "DINOv2 ViT-B/14 + ECAGate",
        "task": "C2_BINARY_TF", "splits_used": str(use_splits or "ALL"),
        "n_samples": len(ds), "per_fold_metrics": fold_metrics,
        "fold_summary": summary, "ensemble_metrics": ens_m,
    }
    with open(output_dir / "opted_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-image
    cn = OPTEDDataset.CLASS_NAMES
    per_img = [
        {"filename": ds.filenames[i], "true_label": int(gt_labels[i]),
         "predicted": int(np.argmax(ens_p[i])),
         "correct": bool(np.argmax(ens_p[i])==gt_labels[i]),
         "prob_Normal": float(ens_p[i,0]), "prob_TF": float(ens_p[i,1])}
        for i in range(len(gt_labels))
    ]
    with open(output_dir / "opted_per_image.json", "w") as f:
        json.dump(per_img, f, indent=2)

    # Figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Confusion matrix
    agg_cm = np.zeros((2,2), dtype=int)
    for m in fold_metrics: agg_cm += np.array(m["confusion_matrix"])
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    fig.suptitle("Confusion Matrix — OPTED External Validation",fontsize=12,fontweight="bold")
    for ax, data, fmt, ttl in zip(
        axes,
        [agg_cm, agg_cm.astype(float)/agg_cm.sum(axis=1,keepdims=True)],
        ["d",".3f"], ["Raw Counts","Row-Normalised"],
    ):
        sns.heatmap(data,annot=True,fmt=fmt,cmap="Blues",
                    xticklabels=cn,yticklabels=cn,linewidths=0.5,ax=ax,
                    annot_kws={"size":13,"weight":"bold"},
                    vmin=0 if fmt==".3f" else None,
                    vmax=1 if fmt==".3f" else None)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(ttl)
    plt.tight_layout()
    plt.savefig(fig_dir/"opted_confusion_matrix.png",dpi=150,bbox_inches="tight")
    plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(6,6))
    RocCurveDisplay.from_predictions(
        gt_labels, ens_p[:,1],
        name=f"Ensemble (AUC={ens_m['auc']:.4f})", ax=ax, color="#2196F3", lw=2)
    for i, (fp_, fm) in enumerate(zip(fold_probs_list, fold_metrics), 1):
        RocCurveDisplay.from_predictions(
            gt_labels, fp_[:,1],
            name=f"Fold {i} (AUC={fm['auc']:.4f})",
            ax=ax, color=plt.cm.tab10.colors[i-1], lw=1, alpha=0.6)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_title("ROC Curves — OPTED External Validation",fontsize=12,fontweight="bold")
    ax.legend(fontsize=8,loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir/"opted_roc_curves.png",dpi=150,bbox_inches="tight")
    plt.close()

    print(f"\n✅ Results → {output_dir / 'opted_results.json'}")
    print(f"✅ Figures → {fig_dir}/")


if __name__ == "__main__":
    main()
