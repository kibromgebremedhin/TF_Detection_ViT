"""
Validate the OPTED folder structure and metadata CSV.

Usage
-----
    python preprocessing/prepare_opted_metadata.py \
        --opted_dir data/opted \
        --csv       data/opted/metadata_opted.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opted_dir", default="data/opted")
    p.add_argument("--csv",       default="data/opted/metadata_opted.csv")
    return p.parse_args()


def main():
    args      = parse_args()
    opted_dir = Path(args.opted_dir)

    df = pd.read_csv(args.csv)
    assert {"filename","label"}.issubset(df.columns)

    print(f"Shape   : {df.shape}")
    print(f"Labels  :\n{df['label'].value_counts().to_string()}")
    if "split" in df.columns:
        print(f"Splits  :\n{df['split'].value_counts().to_string()}")

    for sub in ["Normal","TF","TI"]:
        d = opted_dir / sub
        n = len(list(d.glob("*"))) if d.exists() else 0
        status = "FOUND" if d.exists() else "  MISSING"
        print(f"  {sub}/  {status}  ({n} files)")

    df["full_path"] = df.apply(
        lambda r: str(opted_dir / r["label"] / r["filename"]), axis=1
    )
    missing = df[~df["full_path"].apply(os.path.exists)]
    if len(missing):
        print(f"\n  {len(missing)} images not found.")
        print(f"   First 5: {missing['filename'].tolist()[:5]}")
    else:
        print(f"\n All {len(df)} images found on disk.")

    binary = df[df["label"].isin(["Normal","TF"])]
    test   = binary[binary["split"]=="test"] if "split" in binary.columns else binary
    print(f"\nBinary test set: {len(test)} images "
          f"(TF={len(test[test['label']=='TF'])}, "
          f"Normal={len(test[test['label']=='Normal'])})")


if __name__ == "__main__":
    main()
