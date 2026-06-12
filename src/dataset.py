"""
Dataset and transforms shared by all experiments.
SimpleDataset uses a flat data_dir + filename list, matching the original scripts.
"""

import os
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    Loads images from data_dir/filename.
    Labels are integer class indices.
    """

    def __init__(self, filenames, labels, data_dir, transform):
        self.filenames = list(filenames)
        self.labels    = list(labels)
        self.data_dir  = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img   = np.array(
            Image.open(self.data_dir / self.filenames[idx]).convert("RGB")
        )
        label = self.labels[idx]
        return self.transform(image=img)["image"], label


def get_train_transform(img_size: int = 336) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16,
            min_holes=1, min_height=8,  min_width=8,
            fill_value=0, p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 336) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Label mapping helpers ─────────────────────────────────────────────────────

LABEL_MAPS = {
    "C2_BINARY_TF": {"Normal": 0, "TF": 1},
    "C2_BINARY_TI": {"Normal": 0, "TI": 1},
    "C1_3CLASS":    {"Normal": 0, "TF": 1, "TI": 2},
    "C3_4CLASS":    {"Normal": 0, "TF": 1, "TI": 2, "TF_TI": 3},
}


def derive_labels_from_tfti(
    tfti_csv: str,
    data_dir: str,
    task: str = "C2_BINARY_TF",
    filename_template: str = "image{key}.jpg",
):
    """
    Read tfti.csv (columns: key, TF, TI) and return
    (filenames, labels, label_map) for the requested task.

    Skips rows whose image file does not exist in data_dir.
    """
    df = pd.read_csv(tfti_csv)
    assert {"key", "TF", "TI"}.issubset(df.columns), \
        f"Expected key/TF/TI columns, got {list(df.columns)}"

    label_map = LABEL_MAPS[task]
    rows = []

    for _, r in df.iterrows():
        tf, ti = int(r["TF"]), int(r["TI"])
        fn     = filename_template.format(key=int(r["key"]))
        fpath  = Path(data_dir) / fn

        if not fpath.exists():
            continue  # skip missing images silently

        if task == "C2_BINARY_TF":
            mapping = {(0,0): "Normal", (1,0): "TF", (0,1): None, (1,1): "TF"}
        elif task == "C2_BINARY_TI":
            mapping = {(0,0): "Normal", (1,0): None, (0,1): "TI", (1,1): "TI"}
        elif task == "C1_3CLASS":
            mapping = {
                (0,0): ["Normal"], (1,0): ["TF"],
                (0,1): ["TI"],     (1,1): ["TF", "TI"],
            }
        elif task == "C3_4CLASS":
            mapping = {
                (0,0): "Normal", (1,0): "TF",
                (0,1): "TI",     (1,1): "TF_TI",
            }
        else:
            raise ValueError(f"Unknown task '{task}'")

        lbl = mapping.get((tf, ti))
        if lbl is None:
            continue
        if isinstance(lbl, list):
            for l in lbl:
                rows.append({"filename": fn, "label_str": l})
        else:
            rows.append({"filename": fn, "label_str": lbl})

    result = pd.DataFrame(rows)
    result["label_int"] = result["label_str"].map(label_map)
    return result["filename"].tolist(), result["label_int"].tolist(), label_map
