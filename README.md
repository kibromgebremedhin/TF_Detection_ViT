# DINOv2 + ECA - Trachoma Classification

MSc thesis codebase for binary trachoma follicular (TF) classification using
DINOv2 ViT-B/14 with an ECA channel-attention gate, trained on the Figshare
dataset and externally validated on OPTED.

---

## Project Structure

```
TF_Detection_ViT/
├── src/
│   ├── model.py           ← DINOv2ECABinary, SelectorModel, BackboneModel
│   ├── selectors.py       ← SEGate, L0, ECAGate, ECASTGHybrid + factory
│   ├── dataset.py         ← SimpleDataset, transforms, label mapping
│   ├── losses.py          ← FocalLoss
│   ├── utils.py           ← EMA, AverageMeter, seed_everything
│   ├── logging_utils.py   ← EpochLogger, save_fold_artifacts
│   └── train_utils.py     ← Shared train_one_fold() loop
├── experiments/
│   ├── train_figshare.py            ← Final model: DINOv2+ECA 5-fold CV
│   ├── backbone_experiment.py       ← Stage 1: 5 backbone comparison
│   ├── selector_ablation.py         ← Stage 2: 4 selector ablation
│   ├── baseline_experiment.py       ← Baseline: DINOv2 without ECA
│   └── opted_external_validation.py ← OPTED external validation
├── preprocessing/
│   ├── prepare_figshare_split.py    ← Generates data/figshare/split.json
│   └── prepare_opted_metadata.py   ← Validates OPTED folder structure
├── configs/
│   └── default.yaml                ← All hyperparameters
├── data/
│   ├── figshare/
│   │   ├── tfti.csv                ← Label file (key, TF, TI columns)
│   │   └── *.jpg                   ← Images: image1.jpg … image1656.jpg
│   └── opted/
│       ├── metadata_opted.csv
│       ├── Normal/
│       ├── TF/
│       └── TI/
├── results/                        ← Created at runtime
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/kibromgebremedhin/TF_Detection_ViT.git
cd TF_Detection_ViT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preparation

### Figshare

```bash
# Place images in data/figshare/ as image1.jpg, image2.jpg, ...
# tfti.csv is already included in the repo.

python preprocessing/prepare_figshare_split.py
# → writes data/figshare/split.json
```

### OPTED

OPTED dataset: https://github.com/kibromgebremedhin/OPTED_dataset

```bash
# Organise images into data/opted/Normal/, data/opted/TF/, data/opted/TI/
python preprocessing/prepare_opted_metadata.py
```

---

## Running Experiments

### Step 1 — Generate split (required for all experiments)
```bash
python preprocessing/prepare_figshare_split.py
```

### Step 2 — Final model (DINOv2 + ECA)
```bash
python experiments/train_figshare.py \
    --output_dir results/figshare_dinov2_eca
```

### Step 3 — Baseline (DINOv2 without ECA)
```bash
python experiments/baseline_experiment.py \
    --output_dir results/baseline_no_eca
```

### Step 4 — Stage 1: Backbone comparison
```bash
# Run each backbone separately (one at a time, each takes ~2 hours on GPU)
python experiments/backbone_experiment.py --backbone eva02_base
python experiments/backbone_experiment.py --backbone convnext_tiny
python experiments/backbone_experiment.py --backbone convnextv2_base
python experiments/backbone_experiment.py --backbone biomedclip
python experiments/backbone_experiment.py --backbone mobilenetv3_large

# List available backbones
python experiments/backbone_experiment.py --list
```

### Step 5 — Stage 2: Selector ablation
```bash
# Run all 4 selectors sequentially
python experiments/selector_ablation.py

# Or run a single selector
python experiments/selector_ablation.py --selector eca
python experiments/selector_ablation.py --selector se_gate
python experiments/selector_ablation.py --selector l0
python experiments/selector_ablation.py --selector eca_stg
```

### Step 6 — OPTED external validation
```bash
python experiments/opted_external_validation.py \
    --checkpoint_dir results/figshare_dinov2_eca \
    --opted_img_dir  data/opted \
    --output_dir     results/opted_validation
```

---

## Results

### Figshare Internal (5-Fold CV)
| Accuracy | F1 Macro | AUC-ROC | Sensitivity | Specificity |
|----------|----------|---------|-------------|-------------|
| 91.66±0.97% | 0.9069±0.011 | 0.9606±0.007 | 87.28±2.44% | 93.92±1.05% |


---

## Model Architecture

```
Input [B, 3, 336, 336]
  → DINOv2 ViT-B/14    → CLS token [B, 768]
  → ECAGate (5 params) → [B, 768]
  → LayerNorm → Dropout(0.1) → Linear(768→256) → GELU   ← center loss anchor
  → Dropout(0.1) → Linear(256→2)
```

**Training:**
- Phase 1 (8 epochs): backbone frozen, head + ECA at lr=1e-3
- Phase 2 (up to 52 epochs): full fine-tune, backbone lr=5e-6, cosine LR
- Loss: FocalLoss(α=[0.35,0.65], γ=2) 
- EMA decay=0.999,  WeightedRandomSampler

---
