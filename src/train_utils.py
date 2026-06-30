"""
Shared fold training loop.

All three experiments (train_figshare, backbone_experiment, selector_ablation,
baseline_experiment) use the same protocol — only the model instance differs.
This module provides train_one_fold() to avoid code duplication.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import SimpleDataset, get_train_transform, get_val_transform
from src.losses import CenterLoss, FocalLoss
from src.logging_utils import EpochLogger, save_fold_artifacts
from src.utils import AverageMeter, EMA, cutmix_data, mixup_criterion, mixup_data


def train_one_fold(
    model: nn.Module,
    fold_idx: int,
    tr_f, tr_l,
    va_f, va_l,
    te_f, te_l,
    data_dir: str,
    fold_dir: Path,
    hparams: dict,
    num_classes: int,
    class_names: list,
    device: torch.device,
    experiment_name: str = "",
    extra_hparams: dict = None,
) -> dict:
    """
    Train and evaluate one fold.

    Parameters
    ----------
    model         : already-instantiated nn.Module (on CPU — moved inside)
    fold_idx      : 1-based fold index
    tr_f, va_f, te_f : filename lists
    tr_l, va_l, te_l : integer label lists
    data_dir      : directory containing flat image files
    fold_dir      : where to save checkpoints and logs
    hparams       : config dict (keys match configs/default.yaml)
    Returns
    -------
    fold_result   : dict with all test metrics
    """
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    model    = model.to(device)
    img_size = hparams["img_size"]

    tr_arr = np.array(tr_l)
    counts = np.bincount(tr_arr, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)

    # ── Datasets ──────────────────────────────────────────────────────────
    import os
    nw = min(4, os.cpu_count() or 1)

    train_ds = SimpleDataset(tr_f, tr_l, data_dir, get_train_transform(img_size))
    val_ds   = SimpleDataset(va_f, va_l, data_dir, get_val_transform(img_size))
    test_ds  = SimpleDataset(te_f, te_l, data_dir, get_val_transform(img_size))

    sampler = WeightedRandomSampler(
        [weights[l] for l in tr_l], len(tr_l), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"],
                              sampler=sampler, num_workers=nw,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=hparams["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=hparams["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)

    # ── Losses ────────────────────────────────────────────────────────────
    focal_criterion = FocalLoss(
        alpha=hparams["focal_alpha"],
        gamma=hparams["focal_gamma"],
        label_smoothing=hparams["label_smoothing"],
    ).to(device)

    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=hparams["head_hidden_dim"],
    ).to(device)

    center_class_weights = torch.tensor(
        weights / weights.sum() * num_classes, dtype=torch.float32
    ).to(device)

    # ── EMA + AMP ─────────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None
    ema     = EMA(model, decay=hparams["ema_decay"])

    has_reg = hasattr(model.feature_selector, "get_reg_loss") \
              if hasattr(model, "feature_selector") else False

    # ── Phase 1 optimiser (backbone frozen) ──────────────────────────────
    model.freeze_backbone()

    def non_backbone_params():
        bb_ids = {id(p) for p in model.backbone.parameters()}
        return [p for p in model.parameters()
                if id(p) not in bb_ids and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": non_backbone_params(), "lr": hparams["head_lr"]}
         ] + [{"params": center_criterion.parameters(), "lr": hparams["head_lr"]}],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = None

    # ── Logger ────────────────────────────────────────────────────────────
    epoch_logger = EpochLogger(
        log_path=fold_dir / "logs.log",
        has_reg=has_reg,
    )

    best_f1  = best_acc = 0.0
    patience = 0
    t_start  = time.time()
    best_val_probs = best_val_labels = None

    # ══════════════════ TRAINING LOOP ══════════════════════════════════════
    for epoch in range(1, hparams["epochs"] + 1):
        phase = "frozen" if epoch <= hparams["warmup_epochs"] else "finetuning"
        epoch_logger.start_epoch(epoch, phase)

        # Phase 1 → 2 transition
        if epoch == hparams["warmup_epochs"] + 1:
            model.unfreeze_backbone()
            bb_params  = [p for p in model.backbone.parameters() if p.requires_grad]
            other_par  = non_backbone_params()
            optimizer  = torch.optim.AdamW([
                {"params": bb_params,                    "lr": hparams["backbone_lr"]},
                {"params": other_par,                    "lr": hparams["head_lr"]},
                {"params": center_criterion.parameters(),"lr": hparams["head_lr"]},
            ], weight_decay=hparams["weight_decay"])
            # T_0=10: first restart cycle spans 10 epochs 
            # T_mult=2: each subsequent cycle doubles in length (10 -> 20 -> 40 ...),
            #           giving periodic LR restarts that help escape sharp minima
            #           during the fine-tuning phase rather than a single monotonic decay.
            steps_per_epoch = len(train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10 * steps_per_epoch, T_mult=2, eta_min=1e-7
            )
            if use_amp:
                scaler = GradScaler("cuda")

        # ── Train epoch ───────────────────────────────────────────────────
        model.train()
        loss_m = focal_m = center_m = reg_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for images, labels_b in tqdm(
                train_loader,
                desc=f"  {experiment_name} F{fold_idx} E{epoch:02d}",
                leave=False):

            images   = images.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            use_mix = (
                np.random.rand() < hparams["mix_prob"]
                and epoch > hparams["warmup_epochs"]
            )
            if use_mix:
                fn = mixup_data if np.random.rand() < 0.5 else cutmix_data
                alpha = hparams["mixup_alpha"] if fn is mixup_data else hparams["cutmix_alpha"]
                images, y_a, y_b, lam = fn(images, labels_b, alpha)

            optimizer.zero_grad(set_to_none=True)

            def _forward():
                logits, emb = model(images, return_embedding=True)
                fl = (mixup_criterion(focal_criterion, logits, y_a, y_b, lam)
                      if use_mix else focal_criterion(logits, labels_b))
                cl_labels = (y_a if lam > 0.5 else y_b) if use_mix else labels_b
                cl = center_criterion(emb.float(), cl_labels)
                rl = model.feature_selector.get_reg_loss() if has_reg else torch.tensor(0.0, device=device)
                return logits, emb, fl, cl, rl, cl_labels

            if use_amp:
                with autocast("cuda"):
                    logits, emb, fl, cl, rl, cl_labels = _forward()
                    total = fl + hparams["center_loss_lambda"] * cl + rl
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, emb, fl, cl, rl, cl_labels = _forward()
                total = fl + hparams["center_loss_lambda"] * cl + rl
                total.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
                optimizer.step()

            if scheduler: scheduler.step()
            ema.update(model)
            center_criterion.update_centers(
                emb.detach().float(), cl_labels,
                lr=hparams["center_lr"],
                class_weights=center_class_weights if hparams["center_class_weights"] else None,
            )

            n = images.size(0)
            loss_m[0].update(total.item(), n); focal_m[0].update(fl.item(), n)
            center_m[0].update(cl.item(), n);  reg_m[0].update(rl.item(), n)
            epoch_logger.update_train_batch(
                logits=logits if not use_mix else None,
                labels=labels_b if not use_mix else None,
                was_mixed=use_mix, grad_norm=gn,
            )

        # ── Validate ──────────────────────────────────────────────────────
        ema_m = ema.get_model(); ema_m.eval()
        all_probs_v, all_labels_v = [], []
        vl_m = vf_m = vc_m = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labs_d = labs.to(device, non_blocking=True)
                ctx    = autocast("cuda") if use_amp else torch.no_grad()
                with ctx if use_amp else torch.no_grad():
                    lv, ev = ema_m(imgs, return_embedding=True)
                    vf = focal_criterion(lv, labs_d)
                    vc = center_criterion(ev.float(), labs_d)
                    vl = vf + hparams["center_loss_lambda"] * vc
                vl_m[0].update(vl.item(), imgs.size(0))
                vf_m[0].update(vf.item(), imgs.size(0))
                vc_m[0].update(vc.item(), imgs.size(0))
                all_probs_v.append(torch.softmax(lv.float(), dim=1).cpu().numpy())
                all_labels_v.append(labs.numpy())

        all_probs_v  = np.concatenate(all_probs_v)
        all_labels_v = np.concatenate(all_labels_v)

        acc, f1_mac = epoch_logger.record_val_metrics(
            all_probs_v, all_labels_v,
            val_loss=vl_m[0].avg, val_focal_loss=vf_m[0].avg,
            val_center_loss=vc_m[0].avg,
        )

        improved = False
        if f1_mac > best_f1:
            best_f1  = f1_mac; best_acc = acc; patience = 0; improved = True
            best_val_probs  = all_probs_v.copy()
            best_val_labels = all_labels_v.copy()
            torch.save({
                "model_state_dict": ema_m.state_dict(),
                "epoch":       epoch,
                "val_acc":     float(acc),
                "val_f1":      float(f1_mac),
                "num_classes": num_classes,
                "class_names": class_names,
            }, fold_dir / "best_model.pth")
        else:
            patience += 1

        epoch_logger.end_epoch(
            model=model, optimizer=optimizer,
            patience_counter=patience, improved=improved,
            best_f1=best_f1, best_acc=best_acc,
            train_loss=loss_m[0].avg,
            train_focal_loss=focal_m[0].avg,
            train_center_loss=center_m[0].avg,
            train_reg_loss=reg_m[0].avg,
        )

        star = " ★" if improved else ""
        if epoch % 5 == 0 or improved or patience >= hparams["patience"]:
            print(f"    E{epoch:02d} | {phase:<10} | loss={loss_m[0].avg:.4f} | "
                  f"acc={acc:.4f} | f1={f1_mac:.4f} | p={patience}{star}")

        if patience >= hparams["patience"]:
            print(f"     Early stopping at epoch {epoch}")
            break

    fold_time = (time.time() - t_start) / 60

    # ── Test evaluation ───────────────────────────────────────────────────
    ckpt = torch.load(fold_dir / "best_model.pth",
                      map_location=device, weights_only=True)
    # Filter training-only keys
    model_keys = set(model.state_dict().keys())
    filtered   = {k: v for k, v in ckpt["model_state_dict"].items()
                  if k in model_keys}
    dropped    = set(ckpt["model_state_dict"].keys()) - model_keys
    if dropped:
        print(f"    Dropped training-only key(s): {dropped}")
    model.load_state_dict(filtered, strict=True)
    model.eval()

    all_probs_te, all_labels_te = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            ctx  = autocast("cuda") if use_amp else torch.no_grad()
            with ctx if use_amp else torch.no_grad():
                lt = model(imgs)
            all_probs_te.append(torch.softmax(lt.float(), dim=1).cpu().numpy())
            all_labels_te.append(labs.numpy())

    all_probs_te  = np.concatenate(all_probs_te)
    all_labels_te = np.concatenate(all_labels_te)
    all_preds_te  = np.argmax(all_probs_te, axis=1)

    acc_te = accuracy_score(all_labels_te, all_preds_te)
    f1_te  = f1_score(all_labels_te, all_preds_te, average="macro", zero_division=0)
    try:
        auc_te = roc_auc_score(all_labels_te, all_probs_te[:, 1])
    except ValueError:
        auc_te = 0.0

    cm = confusion_matrix(all_labels_te, all_preds_te)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sel_params = (
        model.get_selector_params()
        if hasattr(model, "get_selector_params") else 0
    )

    per_image = [
        {
            "filename": te_f[i],
            "true_label": int(all_labels_te[i]),
            "predicted":  int(all_preds_te[i]),
            "correct":    bool(all_preds_te[i] == all_labels_te[i]),
            "confidence": float(all_probs_te[i, all_preds_te[i]]),
        }
        for i in range(len(te_f))
    ]

    fold_result = {
        "fold":              fold_idx,
        "test_accuracy":     float(acc_te),
        "test_f1_macro":     float(f1_te),
        "test_auc":          float(auc_te),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "confusion_matrix":  cm.tolist(),
        "best_epoch":        int(ckpt["epoch"]),
        "best_val_acc":      float(best_acc),
        "best_val_f1":       float(best_f1),
        "training_time_min": round(fold_time, 1),
        "selector_params":   sel_params,
        "test_size":         len(te_f),
        "errors":            sum(1 for p in per_image if not p["correct"]),
    }

    with open(fold_dir / "results.json",   "w") as f: json.dump(fold_result, f, indent=2)
    with open(fold_dir / "per_image.json", "w") as f: json.dump(per_image,   f, indent=2)

    save_fold_artifacts(
        fold_dir=fold_dir,
        epoch_logger=epoch_logger,
        hparams=hparams,
        model=model,
        fold_idx=fold_idx,
        test_labels=all_labels_te,
        test_probs=all_probs_te,
        val_labels=best_val_labels,
        val_probs=best_val_probs,
        data_info={"train_size": len(tr_f), "val_size": len(va_f), "test_size": len(te_f)},
        extra_hparams=extra_hparams or {},
    )

    print(f"\n  ✓ Fold {fold_idx}: acc={100*acc_te:.2f}% | f1={f1_te:.4f} | "
          f"auc={auc_te:.4f} | sens={100*sensitivity:.1f}% | "
          f"spec={100*specificity:.1f}% | errors={fold_result['errors']}/{len(te_f)} | "
          f"{fold_time:.1f}min\n")

    del ema, focal_criterion, center_criterion
    torch.cuda.empty_cache()
    return fold_result
