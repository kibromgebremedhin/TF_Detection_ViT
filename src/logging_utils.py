"""
Training logger compatible with both backbone_experiment.py and selector_ablation.py.
Provides EpochLogger and save_fold_artifacts.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class EpochLogger:
    """
    Tracks per-epoch metrics and writes a human-readable log file.

    Parameters
    ----------
    log_path            : path to write the plain-text log (optional)
    has_reg             : whether the selector has a regularisation loss
    include_gate_weights: reserved for future gate weight logging
    """

    def __init__(
        self,
        log_path=None,
        has_reg: bool = False,
        include_gate_weights: bool = False,
    ):
        self.log_path             = Path(log_path) if log_path else None
        self.has_reg              = has_reg
        self.include_gate_weights = include_gate_weights
        self.epochs               = []
        self._cur                 = {}
        self._train_correct       = 0
        self._train_total         = 0
        self._f                   = None

        if self.log_path:
            self._f = open(self.log_path, "w", buffering=1)
            self._f.write(
                "Epoch | Phase      | TrLoss  | ValLoss | ValAcc  | "
                "ValF1   | ValAUC  | LR        | Patience\n"
            )
            self._f.write("-" * 90 + "\n")

    def start_epoch(self, epoch: int, phase: str) -> None:
        self._cur             = {"epoch": epoch, "phase": phase}
        self._train_correct   = 0
        self._train_total     = 0

    def update_train_batch(
        self,
        logits=None,
        labels=None,
        was_mixed: bool = False,
        grad_norm=None,
    ) -> None:
        if logits is not None and labels is not None and not was_mixed:
            import torch
            with torch.no_grad():
                preds = logits.float().argmax(dim=1)
                self._train_correct += (preds == labels).sum().item()
                self._train_total   += labels.size(0)
        if grad_norm is not None:
            self._cur["grad_norm"] = float(grad_norm)

    def record_val_metrics(
        self,
        all_probs,
        all_labels,
        val_loss: float = 0.0,
        val_focal_loss: float = 0.0,
        val_center_loss: float = 0.0,
    ):
        preds  = np.argmax(all_probs, axis=1)
        acc    = accuracy_score(all_labels, preds)
        f1_mac = f1_score(all_labels, preds, average="macro", zero_division=0)
        try:
            auc = (
                roc_auc_score(all_labels, all_probs[:, 1])
                if all_probs.shape[1] == 2
                else roc_auc_score(all_labels, all_probs, multi_class="ovr")
            )
        except ValueError:
            auc = 0.0

        self._cur.update({
            "val_loss":   val_loss,
            "val_focal":  val_focal_loss,
            "val_center": val_center_loss,
            "val_acc":    acc,
            "val_f1":     f1_mac,
            "val_auc":    auc,
        })
        return acc, f1_mac

    def end_epoch(
        self,
        model,
        optimizer,
        patience_counter: int,
        improved: bool,
        best_f1: float,
        best_acc: float,
        train_loss: float,
        train_focal_loss: float = 0.0,
        train_center_loss: float = 0.0,
        train_reg_loss: float = 0.0,
    ) -> None:
        tr_acc = self._train_correct / max(self._train_total, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        self._cur.update({
            "train_loss":   train_loss,
            "train_focal":  train_focal_loss,
            "train_center": train_center_loss,
            "train_reg":    train_reg_loss,
            "train_acc":    tr_acc,
            "lr":           lr_now,
            "patience":     patience_counter,
            "improved":     improved,
        })
        self.epochs.append(dict(self._cur))

        if self._f:
            e   = self._cur
            line = (
                f"{e['epoch']:5d} | {e['phase']:<10} | "
                f"{train_loss:7.4f} | "
                f"{e.get('val_loss',0):7.4f} | "
                f"{e.get('val_acc',0):7.4f} | "
                f"{e.get('val_f1',0):7.4f} | "
                f"{e.get('val_auc',0):7.4f} | "
                f"{lr_now:9.2e} | {patience_counter:8d}"
            )
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f:
            self._f.close()

    def save_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.epochs, f, indent=2)


def save_fold_artifacts(
    fold_dir,
    epoch_logger: EpochLogger,
    hparams: dict,
    model=None,
    fold_idx: int = 0,
    test_labels=None,
    test_probs=None,
    val_labels=None,
    val_probs=None,
    data_info: dict = None,
    extra_hparams: dict = None,
):
    """
    Save epoch log, hparams, and optional val/test prediction arrays
    to fold_dir.  Called by backbone_experiment.py after each fold.
    """
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Epoch log
    epoch_logger.close()
    epoch_logger.save_json(fold_dir / "epoch_log.json")

    # Hparams
    merged = dict(hparams)
    if extra_hparams:
        merged.update(extra_hparams)
    if data_info:
        merged.update(data_info)
    with open(fold_dir / "hparams.json", "w") as f:
        json.dump(merged, f, indent=2)

    # Val predictions (from best epoch)
    if val_probs is not None and val_labels is not None:
        val_preds = np.argmax(val_probs, axis=1)
        val_acc   = accuracy_score(val_labels, val_preds)
        val_f1    = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_summary = {
            "fold": fold_idx,
            "val_acc":  float(val_acc),
            "val_f1":   float(val_f1),
            "n":        int(len(val_labels)),
        }
        with open(fold_dir / "val_summary.json", "w") as f:
            json.dump(val_summary, f, indent=2)

    # Test probability array (for ROC / calibration analysis)
    if test_probs is not None:
        np.save(fold_dir / "test_probs.npy",  test_probs)
    if test_labels is not None:
        np.save(fold_dir / "test_labels.npy", test_labels)
