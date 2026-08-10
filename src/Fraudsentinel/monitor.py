"""
monitor.py — Lightweight Prediction Logging & Probability Shift Monitoring
===========================================================================

Provides lightweight inference-time monitoring utilities for FraudSentinel:
  1. log_prediction_batch — logs batch statistics and appends a row to CSV.
  2. check_probability_shift — compares batch mean probability against a
     reference baseline and triggers a warning if shift exceeds threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score

from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.Monitor")

DEFAULT_LOG_PATH = Path("reports/prediction_log.csv")


def _to_numpy(arr: np.ndarray | torch.Tensor | Sequence) -> np.ndarray:
    """Convert input array or torch.Tensor to 1D numpy float array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return arr


def log_prediction_batch(
    probas: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor | None = None,
    batch_id: str | None = None,
    eval_threshold: float = 0.5,
    save_path: str | Path = DEFAULT_LOG_PATH,
) -> dict[str, float | str | int]:
    """
    Log summary statistics for a batch of predicted probabilities and append to CSV.

    Parameters
    ----------
    probas         : Array or Tensor of predicted probabilities (class=1).
    y_true         : Optional ground-truth labels for metric calculation.
    batch_id       : Optional identifier/tag for the batch.
    eval_threshold : Threshold used for calculating precision & recall if y_true is given.
    save_path      : Path to CSV log file.

    Returns
    -------
    dict with batch summary statistics.
    """
    p = _to_numpy(probas)
    batch_size = len(p)
    if batch_size == 0:
        raise ValueError("Cannot log empty prediction batch.")

    mean_p = float(np.mean(p))
    std_p = float(np.std(p))
    min_p = float(np.min(p))
    max_p = float(np.max(p))

    prec: float | None = None
    rec: float | None = None

    if y_true is not None:
        y = _to_numpy(y_true)
        # Filter out unlabeled/unknown (-1) nodes if present
        valid_mask = y >= 0
        if np.any(valid_mask):
            y_valid = y[valid_mask].astype(int)
            p_valid = p[valid_mask]
            preds = (p_valid >= eval_threshold).astype(int)
            prec = float(precision_score(y_valid, preds, zero_division=0))
            rec = float(recall_score(y_valid, preds, zero_division=0))

    timestamp = datetime.now(timezone.utc).isoformat()
    b_id = batch_id or f"batch_{timestamp[:19]}"

    log_msg = (
        f"Prediction Batch [{b_id}] - N={batch_size:,} | "
        f"Mean Prob={mean_p:.4f} (std={std_p:.4f}, min={min_p:.4f}, max={max_p:.4f})"
    )
    if prec is not None and rec is not None:
        log_msg += f" | Precision@{eval_threshold}={prec:.4f}, Recall@{eval_threshold}={rec:.4f}"
    logger.info(log_msg)

    # Append row to CSV report
    row_data = {
        "timestamp": timestamp,
        "batch_id": b_id,
        "batch_size": batch_size,
        "mean_prob": mean_p,
        "std_prob": std_p,
        "min_prob": min_p,
        "max_prob": max_p,
        "precision": prec if prec is not None else np.nan,
        "recall": rec if rec is not None else np.nan,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row_data])

    if not save_path.exists():
        df_row.to_csv(save_path, index=False)
    else:
        df_row.to_csv(save_path, mode="a", header=False, index=False)

    return row_data


def check_probability_shift(
    current_probas: np.ndarray | torch.Tensor,
    reference_probas: np.ndarray | torch.Tensor | float,
    threshold: float = 0.05,
) -> bool:
    """
    Compare current batch mean predicted probability against reference baseline.

    Parameters
    ----------
    current_probas   : Current batch predicted probabilities.
    reference_probas : Reference predicted probabilities OR pre-calculated baseline mean.
    threshold        : Absolute difference threshold for triggering a warning.

    Returns
    -------
    bool : True if shift exceeds threshold (drift detected), False otherwise.
    """
    curr = _to_numpy(current_probas)
    curr_mean = float(np.mean(curr))

    if isinstance(reference_probas, (float, int, np.floating)):
        ref_mean = float(reference_probas)
    else:
        ref_mean = float(np.mean(_to_numpy(reference_probas)))

    diff = abs(curr_mean - ref_mean)
    is_shifted = diff > threshold

    if is_shifted:
        logger.warning(
            f"[PROBABILITY DRIFT WARNING] Absolute output probability shift detected: "
            f"Reference Mean = {ref_mean:.4f} vs Current Mean = {curr_mean:.4f} "
            f"(Delta = {diff:.4f} > Threshold = {threshold:.4f}). "
            f"Classification threshold recalibration recommended!"
        )
    else:
        logger.info(
            f"[PROBABILITY STABLE] Output probability within normal limits: "
            f"Reference Mean = {ref_mean:.4f} vs Current Mean = {curr_mean:.4f} "
            f"(Delta = {diff:.4f} <= Threshold = {threshold:.4f})."
        )

    return is_shifted
