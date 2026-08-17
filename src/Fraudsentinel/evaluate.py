"""
Evaluation utilities for FraudSentinel.
Computes Precision, Recall, F1-illicit, and PR-AUC for XGBoost and GNN models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import (
    GATClassifier,
    GCNClassifier,
    GraphSAGEClassifier,
    XGBoostFraudClassifier,
)

logger = get_logger("FraudSentinel.Evaluate")


def evaluate_model(
    model: XGBoostFraudClassifier | GraphSAGEClassifier | GCNClassifier | GATClassifier,
    data: Data,
    mask: torch.Tensor,
    split_name: str = "val",
    fixed_threshold: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Evaluates model performance metrics on specified dataset split mask."""
    eval_mask = mask & data.labeled_mask
    n_samples = eval_mask.sum().item()

    if n_samples == 0:
        raise ValueError(f"Mask '{split_name}' contains zero labeled nodes.")

    y_true = data.y[eval_mask].cpu().numpy()

    if isinstance(model, XGBoostFraudClassifier):
        X = data.x[eval_mask].cpu().numpy()
        y_prob = model.predict_proba(X)
    else:
        model.eval()
        device = next(model.parameters()).device
        x_dev = data.x.to(device)
        edge_dev = data.edge_index.to(device)
        with torch.no_grad():
            y_prob_all = model.predict_proba(x_dev, edge_dev)
        y_prob = y_prob_all[eval_mask].cpu().numpy()

    pr_auc = float(average_precision_score(y_true, y_prob))

    precisions_curve, recalls_curve, _ = precision_recall_curve(y_true, y_prob)

    if fixed_threshold is not None:
        best_thresh = fixed_threshold
    else:
        best_thresh, _, _, _ = _find_best_threshold(y_true, y_prob)

    y_pred = (y_prob >= best_thresh).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    n_illicit = int((y_true == 1).sum())
    n_licit = int((y_true == 0).sum())

    logger.info(f"[{split_name.upper()}] Metrics ({n_samples:,} labeled nodes: {n_illicit} illicit, {n_licit} licit)")
    logger.info(f"  PR-AUC          : {pr_auc:.4f}")
    logger.info(f"  Optimal Threshold: {best_thresh:.4f}")
    logger.info(f"  Precision       : {precision:.4f}")
    logger.info(f"  Recall          : {recall:.4f}")
    logger.info(f"  F1-Illicit      : {f1:.4f}")

    return {
        "split": split_name,
        "n_samples": n_samples,
        "n_illicit": n_illicit,
        "n_licit": n_licit,
        "pr_auc": pr_auc,
        "best_threshold": best_thresh,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_prob": y_prob,
        "precisions_curve": precisions_curve,
        "recalls_curve": recalls_curve,
    }


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> tuple[float, float, float, float]:
    """Sweeps 100 probability thresholds to identify max F1 threshold."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresh = float(t)
            best_prec = float(precision_score(y_true, preds, zero_division=0))
            best_rec = float(recall_score(y_true, preds, zero_division=0))

    return best_thresh, best_f1, best_prec, best_rec


def compare_models(
    xgb_metrics: dict,
    gnn_metrics: dict,
    gnn_name: str = "GraphSAGE",
) -> pd.DataFrame:
    """Builds side-by-side DataFrame comparison table."""
    df = pd.DataFrame([
        {
            "Model": "XGBoost (Tabular Baseline)",
            "Split": xgb_metrics["split"],
            "PR-AUC": f"{xgb_metrics['pr_auc']:.4f}",
            "Best Threshold": f"{xgb_metrics['best_threshold']:.4f}",
            "Precision": f"{xgb_metrics['precision']:.4f}",
            "Recall": f"{xgb_metrics['recall']:.4f}",
            "F1-Illicit": f"{xgb_metrics['f1']:.4f}",
        },
        {
            "Model": f"{gnn_name} (GNN)",
            "Split": gnn_metrics["split"],
            "PR-AUC": f"{gnn_metrics['pr_auc']:.4f}",
            "Best Threshold": f"{gnn_metrics['best_threshold']:.4f}",
            "Precision": f"{gnn_metrics['precision']:.4f}",
            "Recall": f"{gnn_metrics['recall']:.4f}",
            "F1-Illicit": f"{gnn_metrics['f1']:.4f}",
        },
    ])
    return df


def plot_pr_curves(
    xgb_metrics: dict,
    gnn_metrics: dict,
    save_path: str | Path,
    gnn_name: str = "GraphSAGE",
) -> None:
    """Plot overlaid Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        xgb_metrics["recalls_curve"],
        xgb_metrics["precisions_curve"],
        label=f"XGBoost (PR-AUC = {xgb_metrics['pr_auc']:.4f})",
        color="navy",
        linewidth=2,
    )
    ax.plot(
        gnn_metrics["recalls_curve"],
        gnn_metrics["precisions_curve"],
        label=f"{gnn_name} (PR-AUC = {gnn_metrics['pr_auc']:.4f})",
        color="darkorange",
        linewidth=2,
    )

    base_rate = xgb_metrics["n_illicit"] / xgb_metrics["n_samples"]
    ax.axhline(
        y=base_rate,
        color="gray",
        linestyle="--",
        label=f"Random Baseline (no-skill = {base_rate:.4f})",
    )

    ax.set_xlabel("Recall (Illicit Class)", fontsize=12)
    ax.set_ylabel("Precision (Illicit Class)", fontsize=12)
    ax.set_title(
        f"Precision-Recall Curve Comparison — {xgb_metrics['split'].upper()} Split",
        fontsize=14,
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PR curve plot to {save_path}")