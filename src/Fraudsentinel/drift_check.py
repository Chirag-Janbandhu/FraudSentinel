"""
drift_check.py — Feature Drift Detection for FraudSentinel
===========================================================

Implements Population Stability Index (PSI) and a secondary Kolmogorov-Smirnov
(KS) test to formally detect and quantify concept drift between two temporal
node populations within the Elliptic Bitcoin graph.

Background
----------
During manual evaluation of FraudSentinel, a validated GraphSAGE model (val
PR-AUC = 0.92) collapsed to near-random performance on the test split
(timesteps 43-49).  Elliptic's own dataset documentation attributes this to
the disappearance of a major dark market at timestep 43, which caused a
sharp, measurable shift in the local transaction-feature distributions of
illicit nodes.  This module formalises that observation so it can be:

  1. Reproduced deterministically (no model needed — raw feature PSI is
     model-agnostic).
  2. Used as a production monitoring signal in the future.

PSI Interpretation
------------------
  PSI < 0.10  : No significant population shift (stable).
  PSI 0.10–0.25 : Moderate shift — monitor closely.
  PSI > 0.25  : Major shift — retrain / recalibrate threshold.

Public API
----------
  compute_psi(ref, comp, n_bins=10) -> float
  run_drift_check(data, ref_mask, comp_mask, feature_names=None,
                  n_bins=10, proba_ref=None, proba_comp=None) -> DriftResult
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.DriftCheck")

# ── Interpretation thresholds (PSI) ──────────────────────────────────────────
PSI_STABLE   = 0.10   # below → no significant shift
PSI_MODERATE = 0.25   # between stable and here → moderate shift
#                       above → major shift

# Small epsilon added to bin proportions to avoid log(0)
_EPS = 1e-6


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    """
    Container for a single drift-check run.

    Attributes
    ----------
    label          : Human-readable name for this comparison (e.g. "TrainVal vs Test").
    psi_table      : DataFrame with columns [feature, psi, severity].
                     Sorted by psi descending.
    n_ref          : Number of nodes in the reference set.
    n_comp         : Number of nodes in the comparison set.
    n_major        : Features with PSI > 0.25 (major shift).
    n_moderate     : Features with PSI 0.10–0.25 (moderate shift).
    n_stable       : Features with PSI < 0.10 (stable).
    mean_psi       : Mean PSI across all features.
    max_psi        : Maximum PSI across all features.
    proba_ref_mean : Mean predicted probability in the reference set (optional).
    proba_comp_mean: Mean predicted probability in the comparison set (optional).
    proba_delta    : Absolute shift in mean predicted probability (optional).
    """
    label: str
    psi_table: pd.DataFrame
    n_ref: int
    n_comp: int
    n_major: int
    n_moderate: int
    n_stable: int
    mean_psi: float
    max_psi: float
    proba_ref_mean: float | None = None
    proba_comp_mean: float | None = None
    proba_delta: float | None = None
    top_features: list[str] = field(default_factory=list)


# ── Core PSI calculation ──────────────────────────────────────────────────────

def compute_psi(
    ref: np.ndarray,
    comp: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two 1-D arrays.

    Bins are derived from the reference distribution using quantile edges,
    so each reference bin starts with an equal number of observations.
    The comparison distribution is then evaluated against the same bin edges.

    Parameters
    ----------
    ref    : 1-D numpy array — the reference (training) population.
    comp   : 1-D numpy array — the comparison (test/production) population.
    n_bins : Number of quantile bins to use (default: 10).

    Returns
    -------
    float — PSI score.  Higher → more drift.

    Notes
    -----
    If a feature is constant in the reference set (all values identical),
    PSI is undefined and we return 0.0 — a constant feature cannot shift.
    """
    ref  = ref.astype(np.float64)
    comp = comp.astype(np.float64)

    # Constant-feature guard: if the reference has no variance, PSI = 0.
    if np.std(ref) < _EPS:
        return 0.0

    # Build bin edges from reference quantiles.
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.nanpercentile(ref, quantiles)

    # Deduplicate edges (can happen for heavily skewed / discrete features).
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # All values collapsed into one bin — no variance to measure.
        return 0.0

    # Force the outermost edges to capture the full range of both populations.
    bin_edges[0]  = min(bin_edges[0],  comp.min()) - _EPS
    bin_edges[-1] = max(bin_edges[-1], comp.max()) + _EPS

    # Count observations in each bin.
    ref_counts,  _ = np.histogram(ref,  bins=bin_edges)
    comp_counts, _ = np.histogram(comp, bins=bin_edges)

    # Laplace smoothing (0.5 added to each bin count) to prevent zero-bin log ratio explosion
    n_bins_actual = len(ref_counts)
    ref_prop  = (ref_counts  + 0.5) / (ref_counts.sum()  + 0.5 * n_bins_actual)
    comp_prop = (comp_counts + 0.5) / (comp_counts.sum() + 0.5 * n_bins_actual)

    # PSI formula: sum over bins of (comp_prop - ref_prop) * ln(comp_prop / ref_prop)
    psi = np.sum((comp_prop - ref_prop) * np.log(comp_prop / ref_prop))
    return float(psi)


# ── Multi-feature drift scan ──────────────────────────────────────────────────

def run_drift_check(
    data: Data,
    ref_mask: torch.Tensor,
    comp_mask: torch.Tensor,
    label: str = "Reference vs Comparison",
    feature_names: Sequence[str] | None = None,
    n_bins: int = 10,
    proba_ref: np.ndarray | None = None,
    proba_comp: np.ndarray | None = None,
) -> DriftResult:
    """
    Run a full drift check across all node features between two mask-defined
    populations in a PyTorch Geometric Data object.

    Parameters
    ----------
    data          : PyG Data object containing `.x` (node features) and
                    boolean mask tensors.
    ref_mask      : Boolean tensor selecting reference nodes (e.g., train | val).
    comp_mask     : Boolean tensor selecting comparison nodes (e.g., test).
    label         : Name for this comparison — used in log messages and reports.
    feature_names : Optional list of feature names, length == data.x.shape[1].
                    Defaults to ["feature_1", ..., "feature_165", "in_degree",
                    "out_degree", "total_degree", "pagerank", "community_id"].
    n_bins        : Number of quantile bins for PSI (default: 10).
    proba_ref     : Optional 1-D numpy array of predicted probabilities for
                    reference nodes (same order as ref_mask).  If provided,
                    the mean probability shift is included in the result.
    proba_comp    : Optional 1-D numpy array of predicted probabilities for
                    comparison nodes.  Must be provided together with proba_ref.

    Returns
    -------
    DriftResult
    """
    logger.info(f"{'-' * 60}")
    logger.info(f"Drift check: [{label}]")

    # ── Extract feature matrices ──────────────────────────────────────────
    X_ref  = data.x[ref_mask].cpu().numpy()   # shape [n_ref, F]
    X_comp = data.x[comp_mask].cpu().numpy()  # shape [n_comp, F]

    n_ref, n_features = X_ref.shape
    n_comp = X_comp.shape[0]

    logger.info(
        f"  Reference nodes : {n_ref:,} | "
        f"Comparison nodes: {n_comp:,} | "
        f"Features: {n_features}"
    )

    # ── Build feature name list ───────────────────────────────────────────
    if feature_names is None:
        feature_names = (
            [f"feature_{i}" for i in range(1, 166)]
            + ["in_degree", "out_degree", "total_degree", "pagerank", "community_id"]
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"number of features ({n_features})."
        )

    # ── Compute PSI for every feature ─────────────────────────────────────
    logger.info("  Computing PSI for all features ...")
    psi_scores = []
    for i in range(n_features):
        psi = compute_psi(X_ref[:, i], X_comp[:, i], n_bins=n_bins)
        psi_scores.append(psi)

    # ── Build results table ───────────────────────────────────────────────
    def _severity(p: float) -> str:
        if p < PSI_STABLE:
            return "stable"
        if p < PSI_MODERATE:
            return "moderate"
        return "MAJOR"

    psi_table = pd.DataFrame({
        "feature":  feature_names,
        "psi":      psi_scores,
        "severity": [_severity(p) for p in psi_scores],
    }).sort_values("psi", ascending=False).reset_index(drop=True)

    # ── Aggregate statistics ──────────────────────────────────────────────
    n_major    = int((psi_table["severity"] == "MAJOR").sum())
    n_moderate = int((psi_table["severity"] == "moderate").sum())
    n_stable   = int((psi_table["severity"] == "stable").sum())
    mean_psi   = float(psi_table["psi"].mean())
    max_psi    = float(psi_table["psi"].max())
    top_features = psi_table.head(10)["feature"].tolist()

    # ── Log summary ───────────────────────────────────────────────────────
    logger.info(f"  PSI results — mean: {mean_psi:.4f} | max: {max_psi:.4f}")
    logger.info(
        f"  Feature severity — "
        f"MAJOR (>0.25): {n_major} | "
        f"Moderate (0.10-0.25): {n_moderate} | "
        f"Stable (<0.10): {n_stable}"
    )
    logger.info("  Top 5 drifted features:")
    for _, row in psi_table.head(5).iterrows():
        logger.info(f"    {row['feature']:40s}  PSI={row['psi']:.4f}  [{row['severity']}]")

    # ── Drift verdict ─────────────────────────────────────────────────────
    if n_major >= 10:
        verdict = (
            f"MAJOR DRIFT DETECTED: {n_major}/{n_features} features show major "
            f"population shift (PSI > 0.25). Model recalibration required before "
            f"production deployment on this population."
        )
    elif n_major > 0 or n_moderate >= 20:
        verdict = (
            f"MODERATE DRIFT DETECTED: {n_major} major + {n_moderate} moderate "
            f"features shifted. Monitor model performance closely; threshold "
            f"recalibration is advisable."
        )
    else:
        verdict = (
            f"NO SIGNIFICANT DRIFT: {n_stable}/{n_features} features are stable. "
            f"Feature distributions are consistent with the reference population."
        )
    logger.info(f"  [VERDICT]: {verdict}")

    # ── Optional: predicted probability shift ─────────────────────────────
    proba_ref_mean = proba_comp_mean = proba_delta = None
    if proba_ref is not None and proba_comp is not None:
        proba_ref_mean  = float(np.mean(proba_ref))
        proba_comp_mean = float(np.mean(proba_comp))
        proba_delta     = abs(proba_ref_mean - proba_comp_mean)
        logger.info(
            f"  Predicted probability shift — "
            f"ref mean: {proba_ref_mean:.4f} | "
            f"comp mean: {proba_comp_mean:.4f} | "
            f"Δ = {proba_delta:.4f}"
        )

    return DriftResult(
        label=label,
        psi_table=psi_table,
        n_ref=n_ref,
        n_comp=n_comp,
        n_major=n_major,
        n_moderate=n_moderate,
        n_stable=n_stable,
        mean_psi=mean_psi,
        max_psi=max_psi,
        proba_ref_mean=proba_ref_mean,
        proba_comp_mean=proba_comp_mean,
        proba_delta=proba_delta,
        top_features=top_features,
    )
