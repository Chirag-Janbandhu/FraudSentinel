"""
drift_check.py — Feature Drift Detection for FraudSentinel
===========================================================

Implements Population Stability Index (PSI) to formally detect and quantify
concept drift between two temporal node populations within the Elliptic Bitcoin graph.
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

PSI_STABLE = 0.10
PSI_MODERATE = 0.25
PSI_CEILING = 3.00
_EPS = 1e-6


@dataclass
class DriftResult:
    """Container for a single drift-check run result."""

    label: str
    psi_table: pd.DataFrame
    n_ref: int
    n_comp: int
    n_unreliable: int
    n_major: int
    n_moderate: int
    n_stable: int
    mean_psi: float
    max_psi: float
    proba_ref_mean: float | None = None
    proba_comp_mean: float | None = None
    proba_delta: float | None = None
    top_features: list[str] = field(default_factory=list)


def compute_psi(
    ref: np.ndarray,
    comp: np.ndarray,
    n_bins: int = 10,
    feature_name: str | None = None,
) -> float:
    """Compute Population Stability Index (PSI) between two feature distributions."""
    ref = ref.astype(np.float64)
    comp = comp.astype(np.float64)

    if np.std(ref) < _EPS:
        return 0.0

    p_low, p_high = np.percentile(ref, [0.5, 99.5])
    if p_low == p_high or np.isnan(p_low) or np.isnan(p_high):
        return 0.0

    ref_clipped = np.clip(ref, p_low, p_high)
    comp_clipped = np.clip(comp, p_low, p_high)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.nanpercentile(ref_clipped, quantiles))

    use_fixed_width = False

    if len(bin_edges) - 1 < 5:
        use_fixed_width = True
    else:
        edges_temp = bin_edges.copy()
        edges_temp[0] = min(edges_temp[0], comp_clipped.min()) - _EPS
        edges_temp[-1] = max(edges_temp[-1], comp_clipped.max()) + _EPS
        ref_c, _ = np.histogram(ref_clipped, bins=edges_temp)
        comp_c, _ = np.histogram(comp_clipped, bins=edges_temp)

        if (ref_c.max() / max(ref_c.sum(), 1) > 0.90) or (comp_c.max() / max(comp_c.sum(), 1) > 0.90):
            p_low_t, p_high_t = np.percentile(ref, [1.0, 99.0])
            if p_low_t < p_high_t:
                p_low, p_high = p_low_t, p_high_t
                ref_clipped = np.clip(ref, p_low, p_high)
                comp_clipped = np.clip(comp, p_low, p_high)
                bin_edges = np.unique(np.nanpercentile(ref_clipped, quantiles))

            if len(bin_edges) - 1 < 5:
                use_fixed_width = True
            else:
                edges_temp = bin_edges.copy()
                edges_temp[0] = min(edges_temp[0], comp_clipped.min()) - _EPS
                edges_temp[-1] = max(edges_temp[-1], comp_clipped.max()) + _EPS
                ref_c, _ = np.histogram(ref_clipped, bins=edges_temp)
                comp_c, _ = np.histogram(comp_clipped, bins=edges_temp)
                if (ref_c.max() / max(ref_c.sum(), 1) > 0.90) or (comp_c.max() / max(comp_c.sum(), 1) > 0.90):
                    use_fixed_width = True

    if use_fixed_width:
        feat_str = f"feature '{feature_name}'" if feature_name else "feature"
        logger.info(
            f"Quantile binning produced excessive bin concentration for {feat_str}; "
            "falling back explicitly to fixed-width binning."
        )
        bin_edges = np.linspace(p_low, p_high, n_bins + 1)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

    bin_edges[0] = min(bin_edges[0], comp_clipped.min()) - _EPS
    bin_edges[-1] = max(bin_edges[-1], comp_clipped.max()) + _EPS

    ref_counts, _ = np.histogram(ref_clipped, bins=bin_edges)
    comp_counts, _ = np.histogram(comp_clipped, bins=bin_edges)

    n_b = len(ref_counts)
    ref_prop = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * n_b)
    comp_prop = (comp_counts + 0.5) / (comp_counts.sum() + 0.5 * n_b)

    psi = float(np.sum((comp_prop - ref_prop) * np.log(comp_prop / ref_prop)))

    if psi > PSI_CEILING:
        feat_str = f"feature '{feature_name}'" if feature_name else "a feature"
        logger.warning(
            f"Computed PSI of {psi:.4f} for {feat_str} exceeds 3.0; "
            "this likely indicates a binning artifact rather than genuine drift. "
            "Flagged as UNRELIABLE."
        )

    return psi


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
    """Run full drift check across all node features between two population masks."""
    logger.info(f"{'-' * 60}")
    logger.info(f"Drift check: [{label}]")

    X_ref = data.x[ref_mask].cpu().numpy()
    X_comp = data.x[comp_mask].cpu().numpy()

    n_ref, n_features = X_ref.shape
    n_comp = X_comp.shape[0]

    logger.info(
        f"  Reference nodes : {n_ref:,} | "
        f"Comparison nodes: {n_comp:,} | "
        f"Features: {n_features}"
    )

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

    logger.info("  Computing PSI for all features ...")
    psi_scores = []
    for i in range(n_features):
        psi = compute_psi(
            X_ref[:, i],
            X_comp[:, i],
            n_bins=n_bins,
            feature_name=feature_names[i],
        )
        psi_scores.append(psi)

    def _severity(p: float) -> str:
        if p > PSI_CEILING:
            return "UNRELIABLE"
        if p < PSI_STABLE:
            return "stable"
        if p < PSI_MODERATE:
            return "moderate"
        return "MAJOR"

    psi_table = pd.DataFrame({
        "feature": feature_names,
        "psi": psi_scores,
        "severity": [_severity(p) for p in psi_scores],
    }).sort_values("psi", ascending=False).reset_index(drop=True)

    n_unreliable = int((psi_table["severity"] == "UNRELIABLE").sum())
    n_major = int((psi_table["severity"] == "MAJOR").sum())
    n_moderate = int((psi_table["severity"] == "moderate").sum())
    n_stable = int((psi_table["severity"] == "stable").sum())
    mean_psi = float(psi_table["psi"].mean())
    max_psi = float(psi_table["psi"].max())
    top_features = psi_table.head(10)["feature"].tolist()

    logger.info(f"  PSI results - mean: {mean_psi:.4f} | max: {max_psi:.4f}")
    logger.info(
        f"  Feature severity - "
        f"UNRELIABLE (>3.0): {n_unreliable} | "
        f"MAJOR (0.25-3.0): {n_major} | "
        f"Moderate (0.10-0.25): {n_moderate} | "
        f"Stable (<0.10): {n_stable}"
    )
    logger.info("  Top 5 drifted features:")
    for _, row in psi_table.head(5).iterrows():
        logger.info(f"    {row['feature']:40s}  PSI={row['psi']:.4f}  [{row['severity']}]")

    if n_major >= 10:
        verdict = (
            f"MAJOR DRIFT DETECTED: {n_major}/{n_features} features show major "
            f"population shift (0.25 < PSI <= 3.0). Model recalibration required before "
            f"production deployment."
        )
    elif n_major > 0 or n_moderate >= 20:
        verdict = (
            f"MODERATE DRIFT DETECTED: {n_major} major + {n_moderate} moderate "
            f"features shifted. Monitor model performance closely."
        )
    else:
        verdict = (
            f"NO SIGNIFICANT DRIFT: {n_stable}/{n_features} features are stable. "
            f"Feature distributions are consistent with the reference population."
        )
    logger.info(f"  [VERDICT]: {verdict}")

    proba_ref_mean = proba_comp_mean = proba_delta = None
    if proba_ref is not None and proba_comp is not None:
        proba_ref_mean = float(np.mean(proba_ref))
        proba_comp_mean = float(np.mean(proba_comp))
        proba_delta = abs(proba_ref_mean - proba_comp_mean)
        logger.info(
            f"  Predicted probability shift - "
            f"ref mean: {proba_ref_mean:.4f} | "
            f"comp mean: {proba_comp_mean:.4f} | "
            f"delta = {proba_delta:.4f}"
        )

    return DriftResult(
        label=label,
        psi_table=psi_table,
        n_ref=n_ref,
        n_comp=n_comp,
        n_unreliable=n_unreliable,
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
