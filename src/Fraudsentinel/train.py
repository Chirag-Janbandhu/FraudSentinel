"""
Training pipelines for FraudSentinel models.

train_xgboost(data, cfg)
    Extracts node features for train/val labeled nodes, fits the XGBoost
    baseline, logs val metrics, and saves the model artifact.

train_graphsage(data, cfg)
    Full-batch GNN training loop with StandardScaler feature normalisation,
    BCEWithLogitsLoss + pos_weight for class imbalance, ReduceLROnPlateau
    scheduler, patience-based early stopping, and best-checkpoint saving.

Why full-batch instead of NeighborLoader?
    For 203K nodes x 170 features (~140 MB) the full feature matrix fits
    comfortably in CPU RAM. Full-batch training is:
      - Faster per epoch (no sampling overhead)
      - Deterministic and reproducible
      - Effectively equivalent in convergence for medium-sized graphs
    For graphs with 10M+ nodes, NeighborLoader with pyg-lib / torch-sparse
    is the scale-out path; the same GraphSAGEClassifier architecture works
    unchanged in that setting.

Why feature normalisation?
    The Elliptic raw features are pre-standardised by the dataset authors,
    but the 5 topological features we engineered (in_degree 0-284, pagerank
    ~1e-6, community_size_ratio ~0.01) are on vastly different scales.
    Without normalisation these features dominate the gradient signal and
    prevent the GNN from converging. We fit a StandardScaler on the training
    nodes only and transform all nodes, preventing any val/test leakage.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import XGBoostFraudClassifier, GraphSAGEClassifier, GCNClassifier, GATClassifier

logger = get_logger("FraudSentinel.Train")

# ---------------------------------------------------------------------------
# Default configuration (all hyperparameters in one place, no magic numbers)
# ---------------------------------------------------------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "xgb": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
        "random_state": 42,
    },
    "sage": {
        "hidden_channels": [256, 128],   # 2 layers -- see models.py rationale
        "dropout": 0.3,
        "lr": 3e-4,                       # lower LR for stable full-batch convergence
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,                   # more patience; full-batch is deterministic
        "random_state": 42,
    },
    "gcn": {
        "hidden_channels": [256, 128],
        "dropout": 0.3,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
        "random_state": 42,
    },
    "gat": {
        "hidden_channels": [64, 128],    # 4 heads out first layer (dim=256), concat=True
        "dropout": 0.3,
        "heads": 4,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
        "random_state": 42,
    },
    "model_dir": "models",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _labeled_mask(data: Data, split_mask: torch.Tensor) -> torch.Tensor:
    """Nodes that are in the split AND have a real label (0 or 1, not -1)."""
    return split_mask & data.labeled_mask


def _extract_xy(data: Data, mask: torch.Tensor):
    """Return (X_numpy, y_numpy) for nodes selected by mask."""
    X = data.x[mask].cpu().numpy()
    y = data.y[mask].cpu().numpy()
    return X, y


def _compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    pos_weight = n_licit / n_illicit for BCEWithLogitsLoss.
    Equivalent to XGBoost's scale_pos_weight; compensates for class imbalance
    without resampling, preserving the temporal split integrity.
    """
    n_illicit = int((y_train == 1).sum())
    n_licit   = int((y_train == 0).sum())
    ratio = n_licit / max(n_illicit, 1)
    logger.info(
        f"Class dist | illicit={n_illicit} licit={n_licit} "
        f"| pos_weight={ratio:.2f}"
    )
    return torch.tensor([ratio], dtype=torch.float)


# ---------------------------------------------------------------------------
# XGBoost Training
# ---------------------------------------------------------------------------

def train_xgboost(
    data: Data,
    cfg: Dict[str, Any] | None = None,
    model_dir: str | Path = "models",
) -> XGBoostFraudClassifier:
    """
    Train the XGBoost baseline on labeled training nodes.

    The baseline uses only node features (data.x) — no graph structure.
    Any improvement from GraphSAGE is attributable purely to neighbourhood
    aggregation over the transaction graph.

    Parameters
    ----------
    data      : PyG Data object (output of graph_construction.py)
    cfg       : Optional dict overriding DEFAULT_CFG["xgb"]
    model_dir : Directory to save the trained model artifact

    Returns
    -------
    Fitted XGBoostFraudClassifier
    """
    xgb_cfg  = {**DEFAULT_CFG["xgb"], **(cfg or {})}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING: XGBoost Baseline")
    logger.info("=" * 60)

    train_mask = _labeled_mask(data, data.train_mask)
    val_mask   = _labeled_mask(data, data.val_mask)

    X_train, y_train = _extract_xy(data, train_mask)
    X_val,   y_val   = _extract_xy(data, val_mask)

    logger.info(f"Train nodes: {len(y_train)} | Val nodes: {len(y_val)}")

    t0  = time.time()
    clf = XGBoostFraudClassifier(**xgb_cfg)
    clf.fit(X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0

    val_proba = clf.predict_proba(X_val)
    val_pred  = (val_proba >= 0.5).astype(int)
    val_f1    = f1_score(y_val, val_pred, pos_label=1, zero_division=0)
    logger.info(
        f"XGBoost trained in {elapsed:.1f}s | "
        f"Val F1-illicit (thr=0.5): {val_f1:.4f}"
    )

    clf.save(model_dir / "xgboost_baseline.json")
    return clf


# ---------------------------------------------------------------------------
# GraphSAGE Training (full-batch, with feature normalisation)
# ---------------------------------------------------------------------------

def train_graphsage(
    data: Data,
    cfg: Dict[str, Any] | None = None,
    model_dir: str | Path = "models",
    device: str | None = None,
    use_pseudo: bool = False,
) -> GraphSAGEClassifier:
    """
    Train GraphSAGE via full-batch gradient descent.

    Feature normalisation
    ---------------------
    A StandardScaler is fitted on training-node features only, then applied
    to all nodes. This prevents the engineered topological features (which
    span very different ranges: in_degree 0-284, pagerank ~1e-6) from
    drowning the gradient signal from the pre-standardised raw features.
    The scaler is saved to model_dir/feature_scaler.pkl for inference.

    Training loop
    -------------
    - Each epoch: full graph forward pass -> BCE loss on train nodes only
    - Optimiser : Adam + weight decay
    - Scheduler : ReduceLROnPlateau on val F1 (factor=0.5, patience=5)
    - Stopping  : patience=10 epochs on val F1-illicit
    - Checkpoint: best val-F1 weights -> model_dir/graphsage_best.pt

    Parameters
    ----------
    data      : PyG Data object
    cfg       : Optional dict overriding DEFAULT_CFG["sage"]
    model_dir : Directory for checkpoints and scaler
    device    : "cpu" | "cuda" | None (auto-detect)

    Returns
    -------
    GraphSAGEClassifier with best-val-F1 weights loaded
    """
    sage_cfg  = {**DEFAULT_CFG["sage"], **(cfg or {})}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Training device: {dev}")

    torch.manual_seed(sage_cfg["random_state"])
    np.random.seed(sage_cfg["random_state"])

    logger.info("=" * 60)
    logger.info("TRAINING: GraphSAGE (full-batch)")
    logger.info("=" * 60)

    # -- Masks (labeled nodes only) ----------------------------------------
    if use_pseudo:
        assert hasattr(data, "pseudo_y"), "Data object must contain pseudo_y for pseudo-label training."
        train_mask = (data.time_step <= 34) & (data.pseudo_y != -1)
        y_train_np = data.pseudo_y[train_mask].cpu().numpy()
    else:
        train_mask = _labeled_mask(data, data.train_mask)
        _, y_train_np = _extract_xy(data, train_mask)

    val_mask   = _labeled_mask(data, data.val_mask)
    pos_weight = _compute_pos_weight(y_train_np).to(dev)

    # -- Feature normalisation (topological features only) ----------------
    # IMPORTANT: The 165 raw Elliptic features are already standardised by
    # the dataset authors. We must NOT re-standardise them -- that would
    # collapse their variance and destroy the model's input signal.
    # We only normalise the 5 topological features we engineered ourselves
    # (columns 165-169: in_degree, out_degree, total_degree, pagerank,
    # community_size_ratio) because these span wildly different scales
    # (in_degree 0-284 vs pagerank ~1e-6).
    logger.info("Fitting StandardScaler on topological features (cols 165+) only...")
    TOPO_START = 165   # first topological feature column index
    X_all_np     = data.x.cpu().numpy().copy()
    train_mask_np = train_mask.cpu().numpy()

    scaler = StandardScaler()
    scaler.fit(X_all_np[train_mask_np, TOPO_START:])
    X_all_np[:, TOPO_START:] = scaler.transform(X_all_np[:, TOPO_START:])
    # Cols 0-164 (raw features) are untouched -- already normalised.

    # Save scaler for reproducible inference
    scaler_path = model_dir / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Topological feature scaler saved to {scaler_path}")

    # -- Move scaled graph to device ---------------------------------------
    x          = torch.tensor(X_all_np, dtype=torch.float).to(dev)
    edge_index = data.edge_index.to(dev)
    if use_pseudo:
        y_train = data.pseudo_y[train_mask].float().to(dev)
    else:
        y_train = data.y[train_mask].float().to(dev)
    y_val      = data.y[val_mask].float().to(dev)

    logger.info(
        f"Full-batch | train_nodes={train_mask.sum().item()} | "
        f"val_nodes={val_mask.sum().item()}"
    )

    # -- Model, optimiser, loss -------------------------------------------
    in_channels = data.num_node_features
    model = GraphSAGEClassifier(
        in_channels=in_channels,
        hidden_channels=sage_cfg["hidden_channels"],
        dropout=sage_cfg["dropout"],
        aggr=sage_cfg.get("aggr", "max"),
    ).to(dev)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=sage_cfg["lr"],
        weight_decay=sage_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- Training loop ----------------------------------------------------
    best_val_f1    = -1.0
    patience_count = 0
    best_ckpt_path = model_dir / "graphsage_best.pt"
    t0             = time.time()

    for epoch in range(1, sage_cfg["epochs"] + 1):
        model.train()
        optimiser.zero_grad()

        # Full graph forward -- all 203K nodes, all edges
        # Loss computed ONLY on labeled training nodes
        all_logits   = model(x, edge_index).squeeze(-1)   # [N]
        train_logits = all_logits[train_mask]

        loss = criterion(train_logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        # Validation
        val_f1 = _evaluate_f1_fullbatch(model, x, edge_index, val_mask, y_val)
        scheduler.step(val_f1)

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{sage_cfg['epochs']} | "
            f"loss={loss.item():.4f} | val_F1={val_f1:.4f} | lr={current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"  [BEST] val F1={best_val_f1:.4f} -- checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= sage_cfg["patience"]:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no val F1 improvement for {sage_cfg['patience']} epochs)"
                )
                break

    # Load best checkpoint
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=dev, weights_only=True)
    )
    elapsed = time.time() - t0
    logger.info(
        f"GraphSAGE training complete in {elapsed:.1f}s | "
        f"Best val F1={best_val_f1:.4f}"
    )

    # Attach scaler to model object for convenient downstream inference
    model.scaler = scaler
    return model


# ---------------------------------------------------------------------------
# GCN Training
# ---------------------------------------------------------------------------

def train_gcn(
    data: Data,
    cfg: Dict[str, Any] | None = None,
    model_dir: str | Path = "models",
    device: str | None = None,
    use_pseudo: bool = False,
) -> GCNClassifier:
    """
    Train GCN via full-batch gradient descent.
    """
    gcn_cfg  = {**DEFAULT_CFG["gcn"], **(cfg or {})}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Training device: {dev}")

    torch.manual_seed(gcn_cfg["random_state"])
    np.random.seed(gcn_cfg["random_state"])

    logger.info("=" * 60)
    logger.info("TRAINING: GCN (full-batch)")
    logger.info("=" * 60)

    if use_pseudo:
        assert hasattr(data, "pseudo_y"), "Data object must contain pseudo_y for pseudo-label training."
        train_mask = (data.time_step <= 34) & (data.pseudo_y != -1)
        y_train_np = data.pseudo_y[train_mask].cpu().numpy()
    else:
        train_mask = _labeled_mask(data, data.train_mask)
        _, y_train_np = _extract_xy(data, train_mask)

    val_mask   = _labeled_mask(data, data.val_mask)
    pos_weight = _compute_pos_weight(y_train_np).to(dev)

    # Use same scaling logic
    TOPO_START = 165
    X_all_np     = data.x.cpu().numpy().copy()
    train_mask_np = train_mask.cpu().numpy()

    # Load pre-fitted scaler from models/feature_scaler.pkl if available, else fit
    scaler_path = model_dir / "feature_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded existing feature scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_all_np[train_mask_np, TOPO_START:])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Fitted and saved new feature scaler to {scaler_path}")

    X_all_np[:, TOPO_START:] = scaler.transform(X_all_np[:, TOPO_START:])

    x          = torch.tensor(X_all_np, dtype=torch.float).to(dev)
    edge_index = data.edge_index.to(dev)
    if use_pseudo:
        y_train = data.pseudo_y[train_mask].float().to(dev)
    else:
        y_train = data.y[train_mask].float().to(dev)
    y_val      = data.y[val_mask].float().to(dev)

    in_channels = data.num_node_features
    model = GCNClassifier(
        in_channels=in_channels,
        hidden_channels=gcn_cfg["hidden_channels"],
        dropout=gcn_cfg["dropout"],
    ).to(dev)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=gcn_cfg["lr"],
        weight_decay=gcn_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1    = -1.0
    patience_count = 0
    best_ckpt_path = model_dir / "gcn_best.pt"
    t0             = time.time()

    for epoch in range(1, gcn_cfg["epochs"] + 1):
        model.train()
        optimiser.zero_grad()

        all_logits   = model(x, edge_index).squeeze(-1)
        train_logits = all_logits[train_mask]

        loss = criterion(train_logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        val_f1 = _evaluate_f1_fullbatch(model, x, edge_index, val_mask, y_val)
        scheduler.step(val_f1)

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{gcn_cfg['epochs']} | "
            f"loss={loss.item():.4f} | val_F1={val_f1:.4f} | lr={current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"  [BEST] val F1={best_val_f1:.4f} -- checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= gcn_cfg["patience"]:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no val F1 improvement for {gcn_cfg['patience']} epochs)"
                )
                break

    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=dev, weights_only=True)
    )
    elapsed = time.time() - t0
    logger.info(
        f"GCN training complete in {elapsed:.1f}s | "
        f"Best val F1={best_val_f1:.4f}"
    )

    model.scaler = scaler
    return model


# ---------------------------------------------------------------------------
# GAT Training
# ---------------------------------------------------------------------------

def train_gat(
    data: Data,
    cfg: Dict[str, Any] | None = None,
    model_dir: str | Path = "models",
    device: str | None = None,
    use_pseudo: bool = False,
) -> GATClassifier:
    """
    Train GAT via full-batch gradient descent.
    """
    gat_cfg  = {**DEFAULT_CFG["gat"], **(cfg or {})}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Training device: {dev}")

    torch.manual_seed(gat_cfg["random_state"])
    np.random.seed(gat_cfg["random_state"])

    logger.info("=" * 60)
    logger.info("TRAINING: GAT (full-batch)")
    logger.info("=" * 60)

    if use_pseudo:
        assert hasattr(data, "pseudo_y"), "Data object must contain pseudo_y for pseudo-label training."
        train_mask = (data.time_step <= 34) & (data.pseudo_y != -1)
        y_train_np = data.pseudo_y[train_mask].cpu().numpy()
    else:
        train_mask = _labeled_mask(data, data.train_mask)
        _, y_train_np = _extract_xy(data, train_mask)

    val_mask   = _labeled_mask(data, data.val_mask)
    pos_weight = _compute_pos_weight(y_train_np).to(dev)

    # Use same scaling logic
    TOPO_START = 165
    X_all_np     = data.x.cpu().numpy().copy()
    train_mask_np = train_mask.cpu().numpy()

    scaler_path = model_dir / "feature_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded existing feature scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_all_np[train_mask_np, TOPO_START:])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Fitted and saved new feature scaler to {scaler_path}")

    X_all_np[:, TOPO_START:] = scaler.transform(X_all_np[:, TOPO_START:])

    x          = torch.tensor(X_all_np, dtype=torch.float).to(dev)
    edge_index = data.edge_index.to(dev)
    if use_pseudo:
        y_train = data.pseudo_y[train_mask].float().to(dev)
    else:
        y_train = data.y[train_mask].float().to(dev)
    y_val      = data.y[val_mask].float().to(dev)

    in_channels = data.num_node_features
    model = GATClassifier(
        in_channels=in_channels,
        hidden_channels=gat_cfg["hidden_channels"],
        dropout=gat_cfg["dropout"],
        heads=gat_cfg["heads"],
    ).to(dev)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=gat_cfg["lr"],
        weight_decay=gat_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1    = -1.0
    patience_count = 0
    best_ckpt_path = model_dir / "gat_best.pt"
    t0             = time.time()

    for epoch in range(1, gat_cfg["epochs"] + 1):
        model.train()
        optimiser.zero_grad()

        all_logits   = model(x, edge_index).squeeze(-1)
        train_logits = all_logits[train_mask]

        loss = criterion(train_logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        val_f1 = _evaluate_f1_fullbatch(model, x, edge_index, val_mask, y_val)
        scheduler.step(val_f1)

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{gat_cfg['epochs']} | "
            f"loss={loss.item():.4f} | val_F1={val_f1:.4f} | lr={current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"  [BEST] val F1={best_val_f1:.4f} -- checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= gat_cfg["patience"]:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no val F1 improvement for {gat_cfg['patience']} epochs)"
                )
                break

    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=dev, weights_only=True)
    )
    elapsed = time.time() - t0
    logger.info(
        f"GAT training complete in {elapsed:.1f}s | "
        f"Best val F1={best_val_f1:.4f}"
    )

    model.scaler = scaler
    return model


# ---------------------------------------------------------------------------
# Internal full-batch val F1 helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_f1_fullbatch(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    y_true_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Full-graph forward pass -> F1 on masked nodes.
    Only nodes selected by `mask` contribute to the metric.
    """
    model.eval()
    all_logits = model(x, edge_index).squeeze(-1)
    proba  = torch.sigmoid(all_logits[mask]).cpu().numpy()
    preds  = (proba >= threshold).astype(int)
    labels = y_true_tensor.cpu().numpy().astype(int)
    return f1_score(labels, preds, pos_label=1, zero_division=0)