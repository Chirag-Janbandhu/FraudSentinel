"""
Training pipelines for FraudSentinel models (XGBoost, GraphSAGE, GCN, GAT).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import (
    GATClassifier,
    GCNClassifier,
    GraphSAGEClassifier,
    XGBoostFraudClassifier,
)

logger = get_logger("FraudSentinel.Train")

DEFAULT_CFG: dict[str, Any] = {
    "xgb": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
        "save_path": "models/xgboost_baseline.json",
    },
    "sage": {
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "aggr": "max",
        "lr": 0.001,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
        "save_path": "models/graphsage_best.pt",
        "scaler_path": "models/scaler.pkl",
    },
    "gcn": {
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
        "save_path": "models/gcn_best.pt",
    },
    "gat": {
        "hidden_channels": 64,
        "num_layers": 2,
        "heads": 8,
        "dropout": 0.2,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
        "save_path": "models/gat_best.pt",
    },
}


def fit_and_scale_features(
    data: Data,
    scaler_save_path: str | Path | None = None,
) -> tuple[Data, StandardScaler]:
    """Fits StandardScaler on training split features only and applies transformation."""
    data_scaled = data.clone()

    train_labeled = data.train_mask & data.labeled_mask
    scaler = StandardScaler()
    scaler.fit(data.x[train_labeled].cpu().numpy())

    X_scaled_np = scaler.transform(data.x.cpu().numpy()).astype(np.float32)
    data_scaled.x = torch.tensor(X_scaled_np, dtype=torch.float)

    if scaler_save_path is not None:
        path = Path(scaler_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved fitted StandardScaler to {path}")

    return data_scaled, scaler


def load_scaler(scaler_path: str | Path) -> StandardScaler:
    """Loads saved StandardScaler from pickle file."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def train_xgboost(
    data: Data,
    cfg: dict[str, Any] | None = None,
) -> XGBoostFraudClassifier:
    """Trains XGBoost baseline classifier on tabular node features."""
    if cfg is None:
        cfg = DEFAULT_CFG["xgb"]

    logger.info("Setting up XGBoost baseline training...")

    train_mask = data.train_mask & data.labeled_mask
    val_mask = data.val_mask & data.labeled_mask

    X_train = data.x[train_mask].cpu().numpy()
    y_train = data.y[train_mask].cpu().numpy()

    X_val = data.x[val_mask].cpu().numpy()
    y_val = data.y[val_mask].cpu().numpy()

    model = XGBoostFraudClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        use_class_weights=True,
    )

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=cfg.get("early_stopping_rounds", 20),
    )
    elapsed = time.time() - t0
    logger.info(f"XGBoost baseline trained in {elapsed:.2f}s")

    save_path = cfg.get("save_path")
    if save_path:
        model.save(save_path)

    return model


def train_graphsage(
    data: Data,
    cfg: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    pseudo_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[GraphSAGEClassifier, StandardScaler, dict[str, list[float]]]:
    """Trains GraphSAGE GNN classifier with full-batch message passing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg is None:
        cfg = DEFAULT_CFG["sage"]

    logger.info(f"Setting up GraphSAGE training on device [{device}]...")

    scaler_path = cfg.get("scaler_path", "models/scaler.pkl")
    data_scaled, scaler = fit_and_scale_features(data, scaler_save_path=scaler_path)

    data_scaled = data_scaled.to(device)

    train_mask = data_scaled.train_mask & data_scaled.labeled_mask
    val_mask = data_scaled.val_mask & data_scaled.labeled_mask

    y_train_effective = data_scaled.y.clone()
    if pseudo_labels is not None:
        pseudo_indices, pseudo_y = pseudo_labels
        pseudo_indices = pseudo_indices.to(device)
        pseudo_y = pseudo_y.to(device)
        y_train_effective[pseudo_indices] = pseudo_y
        train_mask = train_mask | torch.zeros_like(train_mask).scatter_(0, pseudo_indices, True)
        logger.info(f"Incorporated {len(pseudo_indices):,} pseudo-labels into GraphSAGE training.")

    n_illicit = int((y_train_effective[train_mask] == 1).sum().item())
    n_licit = int((y_train_effective[train_mask] == 0).sum().item())

    if n_illicit > 0:
        pos_weight_val = float(n_licit) / float(n_illicit)
        logger.info(
            f"GraphSAGE pos_weight = {pos_weight_val:.2f} "
            f"({n_licit} licit / {n_illicit} illicit in train split)"
        )
    else:
        pos_weight_val = 1.0

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GraphSAGEClassifier(
        in_channels=data_scaled.num_node_features,
        hidden_channels=cfg.get("hidden_channels", 128),
        out_channels=1,
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        aggr=cfg.get("aggr", "max"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 15)
    save_path = Path(cfg.get("save_path", "models/graphsage_best.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data_scaled.x, data_scaled.edge_index)
        loss = criterion(logits[train_mask], y_train_effective[train_mask].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_loss = criterion(val_logits, data_scaled.y[val_mask].float()).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        y_val_np = data_scaled.y[val_mask].cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val_np, val_preds, zero_division=0))

        scheduler.step(val_f1)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            improved = "*"
        else:
            patience_counter += 1
            improved = ""

        if epoch % 10 == 0 or epoch == 1 or improved == "*":
            logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (thr=0.5): {val_f1:.4f} {improved}"
            )

        if patience_counter >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best Val F1: {best_val_f1:.4f}"
            )
            break

    elapsed = time.time() - t0
    logger.info(
        f"GraphSAGE training finished in {elapsed:.2f}s. "
        f"Best checkpoint saved to {save_path}"
    )

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, scaler, history


def train_gcn(
    data: Data,
    cfg: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    pseudo_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[GCNClassifier, StandardScaler, dict[str, list[float]]]:
    """Trains GCN classifier with full-batch message passing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg is None:
        cfg = DEFAULT_CFG["gcn"]

    logger.info(f"Setting up GCN training on device [{device}]...")

    scaler_path = cfg.get("scaler_path", "models/scaler.pkl")
    data_scaled, scaler = fit_and_scale_features(data, scaler_save_path=scaler_path)
    data_scaled = data_scaled.to(device)

    train_mask = data_scaled.train_mask & data_scaled.labeled_mask
    val_mask = data_scaled.val_mask & data_scaled.labeled_mask

    y_train_effective = data_scaled.y.clone()
    if pseudo_labels is not None:
        pseudo_indices, pseudo_y = pseudo_labels
        pseudo_indices = pseudo_indices.to(device)
        pseudo_y = pseudo_y.to(device)
        y_train_effective[pseudo_indices] = pseudo_y
        train_mask = train_mask | torch.zeros_like(train_mask).scatter_(0, pseudo_indices, True)

    n_illicit = int((y_train_effective[train_mask] == 1).sum().item())
    n_licit = int((y_train_effective[train_mask] == 0).sum().item())
    pos_weight_val = float(n_licit) / float(n_illicit) if n_illicit > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GCNClassifier(
        in_channels=data_scaled.num_node_features,
        hidden_channels=cfg.get("hidden_channels", 128),
        out_channels=1,
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 15)
    save_path = Path(cfg.get("save_path", "models/gcn_best.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data_scaled.x, data_scaled.edge_index)
        loss = criterion(logits[train_mask], y_train_effective[train_mask].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_loss = criterion(val_logits, data_scaled.y[val_mask].float()).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        y_val_np = data_scaled.y[val_mask].cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val_np, val_preds, zero_division=0))

        scheduler.step(val_f1)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            improved = "*"
        else:
            patience_counter += 1
            improved = ""

        if epoch % 10 == 0 or epoch == 1 or improved == "*":
            logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (thr=0.5): {val_f1:.4f} {improved}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
            break

    elapsed = time.time() - t0
    logger.info(f"GCN training finished in {elapsed:.2f}s. Saved to {save_path}")
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, scaler, history


def train_gat(
    data: Data,
    cfg: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    pseudo_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[GATClassifier, StandardScaler, dict[str, list[float]]]:
    """Trains GAT classifier with multi-head attention."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg is None:
        cfg = DEFAULT_CFG["gat"]

    logger.info(f"Setting up GAT training on device [{device}]...")

    scaler_path = cfg.get("scaler_path", "models/scaler.pkl")
    data_scaled, scaler = fit_and_scale_features(data, scaler_save_path=scaler_path)
    data_scaled = data_scaled.to(device)

    train_mask = data_scaled.train_mask & data_scaled.labeled_mask
    val_mask = data_scaled.val_mask & data_scaled.labeled_mask

    y_train_effective = data_scaled.y.clone()
    if pseudo_labels is not None:
        pseudo_indices, pseudo_y = pseudo_labels
        pseudo_indices = pseudo_indices.to(device)
        pseudo_y = pseudo_y.to(device)
        y_train_effective[pseudo_indices] = pseudo_y
        train_mask = train_mask | torch.zeros_like(train_mask).scatter_(0, pseudo_indices, True)

    n_illicit = int((y_train_effective[train_mask] == 1).sum().item())
    n_licit = int((y_train_effective[train_mask] == 0).sum().item())
    pos_weight_val = float(n_licit) / float(n_illicit) if n_illicit > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GATClassifier(
        in_channels=data_scaled.num_node_features,
        hidden_channels=cfg.get("hidden_channels", 64),
        out_channels=1,
        num_layers=cfg.get("num_layers", 2),
        heads=cfg.get("heads", 8),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 15)
    save_path = Path(cfg.get("save_path", "models/gat_best.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data_scaled.x, data_scaled.edge_index)
        loss = criterion(logits[train_mask], y_train_effective[train_mask].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_loss = criterion(val_logits, data_scaled.y[val_mask].float()).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        y_val_np = data_scaled.y[val_mask].cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val_np, val_preds, zero_division=0))

        scheduler.step(val_f1)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            improved = "*"
        else:
            patience_counter += 1
            improved = ""

        if epoch % 10 == 0 or epoch == 1 or improved == "*":
            logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1 (thr=0.5): {val_f1:.4f} {improved}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
            break

    elapsed = time.time() - t0
    logger.info(f"GAT training finished in {elapsed:.2f}s. Saved to {save_path}")
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, scaler, history