"""
Model architectures for FraudSentinel.

XGBoostFraudClassifier
    Thin, serialisable wrapper around xgboost.XGBClassifier.
    Auto-computes scale_pos_weight from the training label distribution
    so the model is class-imbalance-aware out of the box.

GraphSAGEClassifier
    3-layer inductive GNN using PyG SAGEConv operators.
    Designed to be trained via NeighborLoader (mini-batch, scalable).
    BatchNorm + Dropout between layers; single logit output for
    BCEWithLogitsLoss (binary fraud / not-fraud).

Design note on class imbalance
    The Elliptic dataset is ~10 % illicit / 90 % licit.
    Both models handle this natively:
      - XGBoost : scale_pos_weight = n_licit / n_illicit
      - GraphSAGE : pos_weight tensor in BCEWithLogitsLoss
    We do NOT oversample/undersample because the temporal split must
    be kept intact to avoid data leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.Models")


# ---------------------------------------------------------------------------
# XGBoost Baseline
# ---------------------------------------------------------------------------

class XGBoostFraudClassifier:
    """
    Wrapper around xgboost.XGBClassifier for illicit-transaction detection.

    Why XGBoost as baseline?
        It operates on node features alone (no graph structure), so any
        improvement from GraphSAGE directly measures the value of
        neighbourhood information.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds (early stopping may reduce this).
    max_depth : int
        Maximum tree depth. 6 is a sensible default for tabular fraud data.
    learning_rate : float
        Shrinkage rate per boosting step.
    subsample : float
        Row subsampling ratio per tree.
    colsample_bytree : float
        Feature subsampling ratio per tree.
    early_stopping_rounds : int
        Stop if val metric does not improve for this many rounds.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ):
        try:
            import xgboost as xgb  # local import keeps the rest importable without xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model: Optional[object] = None
        self.scale_pos_weight: Optional[float] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "XGBoostFraudClassifier":
        """
        Train the XGBoost classifier.

        scale_pos_weight is derived from the training set so the model
        never sees validation label statistics.
        """
        n_licit = int((y_train == 0).sum())
        n_illicit = int((y_train == 1).sum())
        self.scale_pos_weight = n_licit / max(n_illicit, 1)

        logger.info(
            f"XGBoost | train size={len(y_train)} "
            f"(illicit={n_illicit}, licit={n_licit}) | "
            f"scale_pos_weight={self.scale_pos_weight:.2f}"
        )

        self.model = self._xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="aucpr",          # PR-AUC on val — matches our primary concern
            random_state=self.random_state,
            tree_method="hist",           # fast CPU training
            verbosity=0,
            use_label_encoder=False,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_round = self.model.best_iteration
        logger.info(f"XGBoost | best iteration={best_round}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of illicit class (class=1) for each node."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Union[str, Path]) -> "XGBoostFraudClassifier":
        path = Path(path)
        self.model = self._xgb.XGBClassifier()
        self.model.load_model(str(path))
        logger.info(f"XGBoost model loaded from {path}")
        return self


# ---------------------------------------------------------------------------
# GraphSAGE Classifier
# ---------------------------------------------------------------------------

class GraphSAGEClassifier(nn.Module):
    """
    3-layer GraphSAGE for binary node classification (illicit / licit).

    Architecture
    ------------
    Input  : [num_nodes, in_channels]          (170 features per node)
    Layer 1: SAGEConv(in -> hidden[0], aggr=max) + BatchNorm + ReLU + Dropout
    Layer 2: SAGEConv(hidden[0] -> hidden[1], aggr=max) + BatchNorm + ReLU + Dropout
    Head   : Linear(hidden[-1] -> 1)  -- raw logit for BCEWithLogitsLoss

    Why SAGEConv with max aggregation?
        Inductive -- generalises to unseen nodes without retraining.
        max aggregation preserves the strongest fraud signal from any
        neighbour in the receptive field. mean aggregation averages the
        signal over many licit neighbours, diluting the illicit indicator.
        For fraud/anomaly detection, max is the principled choice.

    Why 2 layers?
        2 hops covers the immediate transaction counterparties of each
        node. 3+ hops aggregate from distant nodes whose transactions
        are likely unrelated to the local fraud pattern, adding noise.

    Parameters
    ----------
    in_channels : int
        Number of input node features (170 for this dataset).
    hidden_channels : List[int]
        Width of each SAGEConv layer. Default [256, 128] (2 layers).
    dropout : float
        Dropout probability applied after each hidden layer.
    aggr : str
        Neighborhood aggregation method. Default "max".
        "max" is preferred for fraud detection -- it preserves the strongest
        illicit signal from any single neighbor, rather than averaging it
        over the (mostly licit) neighborhood.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[List[int]] = None,
        dropout: float = 0.3,
        aggr: str = "max",
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128]

        self.dropout = dropout
        dims = [in_channels] + hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # kept name self.bns for minimal diff in forward()
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))
            self.bns.append(nn.LayerNorm(dims[i + 1]))

        self.head = nn.Linear(hidden_channels[-1], 1)

        self.reset_parameters()
        logger.info(
            f"GraphSAGEClassifier | in={in_channels} | "
            f"hidden={hidden_channels} | dropout={dropout} | aggr={aggr}"
        )

    def reset_parameters(self) -> None:
        """Xavier initialisation for reproducible experiments."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
            else:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [N, in_channels]
        edge_index : Tensor [2, E]

        Returns
        -------
        Tensor [N, 1]  — raw logits (not probabilities)
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x)   # [N, 1] raw logit

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> np.ndarray:
        """Return fraud probability in [0, 1] for each node."""
        self.eval()
        logits = self.forward(x, edge_index).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# GCN (Graph Convolutional Network) Classifier
# ---------------------------------------------------------------------------

class GCNClassifier(nn.Module):
    """
    2-layer Graph Convolutional Network (GCN) for binary node classification.

    Architecture
    ------------
    Input  : [num_nodes, in_channels]
    Layer 1: GCNConv(in -> hidden[0]) + BatchNorm + ReLU + Dropout
    Layer 2: GCNConv(hidden[0] -> hidden[1]) + BatchNorm + ReLU + Dropout
    Head   : Linear(hidden[1] -> 1)  -- raw logit out

    Why GCN?
        GCN uses symmetric normalisation to scale aggregation. It is a standard
        isotropic baseline where each transaction counterpart of a node is
        weighted equally, normalized by the degree of both nodes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128]

        self.dropout = dropout
        dims = [in_channels] + hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # kept name self.bns for minimal diff in forward()
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.LayerNorm(dims[i + 1]))

        self.head = nn.Linear(hidden_channels[-1], 1)

        self.reset_parameters()
        logger.info(
            f"GCNClassifier | in={in_channels} | "
            f"hidden={hidden_channels} | dropout={dropout}"
        )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
            else:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> np.ndarray:
        self.eval()
        logits = self.forward(x, edge_index).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# GAT (Graph Attention Network) Classifier
# ---------------------------------------------------------------------------

class GATClassifier(nn.Module):
    """
    2-layer Graph Attention Network (GAT) for binary node classification.

    Architecture
    ------------
    Input  : [num_nodes, in_channels]
    Layer 1: GATConv(in -> hidden[0], heads=4) -> Concatenated outputs (dim = hidden[0]*4)
    Layer 2: GATConv(hidden[0]*4 -> hidden[1], heads=1) -> Output (dim = hidden[1])
    Head   : Linear(hidden[1] -> 1)  -- raw logit out

    Why GAT?
        Financial relationships are anisotropic (direction/importance matters).
        Transacting with a high-risk wallet should carry more weight than transacting
        with a utility or exchange. GAT uses self-attention to learn dynamic transaction
        coefficients, focusing attention on suspicious connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[List[int]] = None,
        dropout: float = 0.3,
        heads: int = 4,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128]  # [64*4=256 out first layer, 128 out second]

        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # kept name self.bns for minimal diff in forward()

        # First layer: GATConv -> output gets concatenated from multiple heads
        self.convs.append(GATConv(in_channels, hidden_channels[0], heads=heads, concat=True, dropout=0.2))
        self.bns.append(nn.LayerNorm(hidden_channels[0] * heads))

        # Second layer: GATConv -> single head output
        self.convs.append(GATConv(hidden_channels[0] * heads, hidden_channels[1], heads=1, concat=False, dropout=0.2))
        self.bns.append(nn.LayerNorm(hidden_channels[1]))

        self.head = nn.Linear(hidden_channels[1], 1)

        self.reset_parameters()
        logger.info(
            f"GATClassifier | in={in_channels} | "
            f"hidden={hidden_channels} | heads={heads} | dropout={dropout}"
        )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
            else:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> np.ndarray:
        self.eval()
        logits = self.forward(x, edge_index).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()