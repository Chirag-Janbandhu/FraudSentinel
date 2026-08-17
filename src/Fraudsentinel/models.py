"""
Model architectures for FraudSentinel.
Contains XGBoostFraudClassifier, GraphSAGEClassifier, GCNClassifier, and GATClassifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.Models")


class XGBoostFraudClassifier:
    """Wrapper around xgboost.XGBClassifier for illicit-transaction detection."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_class_weights: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_class_weights = use_class_weights
        self._model = None
        self._scale_pos_weight = 1.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int = 20,
    ) -> XGBoostFraudClassifier:
        """Fit the XGBoost classifier with early stopping on validation data."""
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError(
                "xgboost is required for XGBoostFraudClassifier. Install via: pip install xgboost"
            ) from err

        if self.use_class_weights:
            n_illicit = int((y_train == 1).sum())
            n_licit = int((y_train == 0).sum())
            if n_illicit > 0:
                self._scale_pos_weight = float(n_licit) / float(n_illicit)
                logger.info(
                    f"Computed scale_pos_weight = {self._scale_pos_weight:.2f} "
                    f"({n_licit} licit / {n_illicit} illicit)"
                )
            else:
                self._scale_pos_weight = 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self._scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        logger.info(
            f"Training XGBoost baseline on {X_train.shape[0]:,} samples "
            f"({X_train.shape[1]} features)..."
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info(f"XGBoost training complete. Best iteration: {self._model.best_iteration}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1-D array of predicted probabilities for the positive class."""
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns 1-D array of binary predictions [0, 1] using custom threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialize model to JSON format."""
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info(f"Saved XGBoost model to {path}")

    def load(self, path: str | Path) -> XGBoostFraudClassifier:
        """Load model weights from JSON file."""
        import xgboost as xgb
        path = Path(path)
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        logger.info(f"Loaded XGBoost model from {path}")
        return self


class GraphSAGEClassifier(nn.Module):
    """Multi-layer GraphSAGE classifier for node-level fraud detection."""

    def __init__(
        self,
        in_channels: int = 170,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "max",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def reset_parameters(self) -> None:
        """Re-initializes all learnable parameters in convolution layers."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass. Returns raw logit output tensor."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Returns 1-D tensor of predicted probabilities in range [0, 1]."""
        self.eval()
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)


class GCNClassifier(nn.Module):
    """Multi-layer Graph Convolutional Network (GCN) for node-level fraud detection."""

    def __init__(
        self,
        in_channels: int = 170,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self) -> None:
        """Re-initializes all learnable parameters in convolution layers."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass. Returns raw logit output tensor."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Returns 1-D tensor of predicted probabilities in range [0, 1]."""
        self.eval()
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)


class GATClassifier(nn.Module):
    """Multi-layer Graph Attention Network (GAT) for node-level fraud detection."""

    def __init__(
        self,
        in_channels: int = 170,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
                self.norms.append(nn.LayerNorm(hidden_channels * heads))

            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def reset_parameters(self) -> None:
        """Re-initializes all learnable parameters in convolution layers."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass. Returns raw logit output tensor."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Returns 1-D tensor of predicted probabilities in range [0, 1]."""
        self.eval()
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)