"""
FraudSentinel Explainability Pipeline — GNNExplainer vs. Captum Comparison
==========================================================================
Generates feature and neighborhood attributions for GraphSAGE predictions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

import numpy as np
import torch
from torch_geometric.explain import (
    CaptumExplainer,
    Explainer,
    GNNExplainer,
    ModelConfig,
)

from Fraudsentinel.evaluate import _find_best_threshold
from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import GraphSAGEClassifier

logger = get_logger("FraudSentinel.Explainability")


class ProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        return torch.sigmoid(logits).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudSentinel — Explain GNN predictions using GNNExplainer and Captum"
    )
    parser.add_argument(
        "--data-path", default="data/processed/graph_data.pt",
        help="Path to the saved PyG Data object"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory containing trained model and parameters"
    )
    parser.add_argument(
        "--output-dir", default="reports/figures/explainability",
        help="Directory to save figures and markdown report"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of optimization epochs per node for GNNExplainer"
    )
    return parser.parse_args()


def load_resources(data_path: Path, model_dir: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info(f"Loading graph data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)

    params_path = model_dir / "graphsage_best_params.json"
    checkpoint_path = model_dir / "graphsage_best.pt"
    scaler_path = model_dir / "scaler.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    if params_path.exists():
        with open(params_path) as f:
            best_params = json.load(f)
        hidden_channels = best_params.get("hidden_channels", 128)
        dropout = best_params.get("dropout", 0.2)
        aggr = best_params.get("aggr", "max")
    else:
        hidden_channels, dropout, aggr = 128, 0.2, "max"

    model = GraphSAGEClassifier(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=dropout,
        aggr=aggr
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    data_scaled = data.clone()
    if scaler is not None:
        X_scaled = scaler.transform(data.x.numpy()).astype(np.float32)
        data_scaled.x = torch.tensor(X_scaled, dtype=torch.float)

    return data_scaled, model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_scaled, model = load_resources(Path(args.data_path), Path(args.model_dir))
    wrapped_model = ProbWrapper(model)
    wrapped_model.eval()

    with torch.no_grad():
        test_mask = data_scaled.test_mask & data_scaled.labeled_mask
        val_mask = data_scaled.val_mask & data_scaled.labeled_mask

        val_probs = wrapped_model(data_scaled.x, data_scaled.edge_index)[val_mask].numpy()
        val_y = data_scaled.y[val_mask].numpy()
        best_thresh, best_f1, _, _ = _find_best_threshold(val_y, val_probs)
        logger.info(f"Optimal decision threshold locked on Val split: {best_thresh:.4f} (Val F1={best_f1:.4f})")

        all_probs = wrapped_model(data_scaled.x, data_scaled.edge_index)
        preds = (all_probs >= best_thresh).long()

    feature_names = [f"feature_{i}" for i in range(1, 166)] + [
        "in_degree", "out_degree", "total_degree", "pagerank", "community_id"
    ]

    explainer_gnn = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=args.epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="probs"
        ),
    )

    explainer_captum = Explainer(
        model=wrapped_model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="probs"
        ),
    )

    logger.info("Explainability pipeline completed.")


if __name__ == "__main__":
    main()
