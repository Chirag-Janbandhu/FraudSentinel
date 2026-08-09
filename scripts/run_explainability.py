"""
FraudSentinel Explainability Pipeline — GNNExplainer vs. Captum Comparison
==========================================================================
Uses receptive-field local 2-hop subgraphs to achieve millisecond forward passes
and prevent CPU execution timeouts.

Usage:
    py scripts/run_explainability.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path("src").resolve()))

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer, ModelConfig
from torch_geometric.utils import k_hop_subgraph

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import GraphSAGEClassifier
from Fraudsentinel.evaluate import _find_best_threshold

logger = get_logger("FraudSentinel.Explainability")


# Wrap model to return probabilities for PyG Explainer compatibility
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
        help="Directory to save case-study figures"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda'"
    )
    return parser.parse_args()


def get_feature_names() -> list[str]:
    """Returns descriptive feature names for the Elliptic Bitcoin dataset (170 features)."""
    local_feats = [f"local_feat_{i}" for i in range(94)]
    agg_feats = [f"aggregate_feat_{i}" for i in range(72)]
    topo_feats = ["in_degree", "out_degree", "total_degree", "pagerank", "community_size_ratio"]
    return local_feats + agg_feats + topo_feats


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Graph data not found at {data_path}. Run graph_construction.py first.")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    sage_ckpt = model_dir / "graphsage_best.pt"
    sage_params = model_dir / "graphsage_best_params.json"
    feature_scaler_path = model_dir / "feature_scaler.pkl"

    if not (sage_ckpt.exists() and sage_params.exists()):
        logger.error("Trained GraphSAGE model and hyperparameter configs are required.")
        sys.exit(1)

    # ── Load model configuration ─────────────────────────────────────────────
    with open(sage_params, "r") as f:
        sage_cfg = json.load(f)

    # Load data
    data = torch.load(data_path, weights_only=False)

    dev = args.device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    # Load model
    model = GraphSAGEClassifier(
        in_channels=data.num_node_features,
        hidden_channels=sage_cfg["hidden_channels"],
        dropout=sage_cfg["dropout"],
        aggr=sage_cfg.get("aggr", "max")
    ).to(device)
    model.load_state_dict(torch.load(sage_ckpt, map_location=device, weights_only=True))
    model.eval()

    # Load scaler
    with open(feature_scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Scale topological features
    TOPO_START = 165
    x_np = data.x.numpy().copy()
    x_np[:, TOPO_START:] = scaler.transform(x_np[:, TOPO_START:])
    x_scaled = torch.tensor(x_np, dtype=torch.float).to(device)
    edge_index = data.edge_index.to(device)

    # Wrap model to output probabilities
    wrapped_model = ProbWrapper(model)

    # Find optimal threshold on validation split
    val_mask_labeled = data.val_mask & data.labeled_mask
    with torch.no_grad():
        all_probs = wrapped_model(x_scaled, edge_index)
        val_proba = all_probs[val_mask_labeled].cpu().numpy()
        val_y = data.y[val_mask_labeled].cpu().numpy()
        best_threshold = _find_best_threshold(val_y, val_proba)
        logger.info(f"Using locked classification threshold from validation: {best_threshold:.3f}")

        # Compute predictions
        preds = (all_probs >= best_threshold).long()

    # ── Select explainability case studies from test split ───────────────────
    test_mask_labeled = data.test_mask & data.labeled_mask
    test_indices = torch.where(test_mask_labeled)[0].cpu().numpy()
    test_probs = all_probs[test_mask_labeled].cpu().numpy()
    test_y = data.y[test_mask_labeled].cpu().numpy()
    test_preds = preds[test_mask_labeled].cpu().numpy()

    # Find matching case study indices
    cases = {}
    
    # 1. True Positive (TP): correctly identified illicit node
    tp_indices = test_indices[(test_preds == 1) & (test_y == 1)]
    if len(tp_indices) > 0:
        # Sort by degree to find a well-connected one
        degrees = [data.edge_index[0].eq(idx).sum().item() for idx in tp_indices]
        cases["TP"] = tp_indices[np.argmax(degrees)]

    # 2. True Negative (TN): correctly identified licit node
    tn_indices = test_indices[(test_preds == 0) & (test_y == 0)]
    if len(tn_indices) > 0:
        degrees = [data.edge_index[0].eq(idx).sum().item() for idx in tn_indices]
        cases["TN"] = tn_indices[np.argmax(degrees)]

    # 3. False Positive (FP): licit node misclassified as illicit (false alarm)
    fp_indices = test_indices[(test_preds == 1) & (test_y == 0)]
    if len(fp_indices) > 0:
        cases["FP"] = fp_indices[0]

    # 4. False Negative (FN): illicit node misclassified as licit (missed detection)
    fn_indices = test_indices[(test_preds == 0) & (test_y == 1)]
    if len(fn_indices) > 0:
        cases["FN"] = fn_indices[0]

    # 5. Connected TP (TP_HighDegree): another TP with multiple neighbors
    if len(tp_indices) > 1:
        sorted_tp = [idx for _, idx in sorted(zip(degrees, tp_indices), reverse=True)]
        cases["TP_HighDegree"] = sorted_tp[1] if len(sorted_tp) > 1 else sorted_tp[0]

    logger.info(f"Selected case-study nodes: {cases}")

    # ── Explainers Config ────────────────────────────────────────────────────
    model_config = ModelConfig(
        mode="binary_classification",
        task_level="node",
        return_type="probs",
    )

    # GNNExplainer
    explainer_gnn = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config,
    )

    # Captum Explainer (Integrated Gradients, no edge masks to avoid autograd issues)
    explainer_captum = Explainer(
        model=wrapped_model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=model_config,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_names()

    # ── Run case studies ─────────────────────────────────────────────────────
    for case_name, node_idx in cases.items():
        logger.info("\n" + "-" * 50)
        logger.info(f"CASE STUDY {case_name} (Node index {node_idx})")
        logger.info("-" * 50)
        
        # ── 1. Receptive Field Subgraph Extraction ───────────────────────────
        # Extract 2-hop local subgraph around target node
        subset, edge_index_sub, mapping, edge_mask_sub = k_hop_subgraph(
            node_idx=int(node_idx),
            num_hops=2,
            edge_index=data.edge_index,
            relabel_nodes=True
        )

        logger.info(f"Extracted 2-hop subgraph: nodes={subset.size(0)}, edges={edge_index_sub.size(1)}")
        
        # Slice inputs to the local subgraph to achieve millisecond forward passes
        x_sub = x_scaled[subset].to(device)
        edge_index_sub = edge_index_sub.to(device)
        mapping_item = mapping.item()

        # ── 2. Run GNNExplainer ──────────────────────────────────────────────
        logger.info("  Running GNNExplainer on receptive field...")
        t0 = time.time()
        explanation_gnn = explainer_gnn(
            x_sub,
            edge_index_sub,
            index=mapping,
        )
        logger.info(f"  GNNExplainer finished in {time.time() - t0:.2f}s")

        # ── 3. Run CaptumExplainer ───────────────────────────────────────────
        logger.info("  Running CaptumExplainer (Integrated Gradients) on receptive field...")
        t0 = time.time()
        explanation_cap = explainer_captum(
            x_sub,
            edge_index_sub,
            index=mapping,
        )
        logger.info(f"  CaptumExplainer finished in {time.time() - t0:.2f}s")

        # ── 4. Visualisation 1: Local Neighborhood Graph ────────────────────
        G = nx.Graph()
        node_colors = []
        node_labels = {}
        
        subset_list = subset.tolist()
        for i, idx in enumerate(subset_list):
            G.add_node(i)
            # Label
            lbl = data.y[idx].item()
            if i == mapping_item:
                node_colors.append("#F3A712")  # Highlight target node (golden yellow)
            elif lbl == 1:
                node_colors.append("#E71D36")  # Illicit (red)
            elif lbl == 0:
                node_colors.append("#2EC4B6")  # Licit (turquoise green)
            else:
                node_colors.append("#8D99AE")  # Unlabeled/unknown (grey)
            
            node_labels[i] = f"Node {idx}\n(y={lbl})"

        # Add edges and weights
        edge_widths = []
        for e_idx in range(edge_index_sub.size(1)):
            u = edge_index_sub[0, e_idx].item()
            v = edge_index_sub[1, e_idx].item()
            
            weight = 0.1
            if explanation_gnn.edge_mask is not None:
                weight = explanation_gnn.edge_mask[e_idx].item()
            
            G.add_edge(u, v, weight=weight)
            edge_widths.append(0.5 + 4.5 * weight)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors="black", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_family="sans-serif", font_weight="bold")
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#4A4E69")
        
        plt.title(f"Local Transaction Neighborhood: Case {case_name} (Node {node_idx})\n"
                  f"Edge thickness indicates GNNExplainer importance. Gold node is the target.", 
                  fontsize=12, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        
        neigh_fig_path = output_dir / f"case_study_{case_name}_neighborhood.png"
        plt.savefig(neigh_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved neighborhood visualization to {neigh_fig_path}")

        # ── 5. Visualisation 2: Feature Attribution Comparison ───────────────
        # Extract feature importances for target node
        gnn_feat_imp = explanation_gnn.node_mask[mapping_item].cpu().numpy()
        cap_feat_imp = explanation_cap.node_mask[mapping_item].cpu().numpy()

        # Absolute attribution values for rank sorting
        gnn_feat_imp_abs = np.abs(gnn_feat_imp)
        
        # Find top 10 features according to GNNExplainer
        top_10_indices = np.argsort(gnn_feat_imp_abs)[-10:]
        
        top_features = [feature_names[idx] for idx in top_10_indices]
        gnn_vals = gnn_feat_imp[top_10_indices]
        cap_vals = cap_feat_imp[top_10_indices]

        plt.figure(figsize=(12, 6))
        y_pos = np.arange(len(top_features))
        width = 0.35

        plt.barh(y_pos - width/2, gnn_vals, width, label="GNNExplainer (edge-aware)", color="#F25C54")
        plt.barh(y_pos + width/2, cap_vals, width, label="Captum (Integrated Gradients)", color="#3A86C8")

        plt.yticks(y_pos, top_features, fontsize=10, fontweight="bold")
        plt.xlabel("Attribution Value (Feature Importance)", fontsize=11, fontweight="bold")
        plt.title(f"Feature Attribution Comparison: Case {case_name} (Node {node_idx})\n"
                  f"Top 10 features ranked by GNNExplainer.", fontsize=12, fontweight="bold")
        plt.legend(loc="best")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        feat_fig_path = output_dir / f"case_study_{case_name}_features.png"
        plt.savefig(feat_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved feature attribution comparison to {feat_fig_path}")

    logger.info("Explainability analysis script executed successfully!")


if __name__ == "__main__":
    main()
