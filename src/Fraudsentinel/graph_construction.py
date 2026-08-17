"""
Build the PyTorch Geometric graph object for the Elliptic Bitcoin dataset.
Handles: loading raw CSVs, transaction ID remapping to sequential node
indices, topological feature engineering, edge index construction,
temporal train/val/test splitting, and saving PyG Data objects.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

logger = get_logger("FraudSentinel.GraphConstruction")


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    features_file = data_dir / "elliptic_txs_features.csv"
    classes_file = data_dir / "elliptic_txs_classes.csv"
    edges_file = data_dir / "elliptic_txs_edgelist.csv"

    logger.info(f"Loading raw dataset from {data_dir}...")

    feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
    features_df = pd.read_csv(features_file, header=None, names=feature_cols)

    classes_df = pd.read_csv(classes_file)
    edges_df = pd.read_csv(edges_file)

    logger.info(
        f"Raw data loaded: {len(features_df)} features rows, "
        f"{len(classes_df)} classes rows, {len(edges_df)} edges rows."
    )
    return features_df, classes_df, edges_df


def remap_nodes(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    edges_df: pd.DataFrame
) -> tuple[dict[int, int], pd.DataFrame, pd.DataFrame]:
    """Remaps non-sequential transaction IDs to 0..N-1 sequential node indices."""
    logger.info("Remapping transaction IDs to 0..N-1 sequential node indices...")

    unique_tx_ids = sorted(features_df["txId"].unique())
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(unique_tx_ids)}

    merged_df = features_df.copy()
    merged_df["node_idx"] = merged_df["txId"].map(id_to_idx)
    merged_df = merged_df.sort_values("node_idx").reset_index(drop=True)

    class_map = {"1": 1, "2": 0, "unknown": -1}
    classes_mapped = classes_df.copy()
    classes_mapped["class"] = classes_mapped["class"].map(class_map).fillna(-1).astype(int)

    classes_mapped["node_idx"] = classes_mapped["txId"].map(id_to_idx)
    classes_mapped = classes_mapped.dropna(subset=["node_idx"])
    classes_mapped["node_idx"] = classes_mapped["node_idx"].astype(int)

    merged_df = merged_df.merge(classes_mapped[["node_idx", "class"]], on="node_idx", how="left")

    edges_valid = edges_df[
        edges_df["txId1"].isin(id_to_idx) & edges_df["txId2"].isin(id_to_idx)
    ].copy()

    edges_valid["source_idx"] = edges_valid["txId1"].map(id_to_idx)
    edges_valid["target_idx"] = edges_valid["txId2"].map(id_to_idx)

    logger.info(f"Remapping complete. {len(merged_df)} nodes, {len(edges_valid)} valid edges.")
    return id_to_idx, merged_df, edges_valid


def engineer_topological_features(
    merged_df: pd.DataFrame,
    edges_df: pd.DataFrame
) -> pd.DataFrame:
    """Engineers in-degree, out-degree, total-degree, PageRank, and Louvain community size ratio."""
    logger.info(f"Engineering topological features for {len(merged_df)} nodes...")

    num_nodes = len(merged_df)
    in_degrees = np.zeros(num_nodes, dtype=np.float32)
    out_degrees = np.zeros(num_nodes, dtype=np.float32)

    in_counts = edges_df["target_idx"].value_counts()
    out_counts = edges_df["source_idx"].value_counts()

    in_degrees[in_counts.index.values] = in_counts.values.astype(np.float32)
    out_degrees[out_counts.index.values] = out_counts.values.astype(np.float32)
    total_degrees = in_degrees + out_degrees

    logger.info("Computing PageRank centrality...")
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    edge_tuples = list(zip(edges_df["source_idx"].values, edges_df["target_idx"].values))
    G.add_edges_from(edge_tuples)

    pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    pagerank_vals = np.array([pr_dict[i] for i in range(num_nodes)], dtype=np.float32)

    logger.info("Performing Community Detection (Louvain)...")
    if HAS_LOUVAIN:
        G_undirected = G.to_undirected()
        partition = community_louvain.best_partition(G_undirected, random_state=42)

        comm_counts = {}
        for comm in partition.values():
            comm_counts[comm] = comm_counts.get(comm, 0) + 1

        comm_size_ratios = np.zeros(num_nodes, dtype=np.float32)
        for i in range(num_nodes):
            comm_id = partition[i]
            comm_size_ratios[i] = comm_counts[comm_id] / num_nodes

        logger.info(
            f"Community detection complete: {len(comm_counts)} communities detected. "
            f"community_size_ratio range: [{comm_size_ratios.min():.6f}, {comm_size_ratios.max():.4f}]"
        )
    else:
        logger.warning("python-louvain not installed; using zero fallback for community_size_ratio.")
        comm_size_ratios = np.zeros(num_nodes, dtype=np.float32)

    df_topo = merged_df.copy()
    df_topo["in_degree"] = in_degrees
    df_topo["out_degree"] = out_degrees
    df_topo["total_degree"] = total_degrees
    df_topo["pagerank"] = pagerank_vals
    df_topo["community_id"] = comm_size_ratios

    logger.info("Topological features computed successfully.")
    return df_topo


def build_pyg_data(
    df_with_topo: pd.DataFrame,
    edges_df: pd.DataFrame
) -> Data:
    """Assembles a PyTorch Geometric Data object with 170-dim features and temporal split masks."""
    logger.info("Assembling PyTorch Geometric Data object...")

    feature_cols = [f"feature_{i}" for i in range(1, 166)]
    topo_cols = ["in_degree", "out_degree", "total_degree", "pagerank", "community_id"]
    all_feature_cols = feature_cols + topo_cols

    X_np = df_with_topo[all_feature_cols].values.astype(np.float32)
    x = torch.tensor(X_np, dtype=torch.float)

    y_np = df_with_topo["class"].values.astype(np.int64)
    y = torch.tensor(y_np, dtype=torch.long)

    time_steps = torch.tensor(df_with_topo["time_step"].values, dtype=torch.long)

    src = torch.tensor(edges_df["source_idx"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["target_idx"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    train_mask = (time_steps >= 1) & (time_steps <= 34)
    val_mask = (time_steps >= 35) & (time_steps <= 42)
    test_mask = (time_steps >= 43) & (time_steps <= 49)

    labeled_mask = (y != -1)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_steps,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labeled_mask=labeled_mask,
    )

    logger.info(
        f"PyG Data assembled: nodes={data.num_nodes}, edges={data.num_edges}, "
        f"features={data.num_node_features}"
    )
    logger.info(
        f"Mask breakdown: Train={train_mask.sum().item()}, Val={val_mask.sum().item()}, "
        f"Test={test_mask.sum().item()}, Unknown/Unlabeled={(~labeled_mask).sum().item()}"
    )

    return data


def run_pipeline(data_dir: str | Path, output_path: str | Path) -> Data:
    """Runs end-to-end graph construction pipeline and serializes graph_data.pt."""
    logger.info("Starting Graph Construction Pipeline...")
    features_df, classes_df, edges_df = load_raw_data(data_dir)
    _, merged_df, edges_valid = remap_nodes(features_df, classes_df, edges_df)
    df_topo = engineer_topological_features(merged_df, edges_valid)
    data = build_pyg_data(df_topo, edges_valid)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    logger.info(f"Pipeline complete! Saved graph_data.pt to {output_path}")

    return data
