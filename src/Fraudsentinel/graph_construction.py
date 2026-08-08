"""
Build the PyTorch Geometric graph object for the Elliptic Bitcoin dataset.
Handles: loading raw CSVs, transaction ID remapping to sequential node
indices, topological feature engineering, edge index construction,
temporal train/val/test splitting, and saving PyG Data objects.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

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


def load_raw_data(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
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
) -> Tuple[Dict[int, int], pd.DataFrame, pd.DataFrame]:
    """
    Remaps non-sequential Bitcoin transaction IDs to 0..N-1 sequential node indices.
    
    Returns:
        id_to_idx: Dictionary mapping txId -> sequential index [0..N-1]
        merged_df: DataFrame containing features + class sorted by node index [0..N-1]
        edges_df: DataFrame with added source_idx and target_idx columns
    """
    logger.info("Remapping transaction IDs to 0..N-1 sequential node indices...")
    
    features_df = features_df.sort_values("txId").reset_index(drop=True)
    
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(features_df["txId"])}
    features_df["node_idx"] = features_df["txId"].map(id_to_idx)
    
    merged_df = pd.merge(features_df, classes_df, on="txId", how="left")
    merged_df = merged_df.sort_values("node_idx").reset_index(drop=True)
    
    edges_df = edges_df.copy()
    edges_df["source_idx"] = edges_df["txId1"].map(id_to_idx)
    edges_df["target_idx"] = edges_df["txId2"].map(id_to_idx)
    
    valid_edges_mask = edges_df["source_idx"].notna() & edges_df["target_idx"].notna()
    num_invalid = len(edges_df) - valid_edges_mask.sum()
    if num_invalid > 0:
        logger.warning(f"Dropping {num_invalid} edges referencing missing nodes.")
        edges_df = edges_df[valid_edges_mask].copy()
        
    edges_df["source_idx"] = edges_df["source_idx"].astype(int)
    edges_df["target_idx"] = edges_df["target_idx"].astype(int)
    
    logger.info(f"Remapping complete. {len(merged_df)} nodes, {len(edges_df)} valid edges.")
    return id_to_idx, merged_df, edges_df


def engineer_topological_features(edges_df: pd.DataFrame, num_nodes: int) -> pd.DataFrame:
    """
    Computes graph topological features per node:
    - in_degree: Number of incoming Bitcoin transaction edges.
    - out_degree: Number of outgoing Bitcoin transaction edges.
    - total_degree: Sum of in-degree and out-degree.
    - pagerank: PageRank structural centrality (alpha=0.85).
    - community_id [int, NOT in x]: Raw Louvain community partition label.
      Community IDs are categorical (community 47 is not numerically greater
      than community 12), so they are stored separately and NEVER placed
      directly into the continuous feature matrix.
    - community_size_ratio [float, in x]: Count-encoding of community membership.
      Value = (number of nodes in same community) / total_nodes.
      Captures whether a transaction belongs to a large dense cluster or a small
      isolated subgraph, without imposing any false ordinal relationship.

    Returns:
        DataFrame indexed 0..num_nodes-1 with columns:
        [in_degree, out_degree, total_degree, pagerank, community_id, community_size_ratio]
    """
    logger.info(f"Engineering topological features for {num_nodes} nodes...")

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    edge_pairs = list(zip(edges_df["source_idx"], edges_df["target_idx"]))
    G.add_edges_from(edge_pairs)

    in_degrees = np.array([deg for _, deg in G.in_degree()])
    out_degrees = np.array([deg for _, deg in G.out_degree()])
    total_degrees = in_degrees + out_degrees

    logger.info("Computing PageRank centrality...")
    pr_dict = nx.pagerank(G, alpha=0.85)
    pageranks = np.array([pr_dict.get(i, 0.0) for i in range(num_nodes)])

    logger.info("Performing Community Detection (Louvain)...")
    G_undirected = G.to_undirected()

    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G_undirected)
        community_ids = np.array([partition.get(i, 0) for i in range(num_nodes)])
    else:
        logger.warning(
            "python-louvain not found; falling back to connected components for community_id. "
            "Install python-louvain for proper Louvain community detection."
        )
        components = list(nx.connected_components(G_undirected))
        community_ids = np.zeros(num_nodes, dtype=int)
        for comp_id, comp in enumerate(components):
            for node in comp:
                community_ids[node] = comp_id

    # --- Community size count-encoding ---
    # community_id is categorical (not ordinal), so we NEVER feed the raw integer
    # into the continuous feature matrix. Instead we use community_size_ratio:
    # the fraction of all nodes that belong to the same community.
    # This preserves cluster-membership signal without implying ordinal ranking.
    
    
    unique_ids, counts = np.unique(community_ids, return_counts=True)
    comm_size_map = dict(zip(unique_ids, counts))
    community_size_ratio = np.array(
        [comm_size_map[c] / num_nodes for c in community_ids], dtype=np.float32
    )
    num_communities = len(unique_ids)
    logger.info(
        f"Community detection complete: {num_communities} communities detected. "
        f"community_size_ratio range: [{community_size_ratio.min():.6f}, {community_size_ratio.max():.4f}]"
    )

    topo_df = pd.DataFrame({
        "in_degree": in_degrees,
        "out_degree": out_degrees,
        "total_degree": total_degrees,
        "pagerank": pageranks,
        "community_id": community_ids,           # kept for interpretability/analysis only
        "community_size_ratio": community_size_ratio  # the actual continuous feature for x
    }, index=range(num_nodes))

    logger.info("Topological features computed successfully.")
    return topo_df


def create_pyg_data(
    merged_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    topo_df: pd.DataFrame
) -> Data:
    """
    Assembles the node feature matrix, label tensor, edge index, and temporal
    boolean masks into a PyTorch Geometric Data object.

    Feature matrix x layout [num_nodes, 169]:
        Cols   0–164  : 165 raw transaction features (pre-standardized by dataset creators)
        Cols 165–168  : 4 continuous topological features:
                        165  in_degree
                        166  out_degree
                        167  total_degree
                        168  pagerank
                        169  community_size_ratio  (count-encoded, NOT raw community_id)

    NOTE on community encoding:
        community_id is a categorical label (Louvain partition). Encoding it as
        a raw float would impose a false ordinal relationship on the model.
        Instead, we use community_size_ratio = community_size / total_nodes,
        which gives a meaningful continuous signal (large cluster vs small cluster)
        without any ordinal artifact.
        The raw integer community_id is stored separately as `data.community_id`
        for downstream interpretability and analysis.
    """
    logger.info("Assembling PyTorch Geometric Data object...")

    num_nodes = len(merged_df)

    # --- Raw node features ---
    raw_feature_cols = [f"feature_{i}" for i in range(1, 166)]
    raw_features = merged_df[raw_feature_cols].values.astype(np.float32)

    # --- Continuous topological features (safe to concatenate) ---
    # community_size_ratio replaces raw community_id in x; see docstring above.
    continuous_topo_cols = ["in_degree", "out_degree", "total_degree", "pagerank", "community_size_ratio"]
    topo_continuous = topo_df[continuous_topo_cols].values.astype(np.float32)

    # Final feature matrix: [num_nodes, 165 + 5] = [num_nodes, 170]
    x_matrix = np.hstack([raw_features, topo_continuous])
    x = torch.tensor(x_matrix, dtype=torch.float)

    # --- Categorical community_id stored separately (NOT in x) ---
    community_id_tensor = torch.tensor(
        topo_df["community_id"].values.astype(np.int64), dtype=torch.long
    )

    # --- Labels ---
    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    y_raw = merged_df["class"].map(label_map).fillna(-1).values.astype(np.int64)
    y = torch.tensor(y_raw, dtype=torch.long)

    # --- Edge Index: [2, num_edges] ---
    edge_index = torch.tensor(
        edges_df[["source_idx", "target_idx"]].values.T,
        dtype=torch.long
    )

    # --- Temporal split masks (labeled nodes only) ---
    time_steps = torch.tensor(merged_df["time_step"].values, dtype=torch.long)
    labeled_mask = (y != -1)
    train_mask = (time_steps >= 1) & (time_steps <= 34) & labeled_mask
    val_mask   = (time_steps >= 35) & (time_steps <= 42) & labeled_mask
    test_mask  = (time_steps >= 43) & (time_steps <= 49) & labeled_mask

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        time_step=time_steps,
        labeled_mask=labeled_mask,
        community_id=community_id_tensor   # categorical, for analysis only — NOT in x
    )

    logger.info(
        f"PyG Data assembled: nodes={graph_data.num_nodes}, "
        f"edges={graph_data.num_edges}, features={graph_data.num_node_features}"
    )
    logger.info(
        f"Mask breakdown: Train={train_mask.sum().item()}, "
        f"Val={val_mask.sum().item()}, Test={test_mask.sum().item()}, "
        f"Unknown/Unlabeled={(y == -1).sum().item()}"
    )
    return graph_data


def build_and_save_graph(
    data_dir: Union[str, Path] = "data/raw/archive/elliptic_bitcoin_dataset",
    output_path: Union[str, Path] = "data/processed/graph_data.pt"
) -> Data:
    """
    Executes end-to-end graph construction and saves the PyG Data object.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features_df, classes_df, edges_df = load_raw_data(data_dir)
    id_to_idx, merged_df, edges_df = remap_nodes(features_df, classes_df, edges_df)
    topo_df = engineer_topological_features(edges_df, len(merged_df))
    graph_data = create_pyg_data(merged_df, edges_df, topo_df)
    
    logger.info(f"Saving graph data object to {output_path}...")
    torch.save(graph_data, output_path)
    logger.info("Graph data object saved successfully!")
    
    return graph_data


if __name__ == "__main__":
    build_and_save_graph()
