import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Fraudsentinel.graph_construction import remap_nodes, engineer_topological_features, create_pyg_data


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataset: 4 nodes, 3 edges, 2 timesteps
        self.features_df = pd.DataFrame({
            "txId": [1001, 1002, 1003, 1004],
            "time_step": [1, 1, 2, 2],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [-0.5, 0.5, -0.2, 0.8]
        })
        # Mock feature columns up to 165
        for i in range(3, 166):
            self.features_df[f"feature_{i}"] = np.random.randn(4).astype(np.float32)

        self.classes_df = pd.DataFrame({
            "txId": [1001, 1002, 1003, 1004],
            "class": ["1", "2", "unknown", "1"]  # 1=illicit, 2=licit, unknown=-1
        })
        
        self.edges_df = pd.DataFrame({
            "txId1": [1001, 1002, 1003],
            "txId2": [1002, 1003, 1004]
        })

    def test_remap_nodes(self):
        id_to_idx, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )
        
        # Verify node counts
        self.assertEqual(len(id_to_idx), 4)
        self.assertEqual(len(merged_df), 4)
        self.assertEqual(len(edges_df), 3)

        # Verify mappings
        self.assertEqual(id_to_idx[1001], 0)
        self.assertEqual(id_to_idx[1004], 3)

        # Verify columns added
        self.assertIn("node_idx", merged_df.columns)
        self.assertIn("source_idx", edges_df.columns)
        self.assertIn("target_idx", edges_df.columns)

        # Verify sequential indices in edges
        self.assertEqual(edges_df.iloc[0]["source_idx"], 0)
        self.assertEqual(edges_df.iloc[0]["target_idx"], 1)

    def test_engineer_topological_features(self):
        id_to_idx, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )
        topo_df = engineer_topological_features(edges_df, len(merged_df))
        
        # Verify shape
        self.assertEqual(len(topo_df), 4)
        self.assertListEqual(
            list(topo_df.columns),
            ["in_degree", "out_degree", "total_degree", "pagerank", "community_id", "community_size_ratio"]
        )
        
        # Verify degree calculations (directed edges: 0->1, 1->2, 2->3)
        # Node 0: in=0, out=1
        self.assertEqual(topo_df.loc[0, "in_degree"], 0)
        self.assertEqual(topo_df.loc[0, "out_degree"], 1)
        # Node 1: in=1, out=1
        self.assertEqual(topo_df.loc[1, "in_degree"], 1)
        self.assertEqual(topo_df.loc[1, "out_degree"], 1)

    def test_create_pyg_data(self):
        id_to_idx, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )
        topo_df = engineer_topological_features(edges_df, len(merged_df))
        data = create_pyg_data(merged_df, edges_df, topo_df)

        self.assertIsInstance(data, Data)
        # Nodes = 4, Features = 165 raw + 5 topo = 170
        self.assertEqual(data.x.shape, (4, 170))
        # Edges = 3
        self.assertEqual(data.edge_index.shape, (2, 3))
        # Labels map checking (1->1, 2->0, unknown->-1)
        self.assertListEqual(data.y.tolist(), [1, 0, -1, 1])


if __name__ == "__main__":
    unittest.main()
