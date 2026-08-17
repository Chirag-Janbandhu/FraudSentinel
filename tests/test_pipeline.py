import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Fraudsentinel.graph_construction import (
    build_pyg_data,
    engineer_topological_features,
    remap_nodes,
)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        feat_dict = {
            "txId": [1001, 1002, 1003, 1004],
            "time_step": [1, 1, 2, 2],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [-0.5, 0.5, -0.2, 0.8],
        }
        for i in range(3, 166):
            feat_dict[f"feature_{i}"] = np.random.randn(4).astype(np.float32)
        self.features_df = pd.DataFrame(feat_dict)

        self.classes_df = pd.DataFrame({
            "txId": [1001, 1002, 1003, 1004],
            "class": ["1", "2", "unknown", "1"]
        })

        self.edges_df = pd.DataFrame({
            "txId1": [1001, 1002, 1003],
            "txId2": [1002, 1003, 1004]
        })

    def test_remap_nodes(self):
        id_to_idx, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )

        self.assertEqual(len(id_to_idx), 4)
        self.assertEqual(len(merged_df), 4)
        self.assertEqual(len(edges_df), 3)

        self.assertEqual(id_to_idx[1001], 0)
        self.assertEqual(id_to_idx[1004], 3)

        self.assertIn("node_idx", merged_df.columns)
        self.assertIn("source_idx", edges_df.columns)
        self.assertIn("target_idx", edges_df.columns)

        self.assertEqual(edges_df["source_idx"].tolist(), [0, 1, 2])
        self.assertEqual(edges_df["target_idx"].tolist(), [1, 2, 3])

    def test_engineer_topological_features(self):
        _, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )
        df_topo = engineer_topological_features(merged_df, edges_df)

        for col in ["in_degree", "out_degree", "total_degree", "pagerank", "community_id"]:
            self.assertIn(col, df_topo.columns)

        self.assertEqual(df_topo.loc[0, "out_degree"], 1)
        self.assertEqual(df_topo.loc[0, "in_degree"], 0)
        self.assertEqual(df_topo.loc[3, "in_degree"], 1)

    def test_build_pyg_data(self):
        _, merged_df, edges_df = remap_nodes(
            self.features_df, self.classes_df, self.edges_df
        )
        df_topo = engineer_topological_features(merged_df, edges_df)
        data = build_pyg_data(df_topo, edges_df)

        self.assertIsInstance(data, Data)
        self.assertEqual(data.num_nodes, 4)
        self.assertEqual(data.num_node_features, 170)
        self.assertEqual(data.edge_index.shape[1], 3)

        self.assertTrue(hasattr(data, "train_mask"))
        self.assertTrue(hasattr(data, "val_mask"))
        self.assertTrue(hasattr(data, "test_mask"))
        self.assertTrue(hasattr(data, "labeled_mask"))


if __name__ == "__main__":
    unittest.main()
