"""
Builds the PyTorch Geometric graph object for the Elliptic dataset.
Handles: loading raw CSVs, transaction ID remapping to sequential node
indices, edge index construction, temporal train/val/test splitting,
and topological feature engineering (degree centrality, community
detection via Louvain).

"""