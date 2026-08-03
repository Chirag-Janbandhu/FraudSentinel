"""
Training loops for all models. Handles class-imbalance-aware loss
weighting, neighbor sampling via NeighborLoader for GNN training,
early stopping on validation F1, and experiment tracking via MLflow
or equivalent logging.
"""