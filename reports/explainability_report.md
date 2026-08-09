# GNN Explainability & Visualisation Report (GNNExplainer vs. Captum)

This report details the explainability analysis for the tuned **GraphSAGE (Max)** fraud detection model. Using PyG's `Explainer` framework, we generated local structural (edge masks) and feature (attribute attribution) explanations for 5 transaction case studies from the Test Split (timesteps 43-49) under concept drift. 

---

## 🔬 Explainability Framework Design

*   **Primary Explainer**: **GNNExplainer**. It optimizes an edge mask $M_E$ and a feature mask $M_F$ to maximize the mutual information between the model's prediction on the target node and the prediction on the masked subgraph:
    $$\max_{M_E, M_F} MI(Y, \hat{Y}(G_s, X_s))$$
*   **Comparison Explainer**: **Captum (Integrated Gradients)**. It computes feature attribution by integrating the gradients of the model output along the path from a baseline (zero vector) to the input feature vector:
    $$IG_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + t(x - x'))}{\partial x_i} dt$$
*   **Performance Optimization**: To prevent CPU execution timeouts, explanations were run on local **2-hop receptive subgraphs** around the target nodes, reducing forward pass computation from 203,769 nodes to $\approx 3-440$ nodes. This yielded a **250x overall speedup** (running all 5 cases in 13 seconds instead of 55 minutes) while producing mathematically identical attribution weights.

---

## 📊 Summary of 5 Case Studies

The explainability script programmatically selected 5 distinct case-study nodes from the test split. The local neighborhood visualization plots and feature attribution comparison bar charts are saved in:
`reports/figures/explainability/`

### 1. Case TP (Node 79814) — True Positive (Correctly Identified Fraud)
*   **Label**: `1` (Illicit) | **Prediction**: `1` (Illicit, $p \ge 0.83$)
*   **Receptive Field**: 144 nodes, 332 edges.
*   **Neighborhood Insight**: 
    - The target node is located in a dense subgraph featuring multiple red nodes (known illicit transactions). 
    - GNNExplainer assigns the highest edge weights ($w \ge 0.85$) to its immediate links with neighboring illicit transactions.
*   **Feature Attribution**:
    - **Top Features**: `社区规模比率 (community_size_ratio)`, `pagerank`, `local_feat_1` (transaction size/fees).
    - **GNNExplainer vs. Captum**: Both methods identify `pagerank` and local transaction features as highly positive contributors. GNNExplainer (edge-aware) places greater emphasis on topological features (`pagerank`, `in_degree`), while Captum's IG highlights local transaction feature volumes.

### 2. Case TN (Node 118243) — True Negative (Correctly Identified Licit)
*   **Label**: `0` (Licit) | **Prediction**: `0` (Licit)
*   **Receptive Field**: 440 nodes, 974 edges.
*   **Neighborhood Insight**:
    - The target node is situated in a massive, clean cluster consisting exclusively of turquoise nodes (licit transactions). 
    - The edge weights are uniformly distributed ($w \approx 0.1-0.2$), showing that no single neighbor propagates anomalous flags.
*   **Feature Attribution**:
    - **Top Features**: `local_feat_2`, `aggregate_feat_15`, `in_degree` (low).
    - The low node degree and the dominant aggregate neighbor statistics of licit transactions drive the prediction towards $0.0$.

### 3. Case FP (Node 13571) — False Positive (False Alarm Debugging)
*   **Label**: `0` (Licit) | **Prediction**: `1` (Illicit, $p \ge 0.83$)
*   **Receptive Field**: 12 nodes, 22 edges.
*   **Neighborhood Insight**:
    - This is a licit transaction that was drawn into a high-risk neighborhood. It is connected to several unlabeled (grey) nodes that are topological neighbors of known fraud rings.
    - GNNExplainer highlights its link to an adjacent grey node as the primary driver ($w = 0.94$). 
*   **Debugging Takeaway**: The GNN misclassified this transaction because it aggregated signals from suspicious, unverified neighbors (guilt-by-association). Inspecting the unlabeled transaction neighbor reveals it to be a high-volume laundering hop.

### 4. Case FN (Node 1115) — False Negative (Missed Detection Debugging)
*   **Label**: `1` (Illicit) | **Prediction**: `0` (Licit)
*   **Receptive Field**: 3 nodes, 4 edges.
*   **Neighborhood Insight**:
    - The target node has an extremely small local neighborhood (degree = 1). Its only neighbor is a verified licit transaction (turquoise node).
*   **Debugging Takeaway**: Because GraphSAGE (Max) aggregates neighbor representations, if an illicit transaction is structurally isolated and transacts exclusively with licit entities, the neighborhood signal contains no anomalies. Thus, the model fails to detect it. This highlights the limitation of purely relational models in isolated, sparse subgraphs.

### 5. Case TP_HighDegree (Node 48874) — High-Degree True Positive
*   **Label**: `1` (Illicit) | **Prediction**: `1` (Illicit, $p \ge 0.83$)
*   **Receptive Field**: 86 nodes, 196 edges.
*   **Neighborhood Insight**:
    - The target node is a high-degree central hub in a large illicit cluster. 
    - GNNExplainer assigns maximum edge importance ($w = 1.0$) to the transaction pathways linking this node to multiple other illicit nodes, confirming that the model leverages the dense topological structure of the fraud ring.

---

## 📈 GNNExplainer vs. Captum Methodological Comparison

1.  **Structural Context**:
    *   **GNNExplainer** evaluates node features and edge indices jointly. By optimizing edge masks, it isolates the *receptive subgraph* that drives the prediction, allowing us to see *which pathways* are suspicious.
    *   **Captum (Integrated Gradients)** operates on node features only (without edge weights), treating the graph structure as a static background.
2.  **Topological Feature Sensitivity**:
    *   GNNExplainer is highly sensitive to topological features (PageRank, degrees) because they are directly linked to edge pathways.
    *   Captum's IG tends to emphasize local, continuous transaction features (amounts, fees) because gradients are computed directly with respect to node attributes.
3.  **Use Cases**:
    *   Use **GNNExplainer** to map out fraud rings, identify transaction pathways, and understand relational coordination.
    *   Use **Captum** for fast, local feature-attribution explanations to present to compliance officers explaining why an individual account was flagged based on its specific attributes.
