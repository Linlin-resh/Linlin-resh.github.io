---
title: "Reading Notes: Newman's Networks Chapter 7 - Network Measures and Metrics"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 7 of Newman's 'Networks: An Introduction' covering network measures, metrics, and their mathematical foundations"
tags: ["reading-notes", "network-theory", "metrics", "measures", "statistics"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 7 of Newman's *Networks: An Introduction* focuses on **network measures and metrics** - the quantitative tools used to characterize and compare networks. This chapter provides a comprehensive overview of the mathematical foundations and practical applications of network measurement.

## 7.1 Basic Network Measures

### Degree-Based Measures

#### Degree Distribution

**Definition**: The probability distribution of node degrees.

**Mathematical formulation**:
$$P(k) = \frac{\text{Number of nodes with degree } k}{n}$$

**Properties**:
- **Normalization**: $\sum_{k=0}^{\infty} P(k) = 1$
- **Average degree**: $\langle k \rangle = \sum_{k=0}^{\infty} k P(k)$
- **Second moment**: $\langle k^2 \rangle = \sum_{k=0}^{\infty} k^2 P(k)$

#### Degree Moments

**First moment (average degree)**:
$$\langle k \rangle = \frac{2m}{n}$$

**Second moment**:
$$\langle k^2 \rangle = \frac{1}{n} \sum_{i=1}^n k_i^2$$

**Degree variance**:
$$\sigma_k^2 = \langle k^2 \rangle - \langle k \rangle^2$$

**Coefficient of variation**:
$$CV = \frac{\sigma_k}{\langle k \rangle}$$

#### Degree Correlation

**Assortativity coefficient**:
$$r = \frac{\sum_{ij} (A_{ij} - \frac{k_i k_j}{2m}) k_i k_j}{\sum_{ij} (k_i \delta_{ij} - \frac{k_i k_j}{2m}) k_i k_j}$$

**Interpretation**:
- $r > 0$: Assortative (high-degree nodes connect to high-degree nodes)
- $r < 0$: Disassortative (high-degree nodes connect to low-degree nodes)
- $r = 0$: No correlation

### Clustering Measures

#### Local Clustering Coefficient

**Definition**:
$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

Where $e_i$ is the number of edges between neighbors of node $i$.

**Properties**:
- $0 \leq C_i \leq 1$
- $C_i = 1$: Complete subgraph
- $C_i = 0$: No triangles

#### Global Clustering Coefficient

**Average clustering**:
$$C = \frac{1}{n} \sum_{i=1}^n C_i$$

**Transitivity**:
$$T = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

**Relationship**:
$$C = T \quad \text{for undirected graphs}$$

#### Clustering Distribution

**Clustering coefficient distribution**:
$$P(C) = \frac{\text{Number of nodes with clustering } C}{n}$$

**Average clustering by degree**:
$$C(k) = \frac{1}{n_k} \sum_{i: k_i = k} C_i$$

Where $n_k$ is the number of nodes with degree $k$.

### Path Length Measures

#### Shortest Path Length

**Definition**: $d_{ij} = \min\{\text{length of path from } i \text{ to } j\}$

**Properties**:
- $d_{ii} = 0$ (self-distance)
- $d_{ij} = d_{ji}$ (symmetric for undirected graphs)
- $d_{ij} \leq d_{ik} + d_{kj}$ (triangle inequality)

#### Average Path Length

**Definition**:
$$L = \frac{1}{n(n-1)} \sum_{i \neq j} d_{ij}$$

**For disconnected graphs**:
$$L = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{d_{ij}}{R_{ij}}$$

Where $R_{ij} = 1$ if nodes $i$ and $j$ are connected, 0 otherwise.

#### Network Diameter

**Definition**: $D = \max_{i,j} d_{ij}$

**Effective diameter**: 90th percentile of path length distribution

**Mathematical formulation**:
$$D_{90} = \min\{d : P(d_{ij} \leq d) \geq 0.9\}$$

## 7.2 Centrality Measures

### Degree Centrality

**Definition**:
$$C_D(i) = \frac{k_i}{n-1}$$

**Normalized**: $0 \leq C_D(i) \leq 1$

**Interpretation**: Direct connections

### Betweenness Centrality

**Definition**:
$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$: Number of shortest paths from $s$ to $t$
- $\sigma_{st}(i)$: Number of shortest paths from $s$ to $t$ passing through $i$

**Normalized**:
$$C_B^{\text{norm}}(i) = \frac{C_B(i)}{(n-1)(n-2)/2}$$

**Weighted version**:
$$C_B^w(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}^w(i)}{\sigma_{st}^w} \cdot \frac{w_{st}}{w_{\max}}$$

### Closeness Centrality

**Definition**:
$$C_C(i) = \frac{n-1}{\sum_{j \neq i} d_{ij}}$$

**Normalized**: $0 \leq C_C(i) \leq 1$

**Harmonic closeness**:
$$C_H(i) = \sum_{j \neq i} \frac{1}{d_{ij}}$$

### Eigenvector Centrality

**Definition**:
$$x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j$$

**Matrix form**:
$$A \mathbf{x} = \lambda \mathbf{x}$$

**Solution**: Principal eigenvector of adjacency matrix

**Weighted version**:
$$x_i = \frac{1}{\lambda} \sum_j w_{ij} x_j$$

### Katz Centrality

**Definition**:
$$x_i = \alpha \sum_j A_{ij} x_j + \beta$$

**Matrix form**:
$$\mathbf{x} = \alpha A \mathbf{x} + \beta \mathbf{1}$$

**Solution**:
$$\mathbf{x} = \beta (I - \alpha A)^{-1} \mathbf{1}$$

**Convergence**: Requires $\alpha < \frac{1}{\rho(A)}$

### PageRank

**Definition**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d P^T \mathbf{PR}$$

**Matrix form**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M} \mathbf{PR}$$

Where $\mathbf{M}$ is the stochastic matrix:
$$M_{ij} = \frac{A_{ji}}{k_j^{\text{out}}}$$

## 7.3 Community Structure Measures

### Modularity

**Definition**:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$: Adjacency matrix
- $k_i, k_j$: Degrees of nodes $i, j$
- $c_i, c_j$: Community assignments
- $\delta(c_i, c_j)$: Kronecker delta

**Properties**:
- $Q \in [-1, 1]$
- $Q > 0$: More edges within communities than expected by chance
- $Q = 0$: Random network structure
- $Q < 0$: Fewer edges within communities than expected

### Modularity Resolution Limit

**Resolution limit**:
$$Q_{\text{max}} = 1 - \frac{1}{2m} \sum_{c} \frac{k_c^2}{2m}$$

Where $k_c$ is the total degree of community $c$.

**Small communities**: May not be detected if $k_c < \sqrt{2m}$

### Conductance

**Definition**:
$$\phi(S) = \frac{\text{cut}(S, \bar{S})}{\min(\text{vol}(S), \text{vol}(\bar{S}))}$$

Where:
- $\text{cut}(S, \bar{S})$: Number of edges between $S$ and $\bar{S}$
- $\text{vol}(S)$: Total degree of nodes in $S$

**Properties**:
- $0 \leq \phi(S) \leq 1$
- $\phi(S) = 0$: No edges between communities
- $\phi(S) = 1$: Maximum possible edges between communities

### Normalized Cut

**Definition**:
$$\text{NCut}(S, T) = \frac{\text{cut}(S, T)}{\text{vol}(S)} + \frac{\text{cut}(S, T)}{\text{vol}(T)}$$

**Spectral relaxation**:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{subject to } \mathbf{x}^T \mathbf{1} = 0, \mathbf{x}^T \mathbf{x} = n$$

## 7.4 Network Robustness Measures

### Connectivity Measures

#### Node Connectivity

**Definition**: Minimum number of nodes that must be removed to disconnect the graph.

**Mathematical formulation**:
$$\kappa(G) = \min_{S \subset V} |S| \quad \text{subject to } G \setminus S \text{ is disconnected}$$

#### Edge Connectivity

**Definition**: Minimum number of edges that must be removed to disconnect the graph.

**Mathematical formulation**:
$$\lambda(G) = \min_{F \subset E} |F| \quad \text{subject to } G \setminus F \text{ is disconnected}$$

#### Menger's Theorem

**Node connectivity**:
$$\kappa(G) = \min_{s,t} \text{number of node-disjoint paths from } s \text{ to } t$$

**Edge connectivity**:
$$\lambda(G) = \min_{s,t} \text{number of edge-disjoint paths from } s \text{ to } t$$

### Robustness to Attacks

#### Random Failure

**Robustness measure**:
$$R_{\text{random}} = \frac{1}{n} \sum_{i=1}^n \frac{S_i}{n}$$

Where $S_i$ is the size of the largest component after removing node $i$.

#### Targeted Attack

**Robustness measure**:
$$R_{\text{targeted}} = \frac{1}{n} \sum_{i=1}^n \frac{S_i^{(i)}}{n}$$

Where $S_i^{(i)}$ is the size of the largest component after removing the $i$-th most important node.

#### Attack Tolerance

**Tolerance threshold**:
$$p_c = 1 - \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$ is the degree ratio.

### Percolation Analysis

#### Bond Percolation

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

#### Site Percolation

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

## 7.5 Network Comparison Measures

### Structural Similarity

#### Graph Edit Distance

**Definition**: Minimum number of operations to transform one graph into another.

**Operations**:
- **Node insertion/deletion**: Cost $c_n$
- **Edge insertion/deletion**: Cost $c_e$
- **Node relabeling**: Cost $c_r$

**Mathematical formulation**:
$$d_{GED}(G_1, G_2) = \min_{\text{operations}} \sum_{i} c_i$$

#### Graph Isomorphism

**Definition**: Two graphs are isomorphic if there exists a bijection between their vertex sets that preserves adjacency.

**Mathematical formulation**:
$$G_1 \cong G_2 \iff \exists \phi : V_1 \to V_2 \text{ such that } (u,v) \in E_1 \iff (\phi(u), \phi(v)) \in E_2$$

### Statistical Similarity

#### Degree Distribution Similarity

**Kolmogorov-Smirnov test**:
$$D = \max_k |P_1(k) - P_2(k)|$$

**P-value**:
$$p = P(D \geq D_{\text{observed}})$$

#### Clustering Similarity

**Clustering coefficient difference**:
$$\Delta C = |C_1 - C_2|$$

**Clustering distribution similarity**:
$$D_C = \max_C |P_1(C) - P_2(C)|$$

### Network Alignment

#### Global Alignment

**Objective function**:
$$\max_{\phi} \sum_{i,j} A_{1,ij} A_{2,\phi(i)\phi(j)}$$

**Constraints**:
- $\phi$ is a bijection
- $A_{1,ij}$ is the adjacency matrix of graph 1
- $A_{2,\phi(i)\phi(j)}$ is the adjacency matrix of graph 2

#### Local Alignment

**Objective function**:
$$\max_{S_1, S_2} \sum_{i \in S_1, j \in S_2} A_{1,ij} A_{2,\phi(i)\phi(j)}$$

**Constraints**:
- $S_1 \subset V_1, S_2 \subset V_2$
- $|S_1| = |S_2|$
- $\phi : S_1 \to S_2$ is a bijection

## 7.6 Applications to Materials Science

### Network Descriptors for Materials

#### Structural Descriptors

**Coordination number distribution**:
$$P(k) = \frac{\text{Number of atoms with coordination } k}{n}$$

**Bond length distribution**:
$$P(r) = \frac{\text{Number of bonds with length } r}{m}$$

**Bond angle distribution**:
$$P(\theta) = \frac{\text{Number of bond angles } \theta}{n_{\text{angles}}}$$

#### Topological Descriptors

**Clustering coefficient**: Local atomic environment

**Path length**: Atomic connectivity

**Centrality measures**: Critical atoms/defects

### Property Prediction

#### Structure-Property Relationships

**Linear regression**:
$$P = \alpha_0 + \alpha_1 \langle k \rangle + \alpha_2 C + \alpha_3 L + \epsilon$$

**Nonlinear regression**:
$$P = f(\langle k \rangle, C, L, \ldots) + \epsilon$$

**Machine learning**:
$$P = \text{ML}(\text{network features}) + \epsilon$$

#### Feature Selection

**Correlation analysis**:
$$r_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sqrt{\text{Var}(X_i) \text{Var}(X_j)}}$$

**Mutual information**:
$$I(X_i; X_j) = \sum_{x_i, x_j} P(x_i, x_j) \log \frac{P(x_i, x_j)}{P(x_i) P(x_j)}$$

## Code Example: Network Measures

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

def compute_network_measures(G):
    """Compute comprehensive network measures"""
    
    # Basic statistics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    
    # Degree measures
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    
    # Degree moments
    avg_degree = np.mean(degrees)
    degree_variance = np.var(degrees)
    degree_std = np.std(degrees)
    coefficient_of_variation = degree_std / avg_degree if avg_degree > 0 else 0
    
    # Degree correlation (assortativity)
    assortativity = nx.degree_assortativity_coefficient(G)
    
    # Clustering measures
    local_clustering = nx.clustering(G)
    global_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    
    # Path length measures
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        # Analyze largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        diameter = nx.diameter(subgraph)
    
    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Connectivity measures
    node_connectivity = nx.node_connectivity(G)
    edge_connectivity = nx.edge_connectivity(G)
    
    # Community structure
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except:
        communities = None
        modularity = None
    
    return {
        'basic_stats': {
            'nodes': n,
            'edges': m,
            'density': density
        },
        'degree_measures': {
            'avg_degree': avg_degree,
            'degree_variance': degree_variance,
            'degree_std': degree_std,
            'coefficient_of_variation': coefficient_of_variation,
            'assortativity': assortativity
        },
        'clustering_measures': {
            'global_clustering': global_clustering,
            'transitivity': transitivity,
            'local_clustering': local_clustering
        },
        'path_measures': {
            'avg_path_length': avg_path_length,
            'diameter': diameter
        },
        'centrality_measures': {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality
        },
        'connectivity_measures': {
            'node_connectivity': node_connectivity,
            'edge_connectivity': edge_connectivity
        },
        'community_measures': {
            'modularity': modularity,
            'num_communities': len(communities) if communities else None
        }
    }

def compare_networks(G1, G2):
    """Compare two networks using various measures"""
    
    # Compute measures for both networks
    measures1 = compute_network_measures(G1)
    measures2 = compute_network_measures(G2)
    
    # Compare basic statistics
    basic_comparison = {}
    for key in measures1['basic_stats']:
        val1 = measures1['basic_stats'][key]
        val2 = measures2['basic_stats'][key]
        basic_comparison[key] = {
            'G1': val1,
            'G2': val2,
            'difference': abs(val1 - val2),
            'relative_difference': abs(val1 - val2) / max(val1, val2) if max(val1, val2) > 0 else 0
        }
    
    # Compare degree distributions
    degrees1 = [d for n, d in G1.degree()]
    degrees2 = [d for n, d in G2.degree()]
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(degrees1, degrees2)
    
    # Compare clustering distributions
    clustering1 = list(nx.clustering(G1).values())
    clustering2 = list(nx.clustering(G2).values())
    
    clustering_ks_statistic, clustering_ks_pvalue = stats.ks_2samp(clustering1, clustering2)
    
    return {
        'basic_comparison': basic_comparison,
        'degree_distribution_ks': {
            'statistic': ks_statistic,
            'pvalue': ks_pvalue
        },
        'clustering_distribution_ks': {
            'statistic': clustering_ks_statistic,
            'pvalue': clustering_ks_pvalue
        }
    }

def plot_network_measures(G, title="Network Measures"):
    """Plot various network measures"""
    
    measures = compute_network_measures(G)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    k_values = list(degree_dist.keys())
    counts = list(degree_dist.values())
    
    ax1.loglog(k_values, counts, 'bo', markersize=8)
    ax1.set_xlabel('Degree k')
    ax1.set_ylabel('Count')
    ax1.set_title('Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Clustering coefficient distribution
    clustering = list(measures['clustering_measures']['local_clustering'].values())
    ax2.hist(clustering, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Clustering Coefficient')
    ax2.set_ylabel('Count')
    ax2.set_title('Clustering Coefficient Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Centrality measures comparison
    centrality_measures = measures['centrality_measures']
    centrality_names = list(centrality_measures.keys())
    centrality_values = [list(values.values()) for values in centrality_measures.values()]
    
    # Plot top 10 nodes for each centrality measure
    for i, (name, values) in enumerate(centrality_measures.items()):
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, scores = zip(*sorted_values)
        ax3.plot(nodes, scores, 'o-', label=name, markersize=6)
    
    ax3.set_xlabel('Node Rank')
    ax3.set_ylabel('Centrality Score')
    ax3.set_title('Centrality Measures Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Network visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, ax=ax4, node_size=50, node_color='lightblue', 
            edge_color='gray', alpha=0.6)
    ax4.set_title('Network Visualization')
    ax4.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example: Analyze a scale-free network
G = nx.barabasi_albert_graph(1000, 3)
measures = compute_network_measures(G)

print("Network Measures Analysis:")
for category, values in measures.items():
    print(f"\n{category.upper()}:")
    for key, value in values.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} values")
        else:
            print(f"  {key}: {value}")

# Plot network measures
plot_network_measures(G, "Scale-Free Network Measures")
```

## Key Takeaways

1. **Comprehensive measurement**: Network measures capture different aspects of network structure
2. **Mathematical foundations**: Each measure has a solid mathematical basis
3. **Comparative analysis**: Measures enable quantitative network comparison
4. **Robustness assessment**: Network measures help evaluate system robustness
5. **Community detection**: Modularity and related measures identify community structure
6. **Centrality importance**: Different centrality measures capture different aspects of importance
7. **Applications**: Network measures are essential for materials science applications

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Wasserman, S., & Faust, K. (1994). Social Network Analysis: Methods and Applications. Cambridge University Press.
3. Brandes, U., & Erlebach, T. (2005). Network Analysis: Methodological Foundations. Springer.
4. Costa, L. da F., et al. (2007). Characterization of complex networks: A survey of measurements. Advances in Physics, 56(1), 167-242.

---

*Network measures and metrics provide the quantitative tools needed to characterize, compare, and analyze complex networks, with important applications in materials science and engineering.*
