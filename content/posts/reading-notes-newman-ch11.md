---
title: "Reading Notes: Newman's Networks Chapter 11 - Community Structure"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 11 of Newman's 'Networks: An Introduction' covering community detection methods, modularity optimization, and spectral clustering"
tags: ["reading-notes", "network-theory", "community-detection", "modularity", "spectral-clustering"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 11 of Newman's *Networks: An Introduction* focuses on **community structure** - the identification of densely connected groups of nodes within networks. This chapter covers various methods for detecting communities, their mathematical foundations, and applications to understanding network organization.

## 11.1 What is Community Structure?

### Definition

**Community structure** refers to the presence of groups of nodes that are:

- **Densely connected** within groups
- **Sparsely connected** between groups
- **Functionally related** or similar in some way

### Mathematical Framework

**Community detection** aims to partition the network into communities such that:

$$\max \sum_{c} \frac{e_c}{m} - \left(\frac{k_c}{2m}\right)^2$$

Where:
- $e_c$: Number of edges within community $c$
- $k_c$: Total degree of nodes in community $c$
- $m$: Total number of edges

## 11.2 Modularity

### Definition

**Modularity** measures the quality of a community partition:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$: Adjacency matrix
- $k_i, k_j$: Degrees of nodes $i, j$
- $c_i, c_j$: Community assignments
- $\delta(c_i, c_j)$: Kronecker delta

### Properties

**Range**: $Q \in [-1, 1]$

**Interpretation**:
- $Q > 0$: More edges within communities than expected by chance
- $Q = 0$: Random network structure
- $Q < 0$: Fewer edges within communities than expected

**Maximum modularity**:
$$Q_{\max} = 1 - \frac{1}{2m} \sum_{c} \frac{k_c^2}{2m}$$

### Resolution Limit

**Problem**: Modularity may not detect small communities

**Resolution limit**:
$$Q_{\text{max}} = 1 - \frac{1}{2m} \sum_{c} \frac{k_c^2}{2m}$$

**Small communities**: May not be detected if $k_c < \sqrt{2m}$

## 11.3 Modularity Optimization

### Greedy Algorithm

**Algorithm**:
1. Start with each node in its own community
2. Merge communities that increase modularity
3. Repeat until no improvement possible

**Modularity change**:
$$\Delta Q = \frac{1}{2m} \left[ 2e_{ij} - \frac{k_i k_j}{m} \right]$$

Where $e_{ij}$ is the number of edges between communities $i$ and $j$.

### Louvain Algorithm

**Two-phase algorithm**:

**Phase 1**: Local optimization
- For each node, move to community that maximizes modularity
- Repeat until no improvement

**Phase 2**: Community aggregation
- Merge communities into single nodes
- Repeat Phase 1

**Time complexity**: $O(m \log n)$

### Simulated Annealing

**Energy function**: $E = -Q$

**Temperature schedule**: $T(t) = T_0 e^{-\alpha t}$

**Acceptance probability**: $P = e^{-\Delta E/T}$

**Advantages**: Can escape local optima

## 11.4 Spectral Clustering

### Laplacian Matrix

**Unnormalized Laplacian**:
$$L = D - A$$

**Normalized Laplacian**:
$$L_{\text{norm}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Random walk Laplacian**:
$$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

### Eigenvalue Analysis

**Eigenvalue decomposition**:
$$L = U \Lambda U^T$$

**Properties**:
- **Smallest eigenvalue**: $\lambda_1 = 0$ (always)
- **Multiplicity of 0**: Number of connected components
- **Second smallest eigenvalue**: $\lambda_2 > 0$ if graph is connected

### Spectral Clustering Algorithm

**Algorithm**:
1. Compute Laplacian matrix $L$
2. Find $k$ smallest eigenvalues and eigenvectors
3. Use eigenvectors to cluster nodes

**Mathematical foundation**:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{subject to } \mathbf{x}^T \mathbf{1} = 0, \mathbf{x}^T \mathbf{x} = n$$

**Solution**: Fiedler vector (second eigenvector)

### Ratio Cut

**Ratio cut**:
$$\text{RatioCut}(S, T) = \frac{\text{cut}(S, T)}{|S|} + \frac{\text{cut}(S, T)}{|T|}$$

**Spectral relaxation**:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{subject to } \mathbf{x}^T \mathbf{1} = 0, \mathbf{x}^T \mathbf{1} = n$$

**Solution**: Fiedler vector

## 11.5 Girvan-Newman Algorithm

### Betweenness Centrality

**Edge betweenness**:
$$C_B(e) = \sum_{s \neq t} \frac{\sigma_{st}(e)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$: Number of shortest paths from $s$ to $t$
- $\sigma_{st}(e)$: Number of shortest paths from $s$ to $t$ passing through edge $e$

### Algorithm

**Girvan-Newman algorithm**:
1. Calculate betweenness centrality for all edges
2. Remove edge with highest betweenness
3. Recalculate betweenness for remaining edges
4. Repeat until no edges remain

**Hierarchical clustering**: Dendrogram shows community structure

### Time Complexity

**Betweenness calculation**: $O(nm)$ per edge
**Total complexity**: $O(nm^2)$

**Optimization**: Use approximation algorithms

## 11.6 Other Community Detection Methods

### Label Propagation

**Algorithm**:
1. Initialize each node with unique label
2. Each node adopts label of majority of neighbors
3. Repeat until convergence

**Convergence**: Guaranteed for connected graphs

**Time complexity**: $O(m)$

### Infomap

**Information-theoretic approach**:
$$\min \sum_{c} p_c \log p_c + \sum_{c} \frac{e_c}{m} \log \frac{e_c}{m}$$

**Random walk perspective**: Minimize description length of random walk

### Stochastic Block Model

**Model**:
$$P(A_{ij} = 1) = \theta_{c_i c_j}$$

Where $\theta_{c_i c_j}$ is the probability of edge between communities $c_i$ and $c_j$.

**Likelihood**:
$$L = \prod_{i<j} \theta_{c_i c_j}^{A_{ij}} (1 - \theta_{c_i c_j})^{1 - A_{ij}}$$

## 11.7 Applications to Materials Science

### Defect Clustering

**Network representation**:
- **Nodes**: Defect sites
- **Edges**: Defect interactions
- **Communities**: Clustered defect regions

**Mathematical framework**:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**Applications**:
- **Defect percolation**: Community structure affects percolation threshold
- **Material properties**: Clustered defects influence mechanical properties

### Phase Separation

**Network-based phase separation**:
- **Nodes**: Atomic sites
- **Edges**: Chemical bonds
- **Communities**: Different phases

**Order parameter**:
$$\phi = \frac{1}{N} \sum_{i=1}^N \delta(c_i, c_{\text{phase}})$$

**Phase transition**: Community structure changes at critical temperature

### Nanowire Networks

**Community structure** in nanowire networks:
- **Nodes**: Nanowire junctions
- **Edges**: Nanowire segments
- **Communities**: Dense nanowire regions

**Electrical properties**: Community structure affects conductivity

## 11.8 Evaluation Metrics

### Adjusted Rand Index

**Definition**:
$$ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2} \left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$

Where:
- $n_{ij}$: Number of nodes in community $i$ of partition $A$ and community $j$ of partition $B$
- $a_i$: Number of nodes in community $i$ of partition $A$
- $b_j$: Number of nodes in community $j$ of partition $B$

### Normalized Mutual Information

**Definition**:
$$NMI = \frac{2I(A, B)}{H(A) + H(B)}$$

Where:
- $I(A, B)$: Mutual information between partitions
- $H(A), H(B)$: Entropy of partitions

### Modularity

**Quality measure**: Higher modularity indicates better community structure

**Limitations**: Resolution limit, may not detect small communities

## Code Example: Community Detection

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict

def detect_communities_modularity(G):
    """Detect communities using modularity optimization"""
    
    # Greedy modularity optimization
    communities = nx.community.greedy_modularity_communities(G)
    modularity = nx.community.modularity(G, communities)
    
    return {
        'communities': communities,
        'modularity': modularity,
        'num_communities': len(communities)
    }

def detect_communities_spectral(G, n_communities=3):
    """Detect communities using spectral clustering"""
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    
    # Spectral clustering
    clustering = SpectralClustering(n_clusters=n_communities, 
                                  affinity='precomputed',
                                  random_state=42)
    labels = clustering.fit_predict(A)
    
    # Convert to community format
    communities = defaultdict(set)
    for i, label in enumerate(labels):
        communities[label].add(i)
    
    communities = list(communities.values())
    modularity = nx.community.modularity(G, communities)
    
    return {
        'communities': communities,
        'modularity': modularity,
        'num_communities': len(communities)
    }

def detect_communities_girvan_newman(G):
    """Detect communities using Girvan-Newman algorithm"""
    
    # Girvan-Newman algorithm
    communities = nx.community.girvan_newman(G)
    
    # Get the best partition (highest modularity)
    best_partition = None
    best_modularity = -1
    
    for partition in communities:
        modularity = nx.community.modularity(G, partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition
    
    return {
        'communities': best_partition,
        'modularity': best_modularity,
        'num_communities': len(best_partition)
    }

def evaluate_communities(G, communities, ground_truth=None):
    """Evaluate community detection results"""
    
    # Convert communities to labels
    labels = np.zeros(G.number_of_nodes())
    for i, community in enumerate(communities):
        for node in community:
            labels[node] = i
    
    # Calculate metrics
    results = {
        'modularity': nx.community.modularity(G, communities),
        'num_communities': len(communities),
        'avg_community_size': np.mean([len(c) for c in communities]),
        'community_size_std': np.std([len(c) for c in communities])
    }
    
    # Compare with ground truth if available
    if ground_truth is not None:
        ground_truth_labels = np.zeros(G.number_of_nodes())
        for i, community in enumerate(ground_truth):
            for node in community:
                ground_truth_labels[node] = i
        
        results['ari'] = adjusted_rand_score(ground_truth_labels, labels)
        results['nmi'] = normalized_mutual_info_score(ground_truth_labels, labels)
    
    return results

def plot_communities(G, communities, title="Community Detection"):
    """Plot network with community structure"""
    
    # Create color map for communities
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_colors = {}
    
    for i, community in enumerate(communities):
        color = colors[i]
        for node in community:
            node_colors[node] = color
    
    # Plot network
    pos = nx.spring_layout(G, k=1, iterations=50)
    node_color_list = [node_colors.get(node, 'lightblue') for node in G.nodes()]
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    nx.draw(G, pos, node_color=node_color_list, 
            edge_color='gray', alpha=0.6, node_size=50)
    plt.title(f'{title} (Modularity: {nx.community.modularity(G, communities):.3f})')
    plt.axis('off')
    
    # Community size distribution
    plt.subplot(2, 2, 2)
    community_sizes = [len(c) for c in communities]
    plt.hist(community_sizes, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Community Size')
    plt.ylabel('Count')
    plt.title('Community Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # Modularity vs number of communities
    plt.subplot(2, 2, 3)
    n_communities_range = range(2, min(20, G.number_of_nodes()))
    modularities = []
    
    for n_comm in n_communities_range:
        # Use spectral clustering for different numbers of communities
        A = nx.adjacency_matrix(G).toarray()
        clustering = SpectralClustering(n_clusters=n_comm, 
                                      affinity='precomputed',
                                      random_state=42)
        labels = clustering.fit_predict(A)
        
        # Convert to community format
        comm_dict = defaultdict(set)
        for i, label in enumerate(labels):
            comm_dict[label].add(i)
        comm_list = list(comm_dict.values())
        
        modularity = nx.community.modularity(G, comm_list)
        modularities.append(modularity)
    
    plt.plot(n_communities_range, modularities, 'bo-', markersize=6)
    plt.xlabel('Number of Communities')
    plt.ylabel('Modularity')
    plt.title('Modularity vs Number of Communities')
    plt.grid(True, alpha=0.3)
    
    # Network statistics
    plt.subplot(2, 2, 4)
    stats = {
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Communities': len(communities),
        'Modularity': nx.community.modularity(G, communities)
    }
    
    y_pos = np.arange(len(stats))
    plt.barh(y_pos, list(stats.values()), alpha=0.7)
    plt.yticks(y_pos, list(stats.keys()))
    plt.xlabel('Value')
    plt.title('Network Statistics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_community_methods(G):
    """Compare different community detection methods"""
    
    methods = {
        'Modularity': detect_communities_modularity,
        'Spectral': detect_communities_spectral,
        'Girvan-Newman': detect_communities_girvan_newman
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            if method_name == 'Spectral':
                # Try different numbers of communities
                best_result = None
                best_modularity = -1
                
                for n_comm in range(2, min(10, G.number_of_nodes())):
                    result = method_func(G, n_comm)
                    if result['modularity'] > best_modularity:
                        best_modularity = result['modularity']
                        best_result = result
                
                results[method_name] = best_result
            else:
                results[method_name] = method_func(G)
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = None
    
    return results

# Example: Community detection on a network
G = nx.karate_club_graph()  # Zachary's karate club

# Compare methods
comparison_results = compare_community_methods(G)

print("Community Detection Comparison:")
for method, result in comparison_results.items():
    if result is not None:
        print(f"\n{method}:")
        print(f"  Number of communities: {result['num_communities']}")
        print(f"  Modularity: {result['modularity']:.3f}")
        print(f"  Average community size: {np.mean([len(c) for c in result['communities']]):.1f}")

# Plot best result
best_method = max(comparison_results.keys(), 
                 key=lambda x: comparison_results[x]['modularity'] if comparison_results[x] else -1)
best_result = comparison_results[best_method]

if best_result is not None:
    plot_communities(G, best_result['communities'], f"Best Method: {best_method}")
```

## Key Takeaways

1. **Community structure**: Identifies densely connected groups within networks
2. **Modularity**: Measures quality of community partitions
3. **Detection methods**: Various algorithms with different strengths
4. **Spectral clustering**: Uses eigenvalues of Laplacian matrix
5. **Girvan-Newman**: Hierarchical method based on edge betweenness
6. **Applications**: Important for understanding network organization
7. **Evaluation**: Multiple metrics for assessing community quality

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Fortunato, S. (2010). Community detection in graphs. Physics Reports, 486(3-5), 75-174.
3. Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. Proceedings of the National Academy of Sciences, 99(12), 7821-7826.
4. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.

---

*Community detection provides insights into the modular organization of networks, with important applications in understanding materials structure and defect clustering.*
