---
title: "Reading Notes: Newman's Networks Chapter 15 - Multilayer Networks"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 15 of Newman's 'Networks: An Introduction' covering multilayer networks, multiplex networks, and their analysis"
tags: ["reading-notes", "network-theory", "multilayer-networks", "multiplex-networks", "network-analysis"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 15 of Newman's *Networks: An Introduction* explores **multilayer networks** - networks that consist of multiple layers, each representing different types of relationships or interactions. This chapter covers the mathematical framework, analysis methods, and applications of multilayer networks.

## 15.1 Multilayer Network Structure

### Definition

**Multilayer network** is a collection of networks with:

- **Nodes**: Can exist in multiple layers
- **Intra-layer edges**: Connections within each layer
- **Inter-layer edges**: Connections between layers
- **Layers**: Different types of relationships or time periods

### Mathematical Representation

#### Supra-adjacency Matrix

**Supra-adjacency matrix**:
$$A = \bigoplus_{\alpha} A^{\alpha} + \bigoplus_{\alpha \neq \beta} C^{\alpha \beta}$$

Where:
- $A^{\alpha}$: Adjacency matrix of layer $\alpha$
- $C^{\alpha \beta}$: Inter-layer coupling matrix
- $\bigoplus$: Direct sum operator

#### Tensor Representation

**Multilayer adjacency tensor**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
1 & \text{if } (i, \alpha) \text{ connected to } (j, \beta) \\
0 & \text{otherwise}
\end{cases}$$

**Properties**:
- **Dimensions**: $n \times n \times L \times L$
- **Symmetric**: $A_{ij}^{\alpha \beta} = A_{ji}^{\beta \alpha}$ for undirected networks
- **Sparse**: Most entries are zero

### Types of Multilayer Networks

#### Multiplex Networks

**Definition**: Each layer represents a different type of relationship

**Example**: Social network with layers for friendship, work, family

**Mathematical representation**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
A_{ij}^{\alpha} & \text{if } \alpha = \beta \\
0 & \text{if } \alpha \neq \beta
\end{cases}$$

#### Temporal Networks

**Definition**: Each layer represents a different time period

**Example**: Network evolution over time

**Mathematical representation**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
A_{ij}^{\alpha} & \text{if } \alpha = \beta \\
C_{ij}^{\alpha \beta} & \text{if } |\alpha - \beta| = 1 \\
0 & \text{otherwise}
\end{cases}$$

#### Multidimensional Networks

**Definition**: Each layer represents a different dimension

**Example**: Network with different interaction types

**Mathematical representation**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
A_{ij}^{\alpha} & \text{if } \alpha = \beta \\
C_{ij}^{\alpha \beta} & \text{if } \alpha \neq \beta
\end{cases}$$

## 15.2 Multilayer Network Measures

### Degree Measures

#### Multilayer Degree

**Multilayer degree** of node $i$:
$$k_i = \sum_{\alpha} k_i^{\alpha}$$

Where $k_i^{\alpha}$ is the degree of node $i$ in layer $\alpha$.

#### Overlapping Degree

**Overlapping degree** of node $i$:
$$o_i = \sum_{\alpha} \mathbb{I}(k_i^{\alpha} > 0)$$

Where $\mathbb{I}(\cdot)$ is the indicator function.

#### Participation Coefficient

**Participation coefficient**:
$$P_i = 1 - \sum_{\alpha} \left(\frac{k_i^{\alpha}}{k_i}\right)^2$$

**Properties**:
- $0 \leq P_i \leq 1$
- $P_i = 0$: Node active in only one layer
- $P_i = 1$: Node equally active in all layers

### Clustering Measures

#### Multilayer Clustering

**Multilayer clustering coefficient**:
$$C_i = \frac{\sum_{\alpha} C_i^{\alpha}}{L}$$

Where $C_i^{\alpha}$ is the clustering coefficient of node $i$ in layer $\alpha$.

#### Cross-layer Clustering

**Cross-layer clustering**:
$$C_i^{\alpha \beta} = \frac{2e_i^{\alpha \beta}}{k_i^{\alpha} k_i^{\beta}}$$

Where $e_i^{\alpha \beta}$ is the number of edges between neighbors of node $i$ in layers $\alpha$ and $\beta$.

### Centrality Measures

#### Multilayer PageRank

**Multilayer PageRank**:
$$PR_i^{\alpha} = (1-d) \frac{1}{nL} + d \sum_{j, \beta} \frac{A_{ij}^{\alpha \beta} PR_j^{\beta}}{k_j^{\beta}}$$

Where:
- $d$: Damping factor
- $n$: Number of nodes
- $L$: Number of layers

#### Multilayer Betweenness

**Multilayer betweenness centrality**:
$$C_B^{\alpha}(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}^{\alpha}(i)}{\sigma_{st}^{\alpha}}$$

Where $\sigma_{st}^{\alpha}(i)$ is the number of shortest paths from $s$ to $t$ in layer $\alpha$ passing through $i$.

## 15.3 Multilayer Network Analysis

### Community Detection

#### Multilayer Modularity

**Multilayer modularity**:
$$Q = \frac{1}{2\mu} \sum_{ij \alpha \beta} \left[ A_{ij}^{\alpha \beta} - \frac{k_i^{\alpha} k_j^{\beta}}{2m^{\alpha}} \right] \delta(c_i^{\alpha}, c_j^{\beta})$$

Where:
- $\mu = \sum_{i \alpha} k_i^{\alpha}$: Total number of edges
- $m^{\alpha} = \sum_i k_i^{\alpha}$: Number of edges in layer $\alpha$
- $c_i^{\alpha}$: Community assignment of node $i$ in layer $\alpha$

#### Multilayer Spectral Clustering

**Multilayer Laplacian**:
$$L^{\alpha \beta} = D^{\alpha \beta} - A^{\alpha \beta}$$

Where $D^{\alpha \beta}$ is the degree matrix.

**Eigenvalue decomposition**:
$$L = U \Lambda U^T$$

**Clustering**: Use first $k$ eigenvectors to cluster nodes

### Random Walks

#### Multilayer Random Walk

**Transition probability**:
$$P_{ij}^{\alpha \beta} = \frac{A_{ij}^{\alpha \beta}}{k_i^{\alpha}}$$

**Stationary distribution**:
$$\pi_i^{\alpha} = \frac{k_i^{\alpha}}{\sum_{j \beta} k_j^{\beta}}$$

#### Multilayer PageRank

**PageRank equation**:
$$\pi_i^{\alpha} = (1-d) \frac{1}{nL} + d \sum_{j \beta} P_{ji}^{\beta \alpha} \pi_j^{\beta}$$

**Solution**: Iterative method or eigenvalue problem

### Synchronization

#### Multilayer Kuramoto Model

**Phase dynamics**:
$$\frac{d\theta_i^{\alpha}}{dt} = \omega_i^{\alpha} + \frac{K}{nL} \sum_{j \beta} A_{ij}^{\alpha \beta} \sin(\theta_j^{\beta} - \theta_i^{\alpha})$$

Where:
- $\theta_i^{\alpha}$: Phase of node $i$ in layer $\alpha$
- $\omega_i^{\alpha}$: Natural frequency
- $K$: Coupling strength

#### Order Parameter

**Multilayer order parameter**:
$$r = \left| \frac{1}{nL} \sum_{i \alpha} e^{i\theta_i^{\alpha}} \right|$$

**Layer-specific order parameter**:
$$r^{\alpha} = \left| \frac{1}{n} \sum_{i} e^{i\theta_i^{\alpha}} \right|$$

## 15.4 Applications to Materials Science

### Multilayer Materials

#### Composite Materials

**Network representation**:
- **Layer 1**: Matrix material
- **Layer 2**: Filler material
- **Layer 3**: Interface region

**Inter-layer connections**: Phase boundaries

**Mathematical model**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
A_{ij}^{\alpha} & \text{if } \alpha = \beta \\
C_{ij}^{\alpha \beta} & \text{if } \alpha \neq \beta
\end{cases}$$

#### Nanocomposite Materials

**Network structure**:
- **Layer 1**: Carbon nanotubes
- **Layer 2**: Polymer matrix
- **Layer 3**: Interface interactions

**Properties**:
- **Electrical conductivity**: Depends on inter-layer connections
- **Mechanical strength**: Depends on intra-layer connections
- **Thermal conductivity**: Depends on both types of connections

### Phase Transitions

#### Multilayer Phase Transitions

**Network representation**:
- **Layer 1**: Crystalline phase
- **Layer 2**: Amorphous phase
- **Layer 3**: Defect phase

**Phase transition dynamics**:
$$\frac{d\phi_i^{\alpha}}{dt} = f_i^{\alpha}(\phi_i^{\alpha}) + \sum_{j \beta} A_{ij}^{\alpha \beta} g_{ij}^{\alpha \beta}(\phi_i^{\alpha}, \phi_j^{\beta})$$

Where $\phi_i^{\alpha}$ is the order parameter of node $i$ in layer $\alpha$.

#### Critical Behavior

**Critical temperature**:
$$T_c = \frac{K \langle k^2 \rangle}{\langle k \rangle}$$

Where $\langle k \rangle$ and $\langle k^2 \rangle$ are calculated across all layers.

### Defect Networks

#### Multilayer Defect Networks

**Network representation**:
- **Layer 1**: Point defects
- **Layer 2**: Line defects
- **Layer 3**: Surface defects

**Inter-layer connections**: Defect interactions

**Mathematical model**:
$$A_{ij}^{\alpha \beta} = \begin{cases} 
A_{ij}^{\alpha} & \text{if } \alpha = \beta \\
C_{ij}^{\alpha \beta} & \text{if } \alpha \neq \beta
\end{cases}$$

#### Defect Clustering

**Multilayer clustering**:
$$C_i = \frac{\sum_{\alpha} C_i^{\alpha}}{L}$$

**Cross-layer clustering**:
$$C_i^{\alpha \beta} = \frac{2e_i^{\alpha \beta}}{k_i^{\alpha} k_i^{\beta}}$$

## 15.5 Multilayer Network Models

### Random Multilayer Networks

#### Erdős-Rényi Multilayer

**Model**: Each layer is an Erdős-Rényi random graph

**Parameters**:
- $n$: Number of nodes
- $L$: Number of layers
- $p^{\alpha}$: Edge probability in layer $\alpha$
- $p^{\alpha \beta}$: Inter-layer edge probability

**Mathematical formulation**:
$$P(A_{ij}^{\alpha \beta} = 1) = \begin{cases} 
p^{\alpha} & \text{if } \alpha = \beta \\
p^{\alpha \beta} & \text{if } \alpha \neq \beta
\end{cases}$$

#### Scale-Free Multilayer

**Model**: Each layer follows preferential attachment

**Algorithm**:
1. Start with small initial network in each layer
2. Add nodes with preferential attachment
3. Add inter-layer edges with probability $p^{\alpha \beta}$

**Degree distribution**:
$$P(k) \sim k^{-\gamma}$$

Where $\gamma$ depends on the number of layers and inter-layer connections.

### Multilayer Community Models

#### Stochastic Block Model

**Model**: Nodes belong to communities across layers

**Parameters**:
- $c_i^{\alpha}$: Community of node $i$ in layer $\alpha$
- $\theta_{rs}^{\alpha \beta}$: Probability of edge between communities $r$ and $s$ in layers $\alpha$ and $\beta$

**Mathematical formulation**:
$$P(A_{ij}^{\alpha \beta} = 1) = \theta_{c_i^{\alpha} c_j^{\beta}}^{\alpha \beta}$$

#### Multilayer Modularity

**Modularity**:
$$Q = \frac{1}{2\mu} \sum_{ij \alpha \beta} \left[ A_{ij}^{\alpha \beta} - \frac{k_i^{\alpha} k_j^{\beta}}{2m^{\alpha}} \right] \delta(c_i^{\alpha}, c_j^{\beta})$$

**Optimization**: Maximize modularity to find communities

## Code Example: Multilayer Network Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix

class MultilayerNetwork:
    """Multilayer network class"""
    
    def __init__(self, n_nodes, n_layers):
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.layers = {}
        self.inter_layer_edges = defaultdict(list)
        
        # Initialize layers
        for layer in range(n_layers):
            self.layers[layer] = nx.Graph()
            for i in range(n_nodes):
                self.layers[layer].add_node(i)
    
    def add_intra_layer_edge(self, layer, u, v, weight=1):
        """Add edge within a layer"""
        self.layers[layer].add_edge(u, v, weight=weight)
    
    def add_inter_layer_edge(self, layer1, layer2, u, v, weight=1):
        """Add edge between layers"""
        self.inter_layer_edges[(layer1, layer2)].append((u, v, weight))
    
    def get_supra_adjacency_matrix(self):
        """Get supra-adjacency matrix"""
        n = self.n_nodes
        L = self.n_layers
        A = np.zeros((n * L, n * L))
        
        # Intra-layer edges
        for layer in range(L):
            G = self.layers[layer]
            for u, v in G.edges():
                weight = G[u][v].get('weight', 1)
                A[u * L + layer, v * L + layer] = weight
                A[v * L + layer, u * L + layer] = weight
        
        # Inter-layer edges
        for (layer1, layer2), edges in self.inter_layer_edges.items():
            for u, v, weight in edges:
                A[u * L + layer1, v * L + layer2] = weight
                A[v * L + layer2, u * L + layer1] = weight
        
        return A
    
    def get_multilayer_degree(self, node):
        """Get multilayer degree of a node"""
        degree = 0
        for layer in self.layers:
            degree += self.layers[layer].degree(node)
        return degree
    
    def get_overlapping_degree(self, node):
        """Get overlapping degree of a node"""
        overlapping = 0
        for layer in self.layers:
            if self.layers[layer].degree(node) > 0:
                overlapping += 1
        return overlapping
    
    def get_participation_coefficient(self, node):
        """Get participation coefficient of a node"""
        degrees = [self.layers[layer].degree(node) for layer in self.layers]
        total_degree = sum(degrees)
        
        if total_degree == 0:
            return 0
        
        participation = 1 - sum((d / total_degree) ** 2 for d in degrees)
        return participation
    
    def get_multilayer_clustering(self, node):
        """Get multilayer clustering coefficient of a node"""
        clusterings = []
        for layer in self.layers:
            if self.layers[layer].degree(node) > 1:
                clustering = nx.clustering(self.layers[layer], node)
                clusterings.append(clustering)
        
        if not clusterings:
            return 0
        
        return np.mean(clusterings)
    
    def get_multilayer_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        """Get multilayer PageRank"""
        n = self.n_nodes
        L = self.n_layers
        
        # Initialize PageRank values
        pr = np.ones(n * L) / (n * L)
        
        # Get supra-adjacency matrix
        A = self.get_supra_adjacency_matrix()
        
        # Calculate transition matrix
        P = np.zeros_like(A)
        for i in range(n * L):
            row_sum = np.sum(A[i, :])
            if row_sum > 0:
                P[i, :] = A[i, :] / row_sum
        
        # Iterate PageRank
        for _ in range(max_iter):
            pr_new = (1 - damping) / (n * L) + damping * np.dot(P.T, pr)
            
            if np.linalg.norm(pr_new - pr) < tol:
                break
            
            pr = pr_new
        
        # Reshape to node-layer format
        pr_reshaped = pr.reshape(n, L)
        return pr_reshaped
    
    def detect_communities(self, n_communities=3):
        """Detect communities using spectral clustering"""
        A = self.get_supra_adjacency_matrix()
        
        # Spectral clustering
        clustering = SpectralClustering(n_clusters=n_communities, 
                                      affinity='precomputed',
                                      random_state=42)
        labels = clustering.fit_predict(A)
        
        # Reshape to node-layer format
        labels_reshaped = labels.reshape(self.n_nodes, self.n_layers)
        return labels_reshaped
    
    def calculate_multilayer_modularity(self, communities):
        """Calculate multilayer modularity"""
        n = self.n_nodes
        L = self.n_layers
        
        # Calculate total number of edges
        mu = 0
        for layer in self.layers:
            mu += self.layers[layer].number_of_edges()
        
        # Calculate modularity
        Q = 0
        for i in range(n):
            for j in range(n):
                for alpha in range(L):
                    for beta in range(L):
                        # Get edge weight
                        if alpha == beta:
                            # Intra-layer edge
                            if self.layers[alpha].has_edge(i, j):
                                A_ij = 1
                            else:
                                A_ij = 0
                        else:
                            # Inter-layer edge
                            A_ij = 0
                            for (layer1, layer2), edges in self.inter_layer_edges.items():
                                if layer1 == alpha and layer2 == beta:
                                    for u, v, weight in edges:
                                        if u == i and v == j:
                                            A_ij = weight
                                            break
        
                        # Get degrees
                        k_i_alpha = self.layers[alpha].degree(i)
                        k_j_beta = self.layers[beta].degree(j)
                        
                        # Get community assignments
                        c_i_alpha = communities[i, alpha]
                        c_j_beta = communities[j, beta]
                        
                        # Calculate modularity contribution
                        if c_i_alpha == c_j_beta:
                            Q += A_ij - (k_i_alpha * k_j_beta) / (2 * mu)
        
        return Q / (2 * mu)

def generate_multilayer_network(n_nodes, n_layers, p_intra=0.1, p_inter=0.05):
    """Generate random multilayer network"""
    
    # Create multilayer network
    ml_net = MultilayerNetwork(n_nodes, n_layers)
    
    # Add intra-layer edges
    for layer in range(n_layers):
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < p_intra:
                    ml_net.add_intra_layer_edge(layer, i, j)
    
    # Add inter-layer edges
    for layer1 in range(n_layers):
        for layer2 in range(layer1+1, n_layers):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if np.random.random() < p_inter:
                        ml_net.add_inter_layer_edge(layer1, layer2, i, j)
    
    return ml_net

def analyze_multilayer_network(ml_net):
    """Analyze multilayer network properties"""
    
    n = ml_net.n_nodes
    L = ml_net.n_layers
    
    # Calculate measures for each node
    measures = {}
    for i in range(n):
        measures[i] = {
            'multilayer_degree': ml_net.get_multilayer_degree(i),
            'overlapping_degree': ml_net.get_overlapping_degree(i),
            'participation_coefficient': ml_net.get_participation_coefficient(i),
            'multilayer_clustering': ml_net.get_multilayer_clustering(i)
        }
    
    # Calculate layer-specific measures
    layer_measures = {}
    for layer in range(L):
        G = ml_net.layers[layer]
        layer_measures[layer] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'clustering': nx.average_clustering(G),
            'avg_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        }
    
    return measures, layer_measures

def plot_multilayer_analysis(ml_net, measures, layer_measures, communities):
    """Plot multilayer network analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Multilayer degree distribution
    degrees = [measures[i]['multilayer_degree'] for i in range(ml_net.n_nodes)]
    ax1.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Multilayer Degree')
    ax1.set_ylabel('Count')
    ax1.set_title('Multilayer Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Participation coefficient
    participation = [measures[i]['participation_coefficient'] for i in range(ml_net.n_nodes)]
    ax2.hist(participation, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Participation Coefficient')
    ax2.set_ylabel('Count')
    ax2.set_title('Participation Coefficient Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Layer comparison
    layers = list(layer_measures.keys())
    densities = [layer_measures[layer]['density'] for layer in layers]
    clusterings = [layer_measures[layer]['clustering'] for layer in layers]
    
    ax3.scatter(densities, clusterings, s=100, alpha=0.7)
    for i, layer in enumerate(layers):
        ax3.annotate(f'Layer {layer}', (densities[i], clusterings[i]))
    ax3.set_xlabel('Density')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Layer Properties')
    ax3.grid(True, alpha=0.3)
    
    # Community structure
    n_communities = len(np.unique(communities))
    community_sizes = []
    for comm in range(n_communities):
        size = np.sum(communities == comm)
        community_sizes.append(size)
    
    ax4.bar(range(n_communities), community_sizes, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Community')
    ax4.set_ylabel('Size')
    ax4.set_title('Community Size Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example: Multilayer network analysis
n_nodes, n_layers = 50, 3
ml_net = generate_multilayer_network(n_nodes, n_layers, p_intra=0.1, p_inter=0.05)

# Analyze network
measures, layer_measures = analyze_multilayer_network(ml_net)

# Detect communities
communities = ml_net.detect_communities(n_communities=3)

# Calculate modularity
modularity = ml_net.calculate_multilayer_modularity(communities)

# Get PageRank
pagerank = ml_net.get_multilayer_pagerank()

print("Multilayer Network Analysis:")
print(f"Number of nodes: {n_nodes}")
print(f"Number of layers: {n_layers}")
print(f"Modularity: {modularity:.3f}")

print("\nLayer Properties:")
for layer, props in layer_measures.items():
    print(f"Layer {layer}: {props['edges']} edges, density={props['density']:.3f}, clustering={props['clustering']:.3f}")

print("\nNode Properties (first 5 nodes):")
for i in range(min(5, n_nodes)):
    print(f"Node {i}: degree={measures[i]['multilayer_degree']}, participation={measures[i]['participation_coefficient']:.3f}")

# Plot analysis
plot_multilayer_analysis(ml_net, measures, layer_measures, communities)
```

## Key Takeaways

1. **Multilayer networks**: Networks with multiple layers of relationships
2. **Mathematical framework**: Supra-adjacency matrix and tensor representations
3. **Network measures**: Degree, clustering, and centrality measures for multilayer networks
4. **Community detection**: Spectral clustering and modularity optimization
5. **Random walks**: PageRank and other random walk methods
6. **Applications**: Important for materials science and complex systems
7. **Analysis methods**: Comprehensive tools for studying multilayer networks

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Kivelä, M., et al. (2014). Multilayer networks. Journal of Complex Networks, 2(3), 203-271.
3. Boccaletti, S., et al. (2014). The structure and dynamics of multilayer networks. Physics Reports, 544(1), 1-122.
4. De Domenico, M., et al. (2013). Mathematical formulation of multilayer networks. Physical Review X, 3(4), 041022.

---

*Multilayer networks provide a powerful framework for understanding complex systems with multiple types of relationships, with important applications in materials science and engineering.*
