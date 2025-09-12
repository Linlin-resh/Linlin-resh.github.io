---
title: "Reading Notes: Newman's Networks Chapter 16 - Network Resilience"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 16 of Newman's 'Networks: An Introduction' covering network robustness, resilience measures, and failure analysis"
tags: ["reading-notes", "network-theory", "network-resilience", "robustness", "failure-analysis"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 16 of Newman's *Networks: An Introduction* explores **network resilience** - the ability of networks to maintain their functionality under stress, attacks, or failures. This chapter covers robustness measures, failure analysis, and strategies for enhancing network resilience.

## 16.1 Robustness Measures

### Connectivity Robustness

#### Robustness Function

**Robustness function**:
$$R(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where:
- $p$: Fraction of nodes removed
- $S_i(p)$: Size of largest component after removing $i$ nodes
- $n$: Total number of nodes

**Properties**:
- $R(0) = 1$: No nodes removed
- $R(1) = 0$: All nodes removed
- $R(p)$: Monotonic decreasing function

#### Critical Threshold

**Critical threshold** $p_c$:
$$R(p_c) = \frac{1}{2}$$

**Interpretation**: Fraction of nodes that can be removed before network loses half its functionality

**Mathematical condition**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$ is the degree ratio.

### Efficiency Robustness

#### Network Efficiency

**Global efficiency**:
$$E = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$$

Where $d_{ij}$ is the shortest path length between nodes $i$ and $j$.

**Local efficiency**:
$$E_{\text{local}} = \frac{1}{n} \sum_{i} E_i$$

Where $E_i$ is the efficiency of the subgraph of neighbors of node $i$.

#### Efficiency Robustness

**Efficiency robustness**:
$$R_E(p) = \frac{E(p)}{E(0)}$$

Where $E(p)$ is the efficiency after removing fraction $p$ of nodes.

**Properties**:
- $R_E(0) = 1$: No nodes removed
- $R_E(1) = 0$: All nodes removed
- $R_E(p)$: Monotonic decreasing function

### Spectral Robustness

#### Algebraic Connectivity

**Algebraic connectivity** $\lambda_2$:
$$\lambda_2 = \min_{x \perp \mathbf{1}} \frac{x^T L x}{x^T x}$$

Where $L$ is the Laplacian matrix.

**Properties**:
- $\lambda_2 > 0$: Network is connected
- $\lambda_2 = 0$: Network is disconnected
- Higher $\lambda_2$: More robust network

#### Spectral Robustness

**Spectral robustness**:
$$R_{\lambda}(p) = \frac{\lambda_2(p)}{\lambda_2(0)}$$

Where $\lambda_2(p)$ is the algebraic connectivity after removing fraction $p$ of nodes.

## 16.2 Attack Strategies

### Random Attacks

#### Random Node Removal

**Process**: Remove nodes randomly with probability $p$

**Robustness**: $R_{\text{random}}(p) = 1 - p$

**Critical threshold**: $p_c = 0.5$

**Mathematical analysis**:
$$R_{\text{random}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ nodes.

#### Random Edge Removal

**Process**: Remove edges randomly with probability $p$

**Robustness**: $R_{\text{random}}(p) = 1 - p$

**Critical threshold**: $p_c = 0.5$

**Mathematical analysis**:
$$R_{\text{random}}(p) = \frac{1}{m} \sum_{i=1}^m \frac{S_i(p)}{m}$$

Where $S_i(p)$ is the size of largest component after removing $i$ edges.

### Targeted Attacks

#### Degree-Based Attacks

**Process**: Remove nodes with highest degree

**Robustness**: $R_{\text{targeted}}(p) < R_{\text{random}}(p)$

**Critical threshold**: $p_c < 0.5$

**Mathematical analysis**:
$$R_{\text{targeted}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ highest-degree nodes.

#### Betweenness-Based Attacks

**Process**: Remove nodes with highest betweenness centrality

**Robustness**: $R_{\text{targeted}}(p) < R_{\text{random}}(p)$

**Critical threshold**: $p_c < 0.5$

**Mathematical analysis**:
$$R_{\text{targeted}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ highest-betweenness nodes.

#### Closeness-Based Attacks

**Process**: Remove nodes with highest closeness centrality

**Robustness**: $R_{\text{targeted}}(p) < R_{\text{random}}(p)$

**Critical threshold**: $p_c < 0.5$

**Mathematical analysis**:
$$R_{\text{targeted}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ highest-closeness nodes.

### Adaptive Attacks

#### Adaptive Degree-Based Attacks

**Process**: 
1. Remove node with highest degree
2. Recalculate degrees
3. Repeat

**Robustness**: $R_{\text{adaptive}}(p) < R_{\text{targeted}}(p)$

**Critical threshold**: $p_c < 0.5$

**Mathematical analysis**:
$$R_{\text{adaptive}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ nodes adaptively.

#### Adaptive Betweenness-Based Attacks

**Process**:
1. Remove node with highest betweenness
2. Recalculate betweenness
3. Repeat

**Robustness**: $R_{\text{adaptive}}(p) < R_{\text{targeted}}(p)$

**Critical threshold**: $p_c < 0.5$

**Mathematical analysis**:
$$R_{\text{adaptive}}(p) = \frac{1}{n} \sum_{i=1}^n \frac{S_i(p)}{n}$$

Where $S_i(p)$ is the size of largest component after removing $i$ nodes adaptively.

## 16.3 Failure Cascades

### Cascade Models

#### Load Redistribution Model

**Model**: When a node fails, its load is redistributed to neighbors

**Load evolution**:
$$L_i(t+1) = L_i(t) + \sum_{j \in \text{failed}} \frac{L_j(t)}{|\text{neighbors}(j)|}$$

Where:
- $L_i(t)$: Load of node $i$ at time $t$
- $\text{failed}$: Set of failed nodes
- $|\text{neighbors}(j)|$: Number of neighbors of node $j$

#### Threshold Model

**Model**: Node fails if its load exceeds threshold

**Failure condition**:
$$L_i(t) > T_i$$

Where $T_i$ is the threshold of node $i$.

**Cascade condition**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} > 2$$

Where $\langle k^2 \rangle$ and $\langle k \rangle$ are the second and first moments of the degree distribution.

### Cascade Analysis

#### Cascade Size

**Cascade size** $S$:
$$S = \frac{1}{n} \sum_{i=1}^n S_i$$

Where $S_i$ is the size of cascade starting from node $i$.

**Expected cascade size**:
$$\langle S \rangle = \frac{1}{n} \sum_{i=1}^n \langle S_i \rangle$$

#### Cascade Probability

**Cascade probability** $P$:
$$P = \frac{1}{n} \sum_{i=1}^n P_i$$

Where $P_i$ is the probability of cascade starting from node $i$.

**Expected cascade probability**:
$$\langle P \rangle = \frac{1}{n} \sum_{i=1}^n \langle P_i \rangle$$

## 16.4 Resilience Enhancement

### Redundancy

#### Edge Redundancy

**Edge redundancy**:
$$R_E = \frac{m - m_{\min}}{m_{\min}}$$

Where:
- $m$: Number of edges
- $m_{\min}$: Minimum number of edges for connectivity

**Properties**:
- $R_E \geq 0$: Always non-negative
- $R_E = 0$: No redundancy
- Higher $R_E$: More redundant network

#### Node Redundancy

**Node redundancy**:
$$R_N = \frac{n - n_{\min}}{n_{\min}}$$

Where:
- $n$: Number of nodes
- $n_{\min}$: Minimum number of nodes for functionality

**Properties**:
- $R_N \geq 0$: Always non-negative
- $R_N = 0$: No redundancy
- Higher $R_N$: More redundant network

### Diversity

#### Degree Diversity

**Degree diversity**:
$$D_k = \frac{\sigma_k}{\langle k \rangle}$$

Where:
- $\sigma_k$: Standard deviation of degrees
- $\langle k \rangle$: Average degree

**Properties**:
- $D_k \geq 0$: Always non-negative
- $D_k = 0$: No diversity
- Higher $D_k$: More diverse network

#### Path Diversity

**Path diversity**:
$$D_p = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{\sigma_{ij}}{\langle d_{ij} \rangle}$$

Where:
- $\sigma_{ij}$: Standard deviation of path lengths between $i$ and $j$
- $\langle d_{ij} \rangle$: Average path length between $i$ and $j$

**Properties**:
- $D_p \geq 0$: Always non-negative
- $D_p = 0$: No diversity
- Higher $D_p$: More diverse network

### Modularity

#### Modular Structure

**Modular structure**: Network divided into modules with dense internal connections and sparse external connections

**Modularity**:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$: Adjacency matrix
- $k_i, k_j$: Degrees of nodes $i, j$
- $c_i, c_j$: Community assignments
- $\delta(c_i, c_j)$: Kronecker delta

**Properties**:
- $Q \in [-1, 1]$: Bounded
- $Q > 0$: More modular than random
- $Q < 0$: Less modular than random

#### Resilience Benefits

**Modular networks**:
- **Localized failures**: Failures contained within modules
- **Faster recovery**: Modules can recover independently
- **Reduced cascades**: Failures less likely to spread

## 16.5 Applications to Materials Science

### Defect Networks

#### Defect Tolerance

**Defect network robustness**:
- **Random defects**: Random node removal
- **Clustered defects**: Targeted node removal
- **Critical defect concentration**: $c_c = \frac{1}{\kappa - 1}$

#### Defect Clustering

**Defect clustering**:
$$C = \frac{\langle k^2 \rangle - \langle k \rangle}{n \langle k \rangle^2}$$

**Resilience benefits**:
- **Localized damage**: Clustered defects cause localized damage
- **Faster healing**: Defects can heal locally
- **Reduced cascades**: Defect propagation limited

### Nanowire Networks

#### Electrical Robustness

**Electrical robustness**:
$$R_E = \frac{\sigma(p)}{\sigma(0)}$$

Where $\sigma(p)$ is the conductivity after removing fraction $p$ of nanowires.

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

#### Mechanical Robustness

**Mechanical robustness**:
$$R_M = \frac{E(p)}{E(0)}$$

Where $E(p)$ is the Young's modulus after removing fraction $p$ of nanowires.

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

### Phase Transitions

#### Phase Stability

**Phase stability**:
$$S = \frac{1}{n} \sum_{i=1}^n S_i$$

Where $S_i$ is the stability of phase at node $i$.

**Critical temperature**:
$$T_c = \frac{K \langle k^2 \rangle}{2 \langle k \rangle}$$

Where $K$ is the coupling strength.

#### Phase Transition Robustness

**Phase transition robustness**:
$$R_P = \frac{T_c(p)}{T_c(0)}$$

Where $T_c(p)$ is the critical temperature after removing fraction $p$ of nodes.

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

## Code Example: Network Resilience Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csgraph
from sklearn.metrics import pairwise_distances

def calculate_robustness_measures(G):
    """Calculate various robustness measures"""
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Connectivity robustness
    if nx.is_connected(G):
        connectivity_robustness = 1.0
    else:
        # Calculate largest component size
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        connectivity_robustness = len(largest_component) / n
    
    # Efficiency robustness
    if nx.is_connected(G):
        efficiency = nx.global_efficiency(G)
    else:
        # Calculate efficiency of largest component
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        subgraph = G.subgraph(largest_component)
        efficiency = nx.global_efficiency(subgraph)
    
    # Spectral robustness
    L = nx.laplacian_matrix(G).toarray()
    eigenvals = np.linalg.eigvals(L)
    eigenvals = np.real(eigenvals)
    eigenvals = np.sort(eigenvals)
    algebraic_connectivity = eigenvals[1] if len(eigenvals) > 1 else 0
    
    # Redundancy measures
    edge_redundancy = (m - (n - 1)) / (n - 1) if n > 1 else 0
    node_redundancy = (n - 1) / 1 if n > 1 else 0
    
    # Diversity measures
    degrees = [G.degree(i) for i in G.nodes()]
    degree_diversity = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0
    
    # Path diversity
    if nx.is_connected(G):
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        path_diversity = 0
        count = 0
        for i in G.nodes():
            for j in G.nodes():
                if i != j:
                    paths = path_lengths[i][j]
                    path_diversity += paths
                    count += 1
        path_diversity = path_diversity / count if count > 0 else 0
    else:
        path_diversity = 0
    
    return {
        'connectivity_robustness': connectivity_robustness,
        'efficiency': efficiency,
        'algebraic_connectivity': algebraic_connectivity,
        'edge_redundancy': edge_redundancy,
        'node_redundancy': node_redundancy,
        'degree_diversity': degree_diversity,
        'path_diversity': path_diversity
    }

def simulate_random_attack(G, p_values):
    """Simulate random attack on network"""
    
    n = G.number_of_nodes()
    results = []
    
    for p in p_values:
        # Remove random nodes
        n_remove = int(p * n)
        nodes_to_remove = np.random.choice(list(G.nodes()), n_remove, replace=False)
        
        # Create attacked network
        G_attacked = G.copy()
        G_attacked.remove_nodes_from(nodes_to_remove)
        
        # Calculate robustness measures
        measures = calculate_robustness_measures(G_attacked)
        measures['p'] = p
        results.append(measures)
    
    return results

def simulate_targeted_attack(G, p_values, attack_type='degree'):
    """Simulate targeted attack on network"""
    
    n = G.number_of_nodes()
    results = []
    
    for p in p_values:
        # Remove targeted nodes
        n_remove = int(p * n)
        
        if attack_type == 'degree':
            # Remove highest degree nodes
            degrees = [(i, G.degree(i)) for i in G.nodes()]
            degrees.sort(key=lambda x: x[1], reverse=True)
            nodes_to_remove = [i for i, _ in degrees[:n_remove]]
        elif attack_type == 'betweenness':
            # Remove highest betweenness nodes
            betweenness = nx.betweenness_centrality(G)
            nodes_to_remove = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)[:n_remove]
        elif attack_type == 'closeness':
            # Remove highest closeness nodes
            closeness = nx.closeness_centrality(G)
            nodes_to_remove = sorted(closeness.keys(), key=lambda x: closeness[x], reverse=True)[:n_remove]
        else:
            # Random attack
            nodes_to_remove = np.random.choice(list(G.nodes()), n_remove, replace=False)
        
        # Create attacked network
        G_attacked = G.copy()
        G_attacked.remove_nodes_from(nodes_to_remove)
        
        # Calculate robustness measures
        measures = calculate_robustness_measures(G_attacked)
        measures['p'] = p
        measures['attack_type'] = attack_type
        results.append(measures)
    
    return results

def simulate_cascade_failure(G, initial_failures, threshold_factor=1.5):
    """Simulate cascade failure in network"""
    
    n = G.number_of_nodes()
    
    # Initialize loads
    loads = {i: G.degree(i) for i in G.nodes()}
    thresholds = {i: loads[i] * threshold_factor for i in G.nodes()}
    
    # Initial failures
    failed_nodes = set(initial_failures)
    active_nodes = set(G.nodes()) - failed_nodes
    
    # Simulate cascade
    cascade_size = len(failed_nodes)
    cascade_steps = 0
    
    while True:
        new_failures = set()
        
        # Check each active node
        for node in active_nodes:
            if loads[node] > thresholds[node]:
                new_failures.add(node)
        
        # If no new failures, cascade stops
        if not new_failures:
            break
        
        # Update failed nodes
        failed_nodes.update(new_failures)
        active_nodes -= new_failures
        
        # Redistribute loads
        for failed_node in new_failures:
            neighbors = list(G.neighbors(failed_node))
            active_neighbors = [n for n in neighbors if n in active_nodes]
            
            if active_neighbors:
                load_per_neighbor = loads[failed_node] / len(active_neighbors)
                for neighbor in active_neighbors:
                    loads[neighbor] += load_per_neighbor
        
        cascade_size = len(failed_nodes)
        cascade_steps += 1
        
        # Prevent infinite loops
        if cascade_steps > n:
            break
    
    return {
        'cascade_size': cascade_size,
        'cascade_steps': cascade_steps,
        'final_failed_nodes': failed_nodes
    }

def analyze_network_resilience(G):
    """Comprehensive network resilience analysis"""
    
    # Calculate initial robustness measures
    initial_measures = calculate_robustness_measures(G)
    
    # Simulate different types of attacks
    p_values = np.linspace(0, 0.5, 20)
    
    random_results = simulate_random_attack(G, p_values)
    degree_results = simulate_targeted_attack(G, p_values, 'degree')
    betweenness_results = simulate_targeted_attack(G, p_values, 'betweenness')
    closeness_results = simulate_targeted_attack(G, p_values, 'closeness')
    
    # Simulate cascade failures
    cascade_results = []
    for i in range(10):  # Test 10 different initial failures
        initial_failures = np.random.choice(list(G.nodes()), 1, replace=False)
        cascade_result = simulate_cascade_failure(G, initial_failures)
        cascade_results.append(cascade_result)
    
    return {
        'initial_measures': initial_measures,
        'random_results': random_results,
        'degree_results': degree_results,
        'betweenness_results': betweenness_results,
        'closeness_results': closeness_results,
        'cascade_results': cascade_results
    }

def plot_resilience_analysis(G, analysis_results):
    """Plot network resilience analysis results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Attack robustness comparison
    p_values = [r['p'] for r in analysis_results['random_results']]
    
    # Connectivity robustness
    random_connectivity = [r['connectivity_robustness'] for r in analysis_results['random_results']]
    degree_connectivity = [r['connectivity_robustness'] for r in analysis_results['degree_results']]
    betweenness_connectivity = [r['connectivity_robustness'] for r in analysis_results['betweenness_results']]
    closeness_connectivity = [r['connectivity_robustness'] for r in analysis_results['closeness_results']]
    
    ax1.plot(p_values, random_connectivity, 'b-', label='Random', linewidth=2)
    ax1.plot(p_values, degree_connectivity, 'r-', label='Degree', linewidth=2)
    ax1.plot(p_values, betweenness_connectivity, 'g-', label='Betweenness', linewidth=2)
    ax1.plot(p_values, closeness_connectivity, 'm-', label='Closeness', linewidth=2)
    ax1.set_xlabel('Fraction of Nodes Removed')
    ax1.set_ylabel('Connectivity Robustness')
    ax1.set_title('Attack Robustness Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency robustness
    random_efficiency = [r['efficiency'] for r in analysis_results['random_results']]
    degree_efficiency = [r['efficiency'] for r in analysis_results['degree_results']]
    betweenness_efficiency = [r['efficiency'] for r in analysis_results['betweenness_results']]
    closeness_efficiency = [r['efficiency'] for r in analysis_results['closeness_results']]
    
    ax2.plot(p_values, random_efficiency, 'b-', label='Random', linewidth=2)
    ax2.plot(p_values, degree_efficiency, 'r-', label='Degree', linewidth=2)
    ax2.plot(p_values, betweenness_efficiency, 'g-', label='Betweenness', linewidth=2)
    ax2.plot(p_values, closeness_efficiency, 'm-', label='Closeness', linewidth=2)
    ax2.set_xlabel('Fraction of Nodes Removed')
    ax2.set_ylabel('Network Efficiency')
    ax2.set_title('Efficiency Robustness Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cascade failure analysis
    cascade_sizes = [r['cascade_size'] for r in analysis_results['cascade_results']]
    cascade_steps = [r['cascade_steps'] for r in analysis_results['cascade_results']]
    
    ax3.scatter(cascade_steps, cascade_sizes, alpha=0.7, s=100)
    ax3.set_xlabel('Cascade Steps')
    ax3.set_ylabel('Cascade Size')
    ax3.set_title('Cascade Failure Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Network visualization with robustness measures
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by degree
    degrees = [G.degree(i) for i in G.nodes()]
    node_colors = degrees
    node_sizes = 100 + 50 * np.array(degrees) / np.max(degrees)
    
    nx.draw(G, pos, ax=ax4, node_color=node_colors, node_size=node_sizes,
            edge_color='gray', alpha=0.6, cmap='viridis')
    ax4.set_title('Network with Degree-based Coloring')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example: Network resilience analysis
G = nx.barabasi_albert_graph(100, 3)

# Analyze network resilience
analysis_results = analyze_network_resilience(G)

# Print initial measures
print("Initial Network Robustness Measures:")
for measure, value in analysis_results['initial_measures'].items():
    print(f"{measure}: {value:.3f}")

# Print cascade analysis
cascade_sizes = [r['cascade_size'] for r in analysis_results['cascade_results']]
cascade_steps = [r['cascade_steps'] for r in analysis_results['cascade_results']]
print(f"\nCascade Failure Analysis:")
print(f"Average cascade size: {np.mean(cascade_sizes):.1f}")
print(f"Average cascade steps: {np.mean(cascade_steps):.1f}")

# Plot results
plot_resilience_analysis(G, analysis_results)
```

## Key Takeaways

1. **Robustness measures**: Multiple ways to quantify network resilience
2. **Attack strategies**: Random vs. targeted attacks have different effects
3. **Failure cascades**: Understanding how failures propagate through networks
4. **Resilience enhancement**: Strategies for improving network robustness
5. **Applications**: Important for materials science and engineering
6. **Mathematical analysis**: Rigorous theory for understanding resilience
7. **Practical implications**: Design principles for robust networks

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Albert, R., Jeong, H., & Barabási, A. L. (2000). Error and attack tolerance of complex networks. Nature, 406(6794), 378-382.
3. Callaway, D. S., et al. (2000). Network robustness and fragility: percolation on random graphs. Physical Review Letters, 85(25), 5468.
4. Holme, P., et al. (2002). Attack vulnerability of complex networks. Physical Review E, 65(5), 056109.

---

*Network resilience analysis provides crucial insights for understanding and improving the robustness of complex systems, with important applications in materials science and engineering.*
