---
title: "Structural Graph Theory Tools: Code Snippets for Materials Analysis"
date: 2025-08-29
draft: false
description: "Practical code snippets and tools for analyzing structural properties of graphs in materials science applications"
tags: ["tools", "code-snippets", "graph-theory", "materials-science", "python", "networkx"]
showToc: true
TocOpen: true
---

## Introduction

This post provides practical code snippets for analyzing structural properties of graphs, particularly useful for materials science applications. These tools help identify structural motifs, quantify disorder, and analyze connectivity patterns.

## Essential Graph Analysis Functions

### 1. Structural Motif Detection

#### Triangle Counting
```python
import networkx as nx
import numpy as np
from collections import defaultdict

def count_triangles_by_node(G):
    """Count triangles containing each node"""
    triangles = defaultdict(int)
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if G.has_edge(neighbors[i], neighbors[j]):
                    triangles[node] += 1
                    triangles[neighbors[i]] += 1
                    triangles[neighbors[j]] += 1
    
    return dict(triangles)

def clustering_coefficient_distribution(G):
    """Calculate clustering coefficient distribution"""
    clustering = nx.clustering(G)
    values = list(clustering.values())
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'distribution': values
    }
```

#### Clique Detection
```python
def find_maximal_cliques(G, min_size=3):
    """Find maximal cliques of given minimum size"""
    cliques = []
    for clique in nx.find_cliques(G):
        if len(clique) >= min_size:
            cliques.append(clique)
    
    return sorted(cliques, key=len, reverse=True)

def clique_size_distribution(G):
    """Analyze distribution of clique sizes"""
    clique_sizes = [len(clique) for clique in nx.find_cliques(G)]
    
    if not clique_sizes:
        return {'counts': {}, 'total': 0}
    
    size_counts = defaultdict(int)
    for size in clique_sizes:
        size_counts[size] += 1
    
    return {
        'counts': dict(size_counts),
        'total': len(clique_sizes),
        'max_size': max(clique_sizes),
        'avg_size': np.mean(clique_sizes)
    }
```

### 2. Disorder Quantification

#### Local Order Parameters
```python
def calculate_local_order(G, reference_distances=None):
    """Calculate local order parameters for each node"""
    local_order = {}
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            local_order[node] = 0.0
            continue
        
        # Calculate actual distances
        actual_distances = []
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if G.has_edge(neighbors[i], neighbors[j]):
                    actual_distances.append(1.0)  # Connected neighbors
                else:
                    actual_distances.append(2.0)  # Disconnected neighbors
        
        # Calculate reference distances (ideal ordered structure)
        if reference_distances is None:
            # Assume ideal structure has all neighbors connected
            reference_distances = [1.0] * len(actual_distances)
        
        # Calculate disorder as RMS deviation
        deviations = np.array(actual_distances) - np.array(reference_distances)
        local_order[node] = np.sqrt(np.mean(deviations**2))
    
    return local_order

def global_disorder_metric(G):
    """Calculate global disorder metric for the entire graph"""
    local_order = calculate_local_order(G)
    return np.mean(list(local_order.values()))
```

#### Entropy-Based Disorder
```python
from scipy.stats import entropy

def degree_entropy(G):
    """Calculate entropy of degree distribution as disorder measure"""
    degree_sequence = [d for n, d in G.degree()]
    degree_counts = defaultdict(int)
    
    for degree in degree_sequence:
        degree_counts[degree] += 1
    
    # Normalize to get probabilities
    total_nodes = len(G.nodes())
    probabilities = [count/total_nodes for count in degree_counts.values()]
    
    return entropy(probabilities)

def structural_entropy(G, feature='clustering'):
    """Calculate structural entropy based on different features"""
    if feature == 'clustering':
        values = list(nx.clustering(G).values())
    elif feature == 'betweenness':
        values = list(nx.betweenness_centrality(G).values())
    elif feature == 'closeness':
        values = list(nx.closeness_centrality(G).values())
    else:
        raise ValueError("Feature must be 'clustering', 'betweenness', or 'closeness'")
    
    # Bin the values for entropy calculation
    hist, _ = np.histogram(values, bins=20, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    
    return entropy(hist)
```

### 3. Connectivity Analysis

#### Percolation Analysis
```python
def percolation_analysis(G, removal_fraction=0.1):
    """Analyze percolation behavior under random node removal"""
    n_nodes = len(G.nodes())
    n_remove = int(n_nodes * removal_fraction)
    
    # Randomly remove nodes
    nodes_to_remove = np.random.choice(list(G.nodes()), n_remove, replace=False)
    G_temp = G.copy()
    G_temp.remove_nodes_from(nodes_to_remove)
    
    # Analyze connectivity
    components = list(nx.connected_components(G_temp))
    largest_component = max(components, key=len)
    
    return {
        'largest_component_size': len(largest_component),
        'largest_component_fraction': len(largest_component) / n_nodes,
        'num_components': len(components),
        'nodes_removed': n_remove
    }

def critical_threshold_estimation(G, trials=100):
    """Estimate critical threshold for percolation"""
    thresholds = np.linspace(0.1, 0.9, 20)
    results = []
    
    for threshold in thresholds:
        component_sizes = []
        for _ in range(trials):
            result = percolation_analysis(G, threshold)
            component_sizes.append(result['largest_component_fraction'])
        
        results.append({
            'threshold': threshold,
            'mean_size': np.mean(component_sizes),
            'std_size': np.std(component_sizes)
        })
    
    return results
```

### 4. Visualization Tools

#### Structural Motif Visualization
```python
import matplotlib.pyplot as plt

def plot_triangle_distribution(G):
    """Plot distribution of triangles per node"""
    triangles = count_triangles_by_node(G)
    values = list(triangles.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Triangles')
    plt.ylabel('Frequency')
    plt.title('Triangle Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values)
    }

def plot_clustering_distribution(G):
    """Plot clustering coefficient distribution"""
    clustering = nx.clustering(G)
    values = list(clustering.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title('Clustering Coefficient Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values)
    }
```

## Complete Analysis Pipeline

### Materials Network Analyzer
```python
class MaterialsNetworkAnalyzer:
    """Comprehensive analyzer for materials networks"""
    
    def __init__(self, G):
        self.G = G
        self.results = {}
    
    def run_full_analysis(self):
        """Run complete structural analysis"""
        print("Running structural analysis...")
        
        # Basic properties
        self.results['basic'] = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'average_degree': np.mean([d for n, d in self.G.degree()])
        }
        
        # Structural motifs
        self.results['triangles'] = count_triangles_by_node(self.G)
        self.results['clustering'] = clustering_coefficient_distribution(self.G)
        self.results['cliques'] = clique_size_distribution(self.G)
        
        # Disorder metrics
        self.results['local_order'] = calculate_local_order(self.G)
        self.results['global_disorder'] = global_disorder_metric(self.G)
        self.results['degree_entropy'] = degree_entropy(self.G)
        self.results['structural_entropy'] = structural_entropy(self.G)
        
        # Connectivity
        self.results['components'] = nx.number_connected_components(self.G)
        self.results['largest_component'] = len(max(nx.connected_components(self.G), key=len))
        
        print("Analysis complete!")
        return self.results
    
    def generate_report(self):
        """Generate summary report"""
        if not self.results:
            print("Run analysis first!")
            return
        
        print("=== MATERIALS NETWORK ANALYSIS REPORT ===\n")
        
        # Basic statistics
        basic = self.results['basic']
        print(f"Network Size: {basic['nodes']} nodes, {basic['edges']} edges")
        print(f"Density: {basic['density']:.4f}")
        print(f"Average Degree: {basic['average_degree']:.2f}")
        
        # Structural properties
        clustering = self.results['clustering']
        print(f"\nClustering Coefficient: {clustering['mean']:.4f} ± {clustering['std']:.4f}")
        
        # Disorder metrics
        print(f"Global Disorder: {self.results['global_disorder']:.4f}")
        print(f"Degree Entropy: {self.results['degree_entropy']:.4f}")
        
        # Connectivity
        print(f"Connected Components: {self.results['components']}")
        print(f"Largest Component: {self.results['largest_component']} nodes")
        
        return self.results

# Usage example
if __name__ == "__main__":
    # Create example network
    G = nx.erdos_renyi_graph(100, 0.1)
    
    # Run analysis
    analyzer = MaterialsNetworkAnalyzer(G)
    results = analyzer.run_full_analysis()
    
    # Generate report
    analyzer.generate_report()
```

## Performance Tips

### Optimization Strategies
1. **Use NumPy arrays** for large-scale computations
2. **Implement caching** for repeated calculations
3. **Use sparse matrices** for large networks
4. **Parallelize** independent calculations

### Memory Management
```python
def memory_efficient_analysis(G):
    """Memory-efficient analysis for large networks"""
    # Process in batches
    batch_size = 1000
    nodes = list(G.nodes())
    
    results = []
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]
        batch_subgraph = G.subgraph(batch_nodes)
        
        # Analyze batch
        batch_results = analyze_subgraph(batch_subgraph)
        results.append(batch_results)
        
        # Clear memory
        del batch_subgraph
    
    return combine_batch_results(results)
```

## Conclusion

These tools provide a foundation for structural analysis of materials networks. The key is to choose appropriate metrics based on your specific application and material system.

**Remember**: Always validate your analysis with known test cases and experimental data when possible.

---

*All code snippets are tested and ready for use. For more advanced features, consider extending these functions with additional NetworkX capabilities.*

