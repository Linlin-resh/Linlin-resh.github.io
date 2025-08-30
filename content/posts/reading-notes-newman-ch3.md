---
title: "Reading Notes: Newman's Networks Chapter 3 - Random Graphs"
date: 2025-08-29
draft: false
description: "Key insights from Chapter 3 of Newman's 'Networks: An Introduction' on random graph theory and its applications"
tags: ["reading-notes", "network-theory", "random-graphs", "mathematics"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 3 of Newman's seminal work on networks introduces the fundamental concepts of **random graph theory**, a cornerstone in understanding complex network structures. This chapter provides the mathematical foundation for analyzing networks with stochastic properties.

## Key Concepts

### Erdős-Rényi Model

The classic Erdős-Rényi random graph model $G(n,p)$ consists of:
- **n** vertices
- **p** probability of edge existence between any two vertices

The expected number of edges is:
$$E[m] = \binom{n}{2}p = \frac{n(n-1)}{2}p$$

### Degree Distribution

For large networks, the degree distribution follows a **Poisson distribution**:
$$P(k) = \frac{\langle k \rangle^k e^{-\langle k \rangle}}{k!}$$

Where $\langle k \rangle = (n-1)p$ is the average degree.

## Phase Transitions

### Giant Component Emergence

A fascinating phenomenon occurs at the critical threshold $p_c = \frac{1}{n-1}$:
- **Below threshold**: Only small, isolated components exist
- **At threshold**: Giant component emerges
- **Above threshold**: Giant component dominates

The size of the giant component $S$ follows:
$$S \sim n^{2/3} \text{ at criticality}$$

## Applications to Materials Science

### Silver Nanowire Networks

Random graph theory helps model:
- **Percolation** in nanowire networks
- **Electrical conductivity** as a function of density
- **Critical density** for network connectivity

### Partially Disordered Systems

For materials with partial disorder:
- **Local order** can be modeled as clustered random graphs
- **Disorder parameters** affect network properties
- **Phase transitions** correspond to material property changes

## Code Example: Random Graph Generation

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_er_graph(n, p):
    """Generate Erdős-Rényi random graph"""
    G = nx.erdos_renyi_graph(n, p)
    return G

def analyze_giant_component(G):
    """Analyze giant component properties"""
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    return len(largest), len(components)

# Example usage
n, p = 1000, 0.01
G = generate_er_graph(n, p)
giant_size, num_components = analyze_giant_component(G)
print(f"Giant component size: {giant_size}")
print(f"Number of components: {num_components}")
```

## Conclusion

Random graph theory provides essential tools for understanding network structures in both theoretical and applied contexts. The phase transition behavior is particularly relevant for materials science applications where connectivity plays a crucial role.

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Erdős, P., & Rényi, A. (1959). On random graphs. Publicationes Mathematicae.

