---
title: "Reading Notes: Newman's Networks Chapter 5 - Biological Networks"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 5 of Newman's 'Networks: An Introduction' covering protein networks, metabolic networks, neural networks, and ecological networks"
tags: ["reading-notes", "network-theory", "biological-networks", "protein-networks", "metabolic-networks"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 5 of Newman's *Networks: An Introduction* explores **biological networks** - the complex webs of interactions that govern life at multiple scales. From molecular interactions to ecosystem dynamics, biological networks exhibit unique structural properties that reflect evolutionary pressures and functional constraints.

## 5.1 Protein-Protein Interaction Networks

### Network Structure

**Protein-protein interaction (PPI) networks** represent:

- **Nodes**: Proteins
- **Edges**: Physical interactions between proteins
- **Scale**: ~20,000 human proteins, ~100,000 interactions
- **Properties**: Undirected, weighted (interaction strength)

### Topological Properties

#### Degree Distribution

PPI networks exhibit **power-law degree distributions**:

$$P(k) \sim k^{-\gamma}$$

Where $\gamma \approx 2.0-2.5$ for most PPI networks.

#### Mathematical Analysis

**Cumulative distribution**:
$$P(K \geq k) \sim k^{-(\gamma-1)}$$

**Moment analysis**:
- **First moment**: $\langle k \rangle = \frac{2m}{n}$
- **Second moment**: $\langle k^2 \rangle = \int k^2 P(k) \, dk$

#### Scale-Free Behavior

**Scale-free networks** have:
- **Infinite variance** when $\gamma \leq 3$
- **Heavy-tailed distribution**
- **Hub proteins** with many interactions

### Functional Analysis

#### Essential Proteins

**Essentiality** correlates with network properties:

- **High-degree proteins**: More likely to be essential
- **High betweenness**: Critical for network connectivity
- **High clustering**: Part of functional modules

#### Mathematical Framework

**Essentiality probability**:
$$P(\text{essential}|k) = \frac{1}{1 + e^{-\alpha(k - k_0)}}$$

Where:
- $\alpha$: Sensitivity parameter
- $k_0$: Threshold degree

#### Gene Ontology Analysis

**Functional enrichment**:
$$P(\text{GO term}|k) = \frac{\text{Proteins with GO term and degree } k}{\text{Proteins with degree } k}$$

### Network Motifs

#### Common Motifs

**Three-node motifs**:
1. **Feed-forward loop**: A→B→C, A→C
2. **Feedback loop**: A→B→C→A
3. **Bi-fan**: A→B, A→C, D→B, D→C

**Mathematical counting**:
$$N_{\text{motif}} = \sum_{i,j,k} A_{ij} A_{jk} A_{ik}$$

#### Statistical Significance

**Z-score**:
$$Z = \frac{N_{\text{observed}} - \langle N_{\text{random}} \rangle}{\sigma_{N_{\text{random}}}}$$

**P-value**:
$$P = P(N_{\text{random}} \geq N_{\text{observed}})$$

## 5.2 Metabolic Networks

### Network Representation

**Metabolic networks** consist of:

- **Nodes**: Metabolites (small molecules)
- **Edges**: Biochemical reactions
- **Scale**: ~1,000-5,000 metabolites, ~2,000-10,000 reactions
- **Properties**: Bipartite (metabolites ↔ reactions)

### Bipartite Network Analysis

#### Projection Networks

**Metabolite-metabolite network**:
$$A_{ij} = \sum_r B_{ir} B_{jr}$$

Where $B_{ir} = 1$ if metabolite $i$ participates in reaction $r$.

**Reaction-reaction network**:
$$A_{rs} = \sum_i B_{ir} B_{is}$$

#### Mathematical Properties

**Degree distributions**:
- **Metabolite degrees**: Power-law with $\gamma \approx 2.0$
- **Reaction degrees**: More uniform distribution

### Flux Analysis

#### Flux Balance Analysis (FBA)

**Objective function**:
$$\max \sum_i v_i$$

**Constraints**:
$$\sum_j S_{ij} v_j = 0 \quad \text{(mass balance)}$$
$$v_j^{\min} \leq v_j \leq v_j^{\max} \quad \text{(flux bounds)}$$

Where:
- $v_j$: Flux through reaction $j$
- $S_{ij}$: Stoichiometric matrix
- $S_{ij} > 0$: Metabolite $i$ is produced by reaction $j$
- $S_{ij} < 0$: Metabolite $i$ is consumed by reaction $j$

#### Elementary Flux Modes

**Definition**: Minimal sets of reactions that can operate in steady state.

**Mathematical formulation**:
$$\mathbf{v} = \sum_k \alpha_k \mathbf{e}_k$$

Where $\mathbf{e}_k$ are elementary flux modes and $\alpha_k \geq 0$.

### Network Robustness

#### Knockout Analysis

**Single knockout**:
$$R_i = \frac{F_{\text{wt}} - F_{\text{ko}_i}}{F_{\text{wt}}}$$

Where:
- $F_{\text{wt}}$: Wild-type flux
- $F_{\text{ko}_i}$: Flux after knocking out reaction $i$

**Double knockout**:
$$R_{ij} = \frac{F_{\text{wt}} - F_{\text{ko}_{ij}}}{F_{\text{wt}}}$$

#### Synthetic Lethality

**Definition**: Two reactions are synthetically lethal if:
- Single knockouts are viable
- Double knockout is lethal

**Mathematical condition**:
$$R_i < \theta \text{ and } R_j < \theta \text{ but } R_{ij} > \theta$$

Where $\theta$ is the viability threshold.

## 5.3 Neural Networks

### Brain Connectivity

#### Structural Networks

**Neural networks** represent:

- **Nodes**: Neurons or brain regions
- **Edges**: Synaptic connections or white matter tracts
- **Scale**: ~86 billion neurons, ~100 trillion synapses
- **Properties**: Directed, weighted, dynamic

#### Network Properties

**Degree distributions**:
- **In-degree**: Input connections to a neuron
- **Out-degree**: Output connections from a neuron
- **Power-law**: $\gamma \approx 2.0-2.5$

**Clustering coefficient**:
$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

**Small-world properties**:
- **High clustering**: Local connectivity
- **Short path length**: Global efficiency

### Functional Networks

#### Correlation Networks

**Functional connectivity**:
$$r_{ij} = \frac{\langle x_i x_j \rangle - \langle x_i \rangle \langle x_j \rangle}{\sigma_i \sigma_j}$$

Where $x_i$ is the activity of region $i$.

**Network construction**:
$$A_{ij} = \begin{cases} 
1 & \text{if } |r_{ij}| > \theta \\
0 & \text{otherwise}
\end{cases}$$

#### Community Structure

**Modularity**:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**Functional modules**:
- **Visual cortex**: Visual processing
- **Motor cortex**: Motor control
- **Prefrontal cortex**: Executive functions

### Synchronization

#### Kuramoto Model

**Phase dynamics**:
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N A_{ij} \sin(\theta_j - \theta_i)$$

Where:
- $\theta_i$: Phase of oscillator $i$
- $\omega_i$: Natural frequency
- $K$: Coupling strength
- $A_{ij}$: Adjacency matrix

#### Synchronization Order Parameter

**Global order parameter**:
$$r = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j} \right|$$

**Local order parameter**:
$$r_i = \left| \frac{1}{k_i} \sum_{j \in \mathcal{N}_i} e^{i\theta_j} \right|$$

#### Critical Coupling

**Synchronization threshold**:
$$K_c = \frac{2}{\pi g(0)} \frac{\langle k^2 \rangle}{\langle k \rangle}$$

Where $g(0)$ is the frequency distribution at zero.

## 5.4 Ecological Networks

### Food Webs

#### Network Structure

**Food webs** represent:

- **Nodes**: Species
- **Edges**: Trophic interactions (predator-prey relationships)
- **Scale**: 10-1000 species, 100-10,000 interactions
- **Properties**: Directed, weighted (interaction strength)

#### Trophic Levels

**Trophic level** of species $i$:
$$TL_i = 1 + \frac{\sum_j A_{ji} TL_j}{\sum_j A_{ji}}$$

Where $A_{ji} = 1$ if species $j$ preys on species $i$.

#### Network Properties

**Connectance**:
$$C = \frac{L}{S(S-1)}$$

Where:
- $L$: Number of links
- $S$: Number of species

**Link density**:
$$LD = \frac{L}{S}$$

**Average path length**:
$$L = \frac{1}{S(S-1)} \sum_{i \neq j} d_{ij}$$

### Stability Analysis

#### May's Stability Criterion

**Random matrix theory**:
$$\lambda_{\max} = \sqrt{SC}$$

Where:
- $\lambda_{\max}$: Largest eigenvalue
- $S$: Number of species
- $C$: Connectance

**Stability condition**:
$$\lambda_{\max} < 1$$

#### Structural Stability

**Robustness** to species removal:
$$R = \frac{1}{S} \sum_{i=1}^S R_i$$

Where $R_i$ is the fraction of species remaining after removing species $i$.

### Biodiversity and Network Structure

#### Species-Area Relationships

**Power-law relationship**:
$$S = cA^z$$

Where:
- $S$: Number of species
- $A$: Area
- $c$: Constant
- $z$: Scaling exponent (typically 0.2-0.3)

#### Network Robustness

**Cascade effects**:
$$P(\text{extinction}|k) = 1 - (1-p)^k$$

Where $p$ is the probability of extinction due to one lost connection.

## 5.5 Applications to Materials Science

### Biomimetic Networks

#### Self-Assembling Materials

**Network formation** in self-assembling systems:

- **Block copolymers**: Form ordered network structures
- **Liquid crystals**: Create defect networks
- **Colloidal systems**: Generate percolating networks

**Mathematical modeling**:
$$P(\text{connection}) = \frac{1}{1 + e^{-\beta(E - \mu)}}$$

Where:
- $E$: Interaction energy
- $\mu$: Chemical potential
- $\beta$: Inverse temperature

#### Protein-Inspired Materials

**Network properties** of protein-inspired materials:

- **Hierarchical structure**: Multiple length scales
- **Self-healing**: Dynamic network reformation
- **Responsive behavior**: Network properties change with environment

### Network-Based Drug Design

#### Target Identification

**Network-based drug targets**:

- **Essential proteins**: High-degree nodes
- **Bottleneck proteins**: High betweenness centrality
- **Module hubs**: High clustering coefficient

**Mathematical framework**:
$$T_i = \alpha \cdot k_i + \beta \cdot C_B(i) + \gamma \cdot C_i$$

Where $T_i$ is the target score for protein $i$.

#### Drug Repurposing

**Network-based drug repurposing**:

- **Similarity networks**: Drug-drug similarity
- **Target networks**: Protein-protein interactions
- **Disease networks**: Disease-disease associations

## Code Example: Biological Network Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_biological_network(G, network_type="ppi"):
    """Analyze biological network properties"""
    
    # Basic statistics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    
    # Degree analysis
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    
    # Power-law fitting
    if len(degree_dist) > 1:
        k_values = list(degree_dist.keys())
        counts = list(degree_dist.values())
        
        if len(k_values) > 1:
            log_k = np.log(k_values[1:])  # Skip k=0
            log_counts = np.log(counts[1:])
            
            if len(log_k) > 1:
                gamma = -np.polyfit(log_k, log_counts, 1)[0]
            else:
                gamma = None
        else:
            gamma = None
    else:
        gamma = None
    
    # Clustering analysis
    clustering = nx.average_clustering(G)
    local_clustering = nx.clustering(G)
    
    # Path length analysis
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
    
    # Essentiality analysis (simplified)
    # High-degree nodes are more likely to be essential
    essentiality_scores = {}
    for node in G.nodes():
        k = G.degree(node)
        # Simple model: essentiality increases with degree
        essentiality_scores[node] = 1 / (1 + np.exp(-0.1 * (k - 10)))
    
    # Network motifs (simplified)
    # Count triangles
    triangles = sum(nx.triangles(G).values()) / 3
    
    # Community detection
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except:
        communities = None
        modularity = None
    
    return {
        'network_type': network_type,
        'nodes': n,
        'edges': m,
        'density': density,
        'avg_degree': np.mean(degrees),
        'gamma_estimate': gamma,
        'clustering': clustering,
        'avg_path_length': avg_path_length,
        'diameter': diameter,
        'triangles': triangles,
        'modularity': modularity,
        'num_communities': len(communities) if communities else None,
        'essentiality_scores': essentiality_scores
    }

def analyze_metabolic_network(metabolites, reactions):
    """Analyze metabolic network using bipartite representation"""
    
    # Create bipartite network
    B = nx.Graph()
    
    # Add metabolite nodes
    for metabolite in metabolites:
        B.add_node(metabolite, bipartite=0)
    
    # Add reaction nodes
    for reaction in reactions:
        B.add_node(reaction, bipartite=1)
    
    # Add edges (metabolite-reaction connections)
    for reaction, metabolite_list in reactions.items():
        for metabolite in metabolite_list:
            B.add_edge(metabolite, reaction)
    
    # Project to metabolite-metabolite network
    metabolite_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    G = nx.Graph()
    G.add_nodes_from(metabolite_nodes)
    
    # Add edges between metabolites that share reactions
    for metabolite1 in metabolite_nodes:
        for metabolite2 in metabolite_nodes:
            if metabolite1 != metabolite2:
                shared_reactions = set(B.neighbors(metabolite1)) & set(B.neighbors(metabolite2))
                if shared_reactions:
                    G.add_edge(metabolite1, metabolite2, weight=len(shared_reactions))
    
    # Analyze projected network
    results = analyze_biological_network(G, "metabolic")
    
    # Metabolic-specific metrics
    results['num_metabolites'] = len(metabolites)
    results['num_reactions'] = len(reactions)
    results['metabolite_degree_dist'] = dict(G.degree())
    
    return results

def plot_network_properties(G, title="Biological Network Properties"):
    """Plot various network properties"""
    
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
    clustering = nx.clustering(G)
    clustering_values = list(clustering.values())
    
    ax2.hist(clustering_values, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Clustering Coefficient')
    ax2.set_ylabel('Count')
    ax2.set_title('Clustering Coefficient Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    betweenness_values = list(betweenness.values())
    
    ax3.hist(betweenness_values, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Betweenness Centrality')
    ax3.set_ylabel('Count')
    ax3.set_title('Betweenness Centrality Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Network visualization (simplified)
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, ax=ax4, node_size=50, node_color='lightblue', 
            edge_color='gray', alpha=0.6)
    ax4.set_title('Network Visualization')
    ax4.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example: Analyze a protein-protein interaction network
G = nx.barabasi_albert_graph(1000, 3)
results = analyze_biological_network(G, "ppi")

print("Biological Network Analysis Results:")
for key, value in results.items():
    if key != 'essentiality_scores':
        print(f"{key}: {value}")

# Plot network properties
plot_network_properties(G, "Protein-Protein Interaction Network")
```

## Key Takeaways

1. **Scale-free structure**: Most biological networks exhibit power-law degree distributions
2. **Functional modules**: Biological networks show strong community structure
3. **Robustness**: Networks are robust to random failures but vulnerable to targeted attacks
4. **Evolutionary constraints**: Network structure reflects evolutionary pressures
5. **Multi-scale organization**: Biological networks operate at multiple length scales
6. **Dynamic behavior**: Network properties change over time
7. **Applications**: Network analysis provides insights into disease mechanisms and drug design

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Barabási, A. L., & Oltvai, Z. N. (2004). Network biology: understanding the cell's functional organization. Nature Reviews Genetics, 5(2), 101-113.
3. Sporns, O. (2011). Networks of the Brain. MIT Press.
4. Pimm, S. L. (1982). Food Webs. Chapman and Hall.

---

*Biological networks provide insights into the complex interactions that govern life, with important applications in understanding disease mechanisms and designing biomimetic materials.*
