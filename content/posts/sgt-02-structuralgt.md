---
title: "Structural Graph Theory II: StructuralGT — Reading Nanomaterial Networks Like Graphs"
date: 2026-04-05
tags: ["graph-theory", "structural-graph-theory", "StructuralGT", "nanowire", "AgNW", "materials-science", "percolation", "python"]
draft: false
description: "How do you measure the structure of a disordered nanomaterial? StructuralGT converts microscopy images of fiber networks into graphs and computes 13 graph-theoretic parameters — connecting network topology to electrical, optical, and mechanical properties."
math: true
---

> *This is the second post in the **Structural Graph Theory** series. [Part I](../sgt-01-planar-minor/) covered planar graphs, graph minors, and treewidth. Here we move from theory to experiment: how do you actually measure the graph structure of a real material?*

---

## 1. The Problem: Structure Without Periodicity

Crystal structures have unit cells. Amorphous and disordered materials don't. How do you describe the structure of a network material — silver nanowires, aramid nanofibers, carbon nanotube films — when there is no repeating pattern?

Traditional descriptors (lattice parameters, coordination shells, pair distribution functions) assume local periodicity. They fail for **percolating nanoscale networks (PNNs)**: materials where extended connectivity, not local order, governs the macroscopic properties.

The key insight of two papers from Kotov's group at the University of Michigan is:

> **Treat the network as a graph. Measure the graph.**

This sounds straightforward. The implementation — automatically extracting a graph from a microscopy image of a disordered nanofiber network, then computing topologically meaningful parameters — required building a new open-source tool: **StructuralGT**.

---

## 2. StructuralGT: From Micrograph to Graph

**StructuralGT** is a Python package (and GUI application) developed by Vecchio, Mahler, Hammig, and Kotov at UMich COMPASS Lab, first described in:

> Vecchio, D. A., Mahler, S. H., Hammig, M. D., & Kotov, N. A. (2021). Structural Analysis of Nanoscale Network Materials Using Graph Theory. *ACS Nano*, **15**(8), 12847–12859. [https://doi.org/10.1021/acsnano.1c04711](https://doi.org/10.1021/acsnano.1c04711) ✅

The package is openly available:
- PyPI: `pip install StructuralGT`
- GitHub: [compass-stc/StructuralGT](https://github.com/compass-stc/StructuralGT)
- Docs: [structuralgt.readthedocs.io](https://structuralgt.readthedocs.io)

### 2.1 Image-to-Graph Pipeline

The conversion from micrograph to graph follows a standard computer vision + graph extraction pipeline:

```
Raw Micrograph (SEM/TEM/AFM/confocal)
        ↓ 
   Preprocessing
   (crop, denoise: median filter + Gaussian blur,
    contrast adjustment, threshold: global/Otsu/adaptive)
        ↓
   Binary Image
   (fibers = white, background = black)
        ↓
   Skeletonization
   (reduce fibers to 1-pixel-wide skeleton)
        ↓
   Node Detection
   (branch points = nodes, endpoints = leaf nodes)
        ↓
   Edge Tracing
   (fiber segments between nodes = edges)
        ↓
   Post-processing
   (remove disconnected fragments, prune jagged edges,
    merge nearby nodes, remove dangling edges)
        ↓
   Graph G = (V, E)
   (spatial coordinates + edge weights available)
```

The geometric interpretation is physically meaningful:
- **Nodes** = fiber branch points (junctions) or fiber ends
- **Edges** = fiber segments between junctions
- **Edge weight** = physical length of fiber segment, or computed conductance

### 2.2 The 13 Graph-Theoretic Parameters

Once the graph $G = (V, E)$ is extracted, StructuralGT computes 13 structural parameters:

| Parameter | Symbol | Physical Meaning |
|:---|:---|:---|
| Degree distribution | $\{k_v\}$ | How many fibers meet at each junction |
| Graph density | $\rho = 2\|E\|/(\|V\|(\|V\|-1))$ | Overall connectivity fraction |
| Network diameter | $d = \max_{u,v} d(u,v)$ | Longest shortest path |
| Global efficiency | $E_{glob} = \frac{1}{\|V\|(\|V\|-1)}\sum_{u \neq v} \frac{1}{d(u,v)}$ | Transport efficiency |
| Wiener index | $W = \sum_{u < v} d(u,v)$ | Total path length (related to diffusion) |
| Clustering coefficient | $C = \frac{1}{\|V\|}\sum_v C_v$ | Local loop density |
| Average nodal connectivity | $\kappa$ | Minimum vertex cuts on average |
| Assortativity coefficient | $r$ | Degree–degree correlation |
| Betweenness centrality | $BC_v = \sum_{s \neq v \neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}$ | Bottleneck vertices in transport |
| Closeness centrality | $CC_v = \frac{\|V\|-1}{\sum_u d(u,v)}$ | Accessibility of a node |
| Eigenvector centrality | $EC_v$ | Influence weighted by neighbor influence |
| *(+ 2 additional geometric/spatial parameters)* | | |

**Key structure–property connections identified:**

- **Average nodal connectivity** $\kappa$ ↔ mechanical toughness and shear resistance (network robustness)
- **Clustering coefficient** $C$ ↔ compressive/tensile strength (local load redistribution)
- **Global efficiency** $E_{glob}$ ↔ electrical conductivity and ion transport
- **Betweenness centrality** distribution ↔ current concentration and failure modes
- **Assortativity** $r$ ↔ network resilience (positive $r$: high-degree hubs connect to hubs; negative $r$: hub-spoke topology)

---

## 3. Matter 2024: Graph Theory Predicts What Microscopy Cannot

The power of this approach became fully clear in a 2024 *Matter* paper:

> Wu, W., Kadar, A., Lee, S. H., Glotzer, S. C., Goss, V., Kotov, N. A., et al. (2024). Layer-by-layer assembled nanowire networks enable graph-theoretical design of multifunctional coatings. *Matter*, **7**(10). [https://doi.org/10.1016/j.matt.2024.09.014](https://doi.org/10.1016/j.matt.2024.09.014) ✅

### 3.1 The Material: LBL Silver and Gold Nanowire Films

**Layer-by-layer (LBL) assembly** is a thin film fabrication technique where charged species are deposited in alternating layers:

1. Spray polyethyleneimine (PEI, positively charged) onto substrate
2. Rinse and dry
3. Spray AgNW or AuNW suspension (negatively charged)
4. Rinse: loosely bound NWs removed
5. Repeat → each LBL cycle adds one nanowire layer

Key structural feature: NWs are confined to the $x$–$y$ plane within each layer, but **layers stack in $z$**. Inter-layer NW–NW junctions also conduct. The result is a **quasi-2D multilayer nanowire network** with both in-plane and cross-plane conduction paths.

### 3.2 The Central Question: Why Does Random Stick Model Fail?

The standard computational model for NW networks is the **random stick model (RSM)**: place straight sticks of fixed length at random positions and orientations in a box, define conductance for each stick–stick junction, solve Kirchhoff's equations.

RSM is tractable and analytically connected to percolation theory. But Wu et al. showed it is **structurally wrong** for LBL films:

| Parameter | RSM | Real AgNW LBL | Ratio |
|:---|:---|:---|:---|
| Average clustering coefficient (ACC) | ~2× higher | — | RSM overestimates |
| Average betweenness centrality | ~3× higher (for AuNW) | — | RSM overestimates |

The reason: RSM assumes **random (Poissonian) spatial disorder**. LBL nanowires have **correlated, non-random disorder** — nanowires align slightly during spray deposition, creating anisotropy and excluded-volume effects not captured by RSM.

**Consequence:** Matching RSM sheet resistance to experiment by adjusting parameters gives the wrong structural description — and wrong predictions for all other properties.

### 3.3 Image-Informed GT Models: What They Predict

By using SEM/TEM/AFM images of actual LBL films as input to StructuralGT, and modeling multilayer films as **vertically stacked single-layer graphs** (with tunable inter-layer connectivity fraction as the single fitting parameter), the GT model correctly predicts:

**Electrical properties:**
- Sheet resistance vs. number of LBL cycles $N$
- **Nonlinear** charge transport vs. $N$ — an unexpected finding that RSM misses
- Conductivity **anisotropy** (different $\sigma_x$, $\sigma_y$) — not visible by microscopy alone, but correctly predicted by graph Laplacian analysis
- Current-carrying capacity (failure current) as a function of $N$

**Optical properties:**
- Optical anisotropy (polarization-dependent absorption)
- THz absorption
- Optical rotation

**Mechanical properties:**
- Elastic modulus estimates
- Surface roughness

The **graph Laplacian pseudoinverse** $L^+$ is the central computational tool for electrical properties:

$$\sigma \propto (L^+)_{ij}$$

where conductance between terminals $i$ and $j$ is computed via the Moore–Penrose pseudoinverse of the weighted graph Laplacian $L = D - A$ (with $D$ the degree matrix and $A$ the weighted adjacency matrix).

### 3.4 Scale-Up: From Lab Sample to Drone Wing

GT-optimized LBL coatings were spray-deposited at **meter scale** on curved drone wings. The coating provides:
- **Lightning protection** (electrical conductivity)
- **De-icing** capability (joule heating)
- **Stealth** properties (THz absorption)
- **Optical anisotropy** for sensing

The GT design framework — extract graph from image, compute parameters, predict properties, optimize composition — was directly transferable from millimeter-scale laboratory samples to meter-scale aeronautical surfaces.

---

## 4. Comparison: Random Networks vs. Correlated Networks

This is the central conceptual lesson of these two papers, directly relevant to your research on partially disordered networks:

```
Random network (RSM)          Correlated network (real LBL)
━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Poissonian disorder         • Non-random, correlated disorder
• Isotropic                   • Anisotropic (spray direction bias)
• No excluded volume          • Excluded volume effects
• Overestimates ACC           • Correct ACC
• Overestimates BC            • Correct BC distribution
• Wrong structure → wrong Δσ  • Correct structure → correct Δσ
```

The mathematical distinction: RSM samples from the **Erdős–Rényi** ensemble $G(n, p)$ (or its geometric analog). Real fiber networks belong to a **spatially correlated random graph** ensemble, closer to **random geometric graphs** with anisotropy and soft-core repulsion.

**Graph-theoretic parameters are sensitive to this distinction.** Average clustering coefficient and betweenness centrality differ by factors of 2–3 between the two ensembles at matched density.

---

## 5. Code Example: Computing GT Parameters with StructuralGT

```python
# Install: pip install StructuralGT
# or: conda install conda-forge::structuralgt

from StructuralGT.electronic import Electronic
from StructuralGT.networks import Network

# Load SEM image of AgNW network
# Options control image preprocessing
agnwn_options = {
    "Thresh_method": 0,    # 0=global, 1=Otsu, 2=adaptive
    "gamma": 1.001,        # gamma correction
    "md_filter": 0,        # median filter kernel size (0=off)
    "g_blur": 0,           # Gaussian blur (0=off)
    "autolvl": 0,          # auto contrast (0=off)
    "fg_color": 0,         # 0=dark background, 1=light background
    "thresh": 128.0,       # binarization threshold
    "asize": 3,            # morphological kernel size
    "bsize": 1,
    "wsize": 1,
}

# Initialize network from image directory
AgNWN = Network('path/to/AgNW_image_directory')
AgNWN.binarize(options=agnwn_options)   # image → binary
AgNWN.img_to_skel()                      # binary → skeleton
AgNWN.set_graph(weight_type=['FixedWidthConductance'])  # skeleton → graph

# Compute electrical properties
# (define source/drain terminal regions)
width = AgNWN.image.shape[0]
elec = Electronic()
elec.compute(AgNWN, 0, 0, [[0, 50], [width-50, width]])

# Access NetworkX graph object for custom analysis
G = AgNWN.Gr  # NetworkX graph
import networkx as nx

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Global efficiency: {nx.global_efficiency(G):.4f}")
print(f"Average clustering: {nx.average_clustering(G):.4f}")
print(f"Avg nodal connectivity: {nx.average_node_connectivity(G):.4f}")
```

---

## 6. Implications for Partially Disordered Network Research

For research on **partially disordered networks (PDNs)** — systems between perfect order and full randomness — these results suggest:

1. **GT parameters are structural fingerprints**: ACC, diameter, global efficiency, and betweenness centrality distribution together uniquely characterize a network's organization, independent of local symmetry.

2. **RSM is a null model, not a ground truth**: When experimental GT parameters deviate from RSM predictions, that deviation *is* the interesting physics — it encodes the non-random correlations in the material.

3. **Anisotropy is detectable by GT without orientation-resolved microscopy**: Graph Laplacian eigenvalue analysis captures directional conductance asymmetry that is invisible to isotropic structural metrics.

4. **Percolation threshold correlates with graph density and clustering**: Near $\phi_c$, GT parameters (especially global efficiency) show sharp transitions — graph theory provides a natural language for percolation.

---

## 7. Summary

| Paper | Key contribution |
|:---|:---|
| Vecchio et al., ACS Nano 2021 | StructuralGT: image → graph → 13 GT parameters for PNNs |
| Wu et al., Matter 2024 | GT predicts electrical/optical/mechanical properties; RSM fails for correlated disorder |

**StructuralGT** is now a standard tool in the Kotov group at UMich for nanomaterial structural characterization. It works on SEM, TEM, AFM, confocal, and electron tomography images, in 2D and 3D.

**Next in the series:** Computing treewidth of experimental AgNW graphs, and what low treewidth tells us about the mechanical fragility of near-threshold percolating networks.

---

## References

1. Vecchio, D. A., Mahler, S. H., Hammig, M. D., & Kotov, N. A. (2021). Structural Analysis of Nanoscale Network Materials Using Graph Theory. *ACS Nano*, **15**(8), 12847–12859. [https://doi.org/10.1021/acsnano.1c04711](https://doi.org/10.1021/acsnano.1c04711) ✅

2. Wu, W., Kadar, A., Lee, S. H., Glotzer, S. C., Goss, V., Kotov, N. A., et al. (2024). Layer-by-layer assembled nanowire networks enable graph-theoretical design of multifunctional coatings. *Matter*, **7**(10). [https://doi.org/10.1016/j.matt.2024.09.014](https://doi.org/10.1016/j.matt.2024.09.014) ✅

3. StructuralGT documentation: [https://structuralgt.readthedocs.io](https://structuralgt.readthedocs.io) ✅

4. StructuralGT GitHub: [https://github.com/compass-stc/StructuralGT](https://github.com/compass-stc/StructuralGT) ✅

5. COMPASS Lab (UMich): [https://compass.engin.umich.edu/structuralgt-software/](https://compass.engin.umich.edu/structuralgt-software/) ✅

---

*Tags: #structural-graph-theory #StructuralGT #nanowire #AgNW #percolation #graph-theory #materials-science #UMich*
