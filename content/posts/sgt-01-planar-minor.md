---
title: "Structural Graph Theory I: Planar Graphs, Minors, and Treewidth"
date: 2026-04-05
tags: ["graph-theory", "structural-graph-theory", "planar-graph", "graph-minor", "treewidth", "materials-science"]
draft: false
description: "An introduction to structural graph theory: planar graphs and Kuratowski's theorem, graph minors and the Robertson–Seymour theorem, treewidth and its role in algorithm design — with connections to disordered networks and materials science."
math: true
---

> *This is the first post in the **Structural Graph Theory** series, connecting foundational graph theory to materials science and disordered network research.*

---

## 1. What Is Structural Graph Theory?

Graph theory divides roughly into two broad programs. **Combinatorial graph theory** counts objects — how many graphs have a given property, how many colorings exist, what is the chromatic polynomial? **Structural graph theory**, by contrast, asks *what does a graph look like* — what is its internal geometric and topological organization, how can it be decomposed, and what forbidden substructures characterize a given family?

The key insight of structural graph theory is that **many important graph properties can be characterized by what a graph does *not* contain**. This is the spirit of forbidden subgraph and forbidden minor theorems, the deepest of which — the Robertson–Seymour theorem — is one of the great achievements of 20th-century mathematics.

In this post we introduce three foundational concepts:
1. **Planar graphs** — graphs that can be drawn without edge crossings
2. **Graph minors** — a fundamental containment relation between graphs
3. **Treewidth** — a measure of how "tree-like" a graph is

All three are directly relevant to the analysis of disordered materials networks (fiber networks, nanowire networks, amorphous solids).

---

## 2. Planar Graphs

### 2.1 Definition

A graph $G$ is **planar** if it can be **embedded** in the plane $\mathbb{R}^2$ — that is, drawn such that edges intersect only at their shared endpoints (no crossings). Such a drawing is called a **plane graph** or **planar embedding**.

$$G \text{ is planar} \iff \exists \text{ embedding } \phi: G \hookrightarrow \mathbb{R}^2 \text{ with no edge crossings}$$

The plane can be replaced by the sphere $S^2$ without loss of generality (via stereographic projection), so planarity is a topological rather than geometric property.

### 2.2 Euler's Formula

For any connected plane graph with $V$ vertices, $E$ edges, and $F$ faces (including the unbounded outer face):

$$V - E + F = 2$$

This is **Euler's formula for planar graphs**, a special case of the Euler characteristic $\chi = V - E + F$ for surfaces of genus 0.

**Consequences:**
- For simple planar graphs with $V \geq 3$: $E \leq 3V - 6$
- For triangle-free simple planar graphs: $E \leq 2V - 4$
- Average degree of a planar graph: $\langle k \rangle = 2E/V \leq 6 - 12/V < 6$

The last inequality means **every planar graph has a vertex of degree at most 5** — a fact used in the proof of the four-color theorem.

### 2.3 Kuratowski's Theorem (1930)

The fundamental characterization of planar graphs is by **forbidden subgraphs**:

> **Kuratowski's Theorem:** A finite graph is planar if and only if it contains no subdivision of $K_5$ (the complete graph on 5 vertices) or $K_{3,3}$ (the complete bipartite graph on $3+3$ vertices) as a subgraph.

A **subdivision** of $H$ is obtained by replacing each edge of $H$ with a path of one or more edges (inserting degree-2 vertices along edges).

**Wagner's Theorem (1937)** gives an equivalent characterization in terms of minors (see §3):

> A finite graph is planar if and only if it has neither $K_5$ nor $K_{3,3}$ as a **minor**.

```
K₅ (non-planar)          K₃,₃ (non-planar)
                          
    1                     a — d
   /|\                    a — e
  / | \                   a — f
 2--+--5                  b — d
  \ | /                   b — e
   \|/                    b — f
    3--4                  c — d
                          c — e
                          c — f
```

### 2.4 Planarity in Materials Networks

Why does planarity matter for materials science?

- **2D fiber networks** (paper, nonwovens, AgNW films) are inherently **planar graphs**: fibers lie in a plane, and fiber-fiber contacts define edges. For idealized zero-width fibers, the network is planar by construction.
- **Grain boundary networks** in 2D polycrystalline materials (thin films, 2D materials like graphene) are planar.
- **Planarity breaks down** in 3D fiber composites, porous media, and amorphous solids — which is precisely where more general structural tools (minors, treewidth) become essential.

The constraint $E \leq 3V - 6$ gives an upper bound on the contact density in planar fiber networks, with implications for mechanical rigidity and percolation.

---

## 3. Graph Minors

### 3.1 Definition

An **edge contraction** merges two adjacent vertices $u, v$ into a single vertex $w$, where $w$ inherits all neighbors of both $u$ and $v$ (excluding the edge $uv$ itself).

A graph $H$ is a **minor** of $G$ (written $H \preceq G$) if $H$ can be obtained from $G$ by:
1. Deleting edges
2. Deleting vertices (with their incident edges)
3. Contracting edges

Equivalently, $H \preceq G$ if and only if there exists a collection of disjoint connected subgraphs $\{B_v\}_{v \in V(H)}$ of $G$ (called **branch sets**) such that for every edge $uv \in E(H)$, there is an edge in $G$ between $B_u$ and $B_v$.

$$H \preceq G \iff \exists \text{ branch sets } \{B_v\} \text{ in } G \text{ s.t. edges of } H \text{ correspond to inter-branch edges of } G$$

### 3.2 The Robertson–Seymour Theorem

The most profound result in structural graph theory is:

> **Robertson–Seymour Theorem** (proved in a series of 23 papers, 1983–2004): The set of all finite graphs is **well-quasi-ordered** under the minor relation. Equivalently, in any infinite sequence of graphs $G_1, G_2, G_3, \ldots$, there exist indices $i < j$ such that $G_i \preceq G_j$.

**Corollary (Graph Minor Theorem):** For every graph property $\mathcal{P}$ that is **closed under taking minors** (i.e., if $G \in \mathcal{P}$ and $H \preceq G$ then $H \in \mathcal{P}$), there exists a **finite** set of forbidden minors $\mathcal{F}$ such that:

$$G \in \mathcal{P} \iff \text{no graph in } \mathcal{F} \text{ is a minor of } G$$

This is an existence theorem — it does not tell us what $\mathcal{F}$ is, only that it is finite. Known forbidden minor sets include:

| Property | Forbidden minors |
|:---|:---|
| Planar | $K_5$, $K_{3,3}$ (Wagner's theorem) |
| Outerplanar | $K_4$, $K_{2,3}$ |
| Treewidth $\leq k$ | Finite set (unknown for $k \geq 4$) |
| Genus $\leq g$ | Finite set (unknown explicitly for $g \geq 1$) |

### 3.3 Minor-Closed Properties in Materials Networks

The minor relation is not just abstract mathematics — it has physical meaning in network science:

- **Edge deletion** = removing a bond/contact between two structural units
- **Edge contraction** = "coarse-graining" two strongly bonded units into a single effective unit

A property is minor-closed if it is preserved under both operations. Examples:

- **Planarity** is minor-closed (removing or contracting an edge in a planar graph preserves planarity)
- **Connectivity $\leq k$** is minor-closed
- **Treewidth $\leq k$** is minor-closed

**Application to AgNW networks:** When nanowires are removed from a network (e.g., by laser ablation for patterning), the resulting graph is a minor of the original. If the original network has a certain structural property (e.g., bounded treewidth, planarity), the patterned network inherits it.

---

## 4. Treewidth

### 4.1 Motivation

A **tree** is the simplest connected graph: no cycles, unique path between any two vertices. Most hard combinatorial problems (graph coloring, independent set, Hamiltonian path) are efficiently solvable on trees. **Treewidth** measures how close a graph is to being a tree.

Formally, the treewidth of a graph $G$ is the minimum, over all **tree decompositions** of $G$, of one less than the maximum bag size.

### 4.2 Tree Decomposition

A **tree decomposition** of $G$ is a pair $(T, \{X_t\}_{t \in V(T)})$ where:
- $T$ is a tree
- Each $X_t \subseteq V(G)$ is a **bag** (subset of vertices of $G$)
- **Coverage:** every $v \in V(G)$ appears in at least one bag
- **Coherence:** for every edge $uv \in E(G)$, there exists a bag containing both $u$ and $v$
- **Connectivity:** for every $v \in V(G)$, the set of bags containing $v$ forms a connected subtree of $T$

The **width** of a tree decomposition is $\max_t |X_t| - 1$. The **treewidth** $\text{tw}(G)$ is the minimum width over all tree decompositions.

$$\text{tw}(G) = \min_{\text{tree decompositions}} \left(\max_t |X_t| - 1\right)$$

**Key values:**
- $\text{tw}(G) = 1 \iff G$ is a forest (tree or union of trees)
- $\text{tw}(K_n) = n-1$ (complete graph has maximum treewidth)
- $\text{tw}(G) \leq 2$ for series-parallel graphs
- Planar graphs have treewidth $O(\sqrt{n})$ [Alon, Seymour, Thomas, 1990] [needs verification]

### 4.3 Treewidth and Algorithmic Complexity

The fundamental theorem of parameterized complexity for treewidth:

> **Courcelle's Theorem** (1990): Every graph property expressible in **monadic second-order logic** (MSO₂) can be decided in **linear time** on graphs of bounded treewidth.

This means: if $\text{tw}(G) \leq k$ for fixed $k$, then problems like graph coloring, Hamiltonian cycle, and independent set — all NP-hard in general — become solvable in $O(f(k) \cdot n)$ time.

**Reference:** Courcelle, B. (1990). The monadic second-order logic of graphs I. *Information and Computation*, **85**(1), 12–75. [https://doi.org/10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401(90)90043-H) [needs verification]

### 4.4 Treewidth in Disordered Networks

For random and disordered networks relevant to materials science:

| Network type | Typical treewidth | Interpretation |
|:---|:---|:---|
| Tree (perfect) | 1 | No cycles, brittle, low redundancy |
| 2D planar lattice ($n \times n$) | $\Theta(n)$ | High treewidth, many independent paths |
| Random Erdős–Rényi $G(n, c/n)$, $c < 1$ | $O(\log n)$ | Sparse, tree-like components |
| Random $G(n, c/n)$, $c > 1$ | $\Omega(n)$ [needs verification] | Giant component, large cycles |
| 2D fiber network near percolation threshold | Low–intermediate | Near-critical, sparse spanning structure |

**Physical significance:** Low treewidth = the network can be "unfolded" into a nearly-tree structure, meaning:
- Fewer redundant paths (lower mechanical resilience)
- Easier to compute transport properties analytically
- More susceptible to targeted removal of vertices (percolation)

Near the **percolation threshold** $\phi_c$, the spanning cluster is known to be statistically self-similar and fractal — its treewidth grows sub-linearly, reflecting the sparse, tree-like structure of the incipient infinite cluster.

---

## 5. Connections: Planar Graphs, Minors, Treewidth

These three concepts are deeply interrelated:

```
Treewidth tw(G) = 1
        ↕ (iff)
    G is a forest
        ↓ (minor-closed, forbidden minors: {K₃})

Treewidth tw(G) ≤ 2
        ↕ (iff)
  G is series-parallel
        ↓ (forbidden minor: {K₄})

Treewidth tw(G) ≤ k
        ↕ (iff)
  No (k+2)-clique minor K_{k+2} ⪯ G  [Necessary; sufficient only for k ≤ 2]
        ↓

Planar graphs: tw(G) = O(√n), forbidden minors: {K₅, K₃,₃}
        ↓
General graphs: tw(G) up to n-1
```

**Hadwiger's Conjecture** (one of the most famous open problems in graph theory) relates treewidth-adjacent concepts to coloring:

> If $\chi(G) \geq k$, then $K_k \preceq G$.

Verified for $k \leq 6$ (Robertson, Seymour, Thomas 1993 for $k=6$). Open for $k \geq 7$.

---

## 6. Summary and Outlook

| Concept | Definition | Materials relevance |
|:---|:---|:---|
| Planar graph | Embeddable in $\mathbb{R}^2$ without crossings | 2D fiber/AgNW networks |
| Kuratowski/Wagner | No $K_5$, $K_{3,3}$ minor/subdivision | Test planarity of network |
| Graph minor | Obtained by deletion + contraction | Coarse-graining, network patterning |
| Robertson–Seymour | Every minor-closed property has finite forbidden set | Existence of structural characterization |
| Treewidth | Distance from being a tree | Mechanical redundancy, transport |
| Courcelle's Theorem | MSO₂ problems linear-time on bounded-tw graphs | Efficient analysis of near-tree networks |

**Next in the series:** Treewidth algorithms, tree decomposition in practice, and computing structural properties of AgNW networks with NetworkX.

---

## References

> DOIs verified where possible via browser access (April 2026). [needs verification] marks entries not directly confirmed.

1. Kuratowski, K. (1930). Sur le problème des courbes gauches en topologie. *Fundamenta Mathematicae*, **15**, 271–283. [https://doi.org/10.4064/fm-15-1-271-283](https://doi.org/10.4064/fm-15-1-271-283) [needs verification]

2. Wagner, K. (1937). Über eine Eigenschaft der ebenen Komplexe. *Mathematische Annalen*, **114**, 570–590. [https://doi.org/10.1007/BF01594196](https://doi.org/10.1007/BF01594196) [needs verification]

3. Robertson, N., & Seymour, P. D. (2004). Graph minors. XX. Wagner's conjecture. *Journal of Combinatorial Theory, Series B*, **92**(2), 325–357. [https://doi.org/10.1016/j.jctb.2004.08.001](https://doi.org/10.1016/j.jctb.2004.08.001) [needs verification]

4. Courcelle, B. (1990). The monadic second-order logic of graphs I. *Information and Computation*, **85**(1), 12–75. [https://doi.org/10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401(90)90043-H) [needs verification]

5. Diestel, R. (2017). *Graph Theory* (5th ed.). Springer. [https://diestel-graph-theory.com](https://diestel-graph-theory.com) ✅ (freely available online)

6. Robertson, N., & Seymour, P. D. (1986). Graph minors. II. Algorithmic aspects of tree-width. *Journal of Algorithms*, **7**(3), 309–322. [https://doi.org/10.1016/0196-6780(86)90023-4](https://doi.org/10.1016/0196-6780(86)90023-4) [needs verification]

---

*Tags: #structural-graph-theory #planar-graph #graph-minor #treewidth #materials-science #graph-theory*
