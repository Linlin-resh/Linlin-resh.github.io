---
title: "结构图论 I：平面图、图子式与树宽"
date: 2026-04-05
tags: ["graph-theory", "structural-graph-theory", "planar-graph", "graph-minor", "treewidth", "materials-science"]
draft: false
description: "结构图论入门：平面图（planar graph）与 Kuratowski 定理、图子式（graph minor）与 Robertson–Seymour 定理、树宽（treewidth）及其在材料科学无序网络分析中的应用。"
math: true
---

> *本文是**结构图论**系列的第一篇，将图论基础与材料科学、无序网络研究贯通。*

---

## 1. 什么是结构图论（Structural Graph Theory）？

图论大致可以分为两个研究纲领。**组合图论（combinatorial graph theory）**以计数为核心——有多少种图满足某个性质？着色多项式（chromatic polynomial）是什么？而**结构图论（structural graph theory）**则追问：*一个图长什么样？* 它的内部几何（geometry）和拓扑（topology）结构如何？能否分解？哪些"禁止子结构（forbidden substructure）"刻画了某一族图？

结构图论的核心洞见是：**许多重要的图性质可以用图"不包含什么"来刻画**。这是禁止子图（forbidden subgraph）和禁止子式（forbidden minor）定理的精髓所在。其中最深刻的——Robertson–Seymour 定理——是20世纪数学最伟大的成就之一。

本文介绍三个基础概念：
1. **平面图（planar graph）**——可以无交叉地画在平面上的图
2. **图子式（graph minor）**——图之间的一种基本包含关系
3. **树宽（treewidth）**——衡量图"像树程度"的参数

这三个概念对分析无序材料网络（fiber networks、AgNW networks、非晶固体）都直接相关。

---

## 2. 平面图（Planar Graph）

### 2.1 定义

图 $G$ 是**平面图（planar graph）**，当且仅当它可以**嵌入（embed）**到平面 $\mathbb{R}^2$ 中——即可以画在纸面上，使得边仅在顶点处相交（无交叉）。这样的画法叫做**平面嵌入（planar embedding）**或**平面图（plane graph）**。

$$G \text{ 是平面图} \iff \exists \text{ 嵌入 } \phi: G \hookrightarrow \mathbb{R}^2 \text{ 使得边无交叉}$$

平面可以用球面 $S^2$ 代替（通过球极投影 / stereographic projection），因此平面性（planarity）是拓扑性质，而非几何性质。

### 2.2 欧拉公式（Euler's Formula）

对于任意连通平面图，设顶点数（vertices）为 $V$，边数（edges）为 $E$，面数（faces，包含无界外面）为 $F$，则：

$$V - E + F = 2$$

这是**平面图的欧拉公式**，是亏格（genus）为0的曲面的欧拉示性数（Euler characteristic）$\chi = 2$ 的特例。

**推论（corollaries）：**
- 简单平面图（$V \geq 3$）：$E \leq 3V - 6$
- 无三角形（triangle-free）简单平面图：$E \leq 2V - 4$
- 平面图的平均度（average degree）：$\langle k \rangle = 2E/V \leq 6 - 12/V < 6$

最后一个不等式说明：**每个平面图中必有度数不超过5的顶点**——这是四色定理（four-color theorem）证明的关键引理。

### 2.3 Kuratowski 定理（1930）

平面图的基本刻画依赖于**禁止子图（forbidden subgraph）**：

> **Kuratowski 定理：** 有限图是平面图，当且仅当它不包含 $K_5$（5个顶点的完全图 / complete graph）或 $K_{3,3}$（完全二部图 / complete bipartite graph）的细分（subdivision）作为子图。

$H$ 的**细分（subdivision）**是将 $H$ 的每条边用一条路（插入若干度为2的顶点）替换后得到的图。

**Wagner 定理（1937）**给出等价的子式刻画（见§3）：

> 有限图是平面图，当且仅当它没有 $K_5$ 或 $K_{3,3}$ 作为**图子式（minor）**。

```
K₅（非平面）             K₃,₃（非平面）
    1                    a — d
   /|\                   a — e  
  / | \                  a — f
 2--+--5                 b — d
  \ | /                  b — e
   \|/                   b — f
    3--4                 c — d, c — e, c — f
```

> **直觉（Intuition）：** 为什么 $K_5$ 不能平面嵌入？5个点两两连接，每对点之间的路径太多，必然产生交叉。$K_{3,3}$ 则是著名的"三栋房子-三口井"问题的数学模型——三栋房子各自连接三口井，无论怎么画都会有管道交叉。

### 2.4 平面性与材料网络

平面性为什么对材料科学重要？

- **二维纤维网络（2D fiber network）**（纸张、无纺布、AgNW 薄膜）天然是**平面图（planar graph）**：纤维躺在平面上，纤维-纤维接触点（fiber-fiber contacts）定义边。对于理想零宽度纤维，网络构型（network topology）由定义保证是平面的。
- **二维多晶材料（2D polycrystalline materials）**的晶界网络（grain boundary network）（如石墨烯薄膜）也是平面图。
- **三维纤维复合材料（3D fiber composites）**、多孔介质（porous media）、非晶固体（amorphous solids）中平面性不再成立——这正是需要引入更一般结构工具（子式、树宽）的原因。

约束 $E \leq 3V - 6$ 给出了平面纤维网络中接触点密度（contact density）的上界，对力学刚性（mechanical rigidity）和渗流行为（percolation behavior）有直接影响。

---

## 3. 图子式（Graph Minor）

### 3.1 定义

**边收缩（edge contraction）**将相邻两个顶点 $u, v$ 合并为单个顶点 $w$，$w$ 继承 $u$ 和 $v$ 的所有邻居（去除 $uv$ 边本身）。

图 $H$ 是图 $G$ 的**子式（minor）**（记作 $H \preceq G$），如果 $H$ 可以通过以下操作从 $G$ 得到：
1. 删除边（edge deletion）
2. 删除顶点（vertex deletion，连同关联边）
3. 边收缩（edge contraction）

等价地，$H \preceq G$ 当且仅当 $G$ 中存在一组不相交连通子图 $\{B_v\}_{v \in V(H)}$（称为**分支集 / branch sets**），使得 $H$ 中每条边 $uv$ 对应 $G$ 中 $B_u$ 与 $B_v$ 之间的至少一条边。

**粗粒化视角（coarse-graining perspective）：** 子式关系可理解为对网络进行粗粒化（coarse-graining）——将强连接的节点团簇（cluster）视为单个有效节点，忽略内部细节，保留宏观连接结构。

### 3.2 Robertson–Seymour 定理

结构图论中最深刻的结论是：

> **Robertson–Seymour 定理**（1983–2004年，共23篇论文完成证明）：所有有限图的集合在子式关系下是**拟序良序（well-quasi-ordered）**的。等价地，任意无穷图序列 $G_1, G_2, G_3, \ldots$ 中，必存在 $i < j$ 使得 $G_i \preceq G_j$。

**推论（图子式定理 / Graph Minor Theorem）：** 对于每个**在子式下封闭（minor-closed）**的图性质 $\mathcal{P}$（即若 $G \in \mathcal{P}$ 且 $H \preceq G$，则 $H \in \mathcal{P}$），存在一个**有限的**禁止子式集合（forbidden minor set）$\mathcal{F}$，使得：

$$G \in \mathcal{P} \iff \mathcal{F} \text{ 中没有图是 } G \text{ 的子式}$$

这是一个存在性定理（existence theorem）——它告诉我们 $\mathcal{F}$ 是有限的，但不告诉我们 $\mathcal{F}$ 具体是什么。已知的结果包括：

| 图性质 | 禁止子式集合 $\mathcal{F}$ |
|:---|:---|
| 平面图（planar） | $\{K_5,\ K_{3,3}\}$（Wagner 定理） |
| 外平面图（outerplanar） | $\{K_4,\ K_{2,3}\}$ |
| 树宽 $\leq k$（treewidth $\leq k$） | 有限集合（$k \geq 4$ 时未显式给出） |
| 亏格 $\leq g$（genus $\leq g$） | 有限集合（$g \geq 1$ 时未显式给出） |

### 3.3 子式关系在材料网络中的物理含义

- **删除边 = 断键**：去除两个结构单元之间的接触/键合（bond breakage）
- **边收缩 = 粗粒化**：将强键合的两个单元视为同一个有效单元

**在 AgNW 网络的激光图案化（laser patterning）中**：激光移除部分纳米线后，剩余网络是原网络的一个子图，也是其子式。若原网络具有某个子式封闭性质（如平面性、有界树宽），图案化后的网络自动继承该性质。

---

## 4. 树宽（Treewidth）

### 4.1 动机

**树（tree）**是最简单的连通图：无环，任意两点之间路径唯一。大多数困难的组合问题（图着色、独立集、哈密顿路径）在树上都有高效算法。**树宽（treewidth）**度量一个图距离树有多远。

### 4.2 树分解（Tree Decomposition）

图 $G$ 的**树分解（tree decomposition）**是一对 $(T, \{X_t\}_{t \in V(T)})$，其中：
- $T$ 是一棵树
- 每个 $X_t \subseteq V(G)$ 是一个**包（bag）**
- **覆盖性（coverage）：** 每个 $v \in V(G)$ 至少出现在一个包中
- **相干性（coherence）：** 对 $G$ 的每条边 $uv$，存在同时包含 $u$ 和 $v$ 的包
- **连通性（connectivity）：** 对每个 $v \in V(G)$，包含 $v$ 的所有包在 $T$ 中构成连通子树

树分解的**宽度（width）**为 $\max_t |X_t| - 1$。图 $G$ 的**树宽（treewidth）** $\text{tw}(G)$ 是所有树分解中最小的宽度：

$$\text{tw}(G) = \min_{\text{所有树分解}} \left(\max_t |X_t| - 1\right)$$

**关键数值（key values）：**
- $\text{tw}(G) = 1 \iff G$ 是森林（forest）
- $\text{tw}(K_n) = n-1$（完全图树宽最大）
- 串并联图（series-parallel graph）：$\text{tw} \leq 2$
- 平面图：$\text{tw}(G) = O(\sqrt{n})$ [需验证]

> **直觉（Intuition）：** 树宽小意味着图可以被"展开"成类树结构——存在一种层次性的分解，每层只需同时处理少数顶点。树宽大意味着图高度互连，无法简化为树状层级。

### 4.3 树宽与算法复杂性

**Courcelle 定理（1990）：** 每个可以用**一元二阶逻辑（monadic second-order logic, MSO₂）**表达的图性质，在有界树宽的图上都可以在**线性时间（linear time）**内判定。

这意味着：若 $\text{tw}(G) \leq k$（$k$ 固定），则图着色（graph coloring）、哈密顿圈（Hamiltonian cycle）、独立集（independent set）等 NP-困难问题都能在 $O(f(k) \cdot n)$ 时间内求解。

**参考文献：** Courcelle, B. (1990). The monadic second-order logic of graphs I. *Information and Computation*, **85**(1), 12–75. [https://doi.org/10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401(90)90043-H) [需验证]

### 4.4 树宽在无序网络中的意义

| 网络类型 | 典型树宽 | 物理含义 |
|:---|:---|:---|
| 树（完美树状） | 1 | 无冗余路径，脆性高 |
| 二维方格子（$n \times n$） | $\Theta(n)$ | 高树宽，路径冗余度高 |
| 随机 Erdős–Rényi $G(n,c/n)$，$c < 1$ | $O(\log n)$ | 稀疏，树状连通分量 |
| 随机 $G(n,c/n)$，$c > 1$ | $\Omega(n)$ [需验证] | 巨连通分量，大量环 |
| 渗流阈值附近的二维纤维网络 | 低–中等 | 近临界，生成树稀疏 |

**物理意义：** 低树宽 = 网络可被"展开"为近树结构，意味着：
- 冗余路径（redundant paths）少，力学韧性（mechanical resilience）低
- 输运性质（transport properties）可以更解析地计算
- 更容易被靶向顶点移除破坏（percolation resilience 差）

在**渗流阈值（percolation threshold）** $\phi_c$ 附近，生成团簇（spanning cluster）是统计自相似（statistically self-similar）的分形——其树宽亚线性增长，反映了临界点团簇的稀疏树状结构（sparse tree-like structure）。

---

## 5. 三者之间的关系

```
树宽 tw(G) = 1
      ↕（等价）
   G 是森林（forest）
      ↓（子式封闭，禁止子式：{K₃}）

树宽 tw(G) ≤ 2
      ↕（等价）
  G 是串并联图（series-parallel）
      ↓（禁止子式：{K₄}）

树宽 tw(G) ≤ k
      ↓（必要条件）
  无 (k+2)-团子式，即 K_{k+2} ⋢ G

平面图：tw(G) = O(√n)，禁止子式：{K₅, K₃,₃}
      ↓
一般图：tw(G) 最大可达 n-1
```

**Hadwiger 猜想（Hadwiger's conjecture）**——图论最著名的未解问题之一：

> 若 $\chi(G) \geq k$（$G$ 的色数 / chromatic number 至少为 $k$），则 $K_k \preceq G$。

已验证到 $k \leq 6$（Robertson, Seymour, Thomas 1993 证明了 $k=6$ 的情形）。$k \geq 7$ 至今未解。

---

## 6. 总结与展望

| 概念 | 定义 | 材料科学意义 |
|:---|:---|:---|
| 平面图（planar graph） | 可无交叉嵌入 $\mathbb{R}^2$ | 二维纤维/AgNW 网络 |
| Kuratowski/Wagner 定理 | 无 $K_5$, $K_{3,3}$ 子式/细分 | 检验网络是否平面 |
| 图子式（graph minor） | 删边+删点+收缩边得到 | 粗粒化、网络图案化 |
| Robertson–Seymour 定理 | 子式封闭性质必有有限禁止集 | 结构性质的存在性保证 |
| 树宽（treewidth） | 距离树的远近 | 力学冗余度、输运可计算性 |
| Courcelle 定理 | 有界树宽图上 MSO₂ 线性时间 | 近树网络的高效计算 |

**系列下一篇：** 树宽算法实现、实际树分解，以及用 NetworkX 计算 AgNW 网络结构属性的代码实战。

---

## 参考文献（References）

> 以下链接已尽量提供 DOI，标注 [需验证] 的条目因访问限制无法实时核实，但来源可信。

1. Kuratowski, K. (1930). Sur le problème des courbes gauches en topologie. *Fundamenta Mathematicae*, **15**, 271–283. [https://doi.org/10.4064/fm-15-1-271-283](https://doi.org/10.4064/fm-15-1-271-283) [需验证]

2. Wagner, K. (1937). Über eine Eigenschaft der ebenen Komplexe. *Mathematische Annalen*, **114**, 570–590. [https://doi.org/10.1007/BF01594196](https://doi.org/10.1007/BF01594196) [需验证]

3. Robertson, N., & Seymour, P. D. (2004). Graph minors. XX. Wagner's conjecture. *Journal of Combinatorial Theory, Series B*, **92**(2), 325–357. [https://doi.org/10.1016/j.jctb.2004.08.001](https://doi.org/10.1016/j.jctb.2004.08.001) [需验证]

4. Courcelle, B. (1990). The monadic second-order logic of graphs I. *Information and Computation*, **85**(1), 12–75. [https://doi.org/10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401(90)90043-H) [需验证]

5. Diestel, R. (2017). *Graph Theory* (5th ed.). Springer. 免费在线版：[https://diestel-graph-theory.com](https://diestel-graph-theory.com) ✅

6. Robertson, N., & Seymour, P. D. (1986). Graph minors. II. Algorithmic aspects of tree-width. *Journal of Algorithms*, **7**(3), 309–322. [https://doi.org/10.1016/0196-6780(86)90023-4](https://doi.org/10.1016/0196-6780(86)90023-4) [需验证]

---

*标签：#structural-graph-theory #planar-graph #graph-minor #treewidth #materials-science #graph-theory*
