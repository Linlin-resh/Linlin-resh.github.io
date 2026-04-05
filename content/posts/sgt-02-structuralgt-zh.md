---
title: "结构图论 II：StructuralGT——把纳米材料网络读成图"
date: 2026-04-05
tags: ["graph-theory", "structural-graph-theory", "StructuralGT", "nanowire", "AgNW", "materials-science", "percolation", "python"]
draft: false
description: "如何定量描述无序纳米材料的结构？StructuralGT 把显微镜图像中的纤维网络自动转化为图，计算13个图论参数，将网络拓扑与电学、光学、力学性质直接关联——来自 UMich Kotov 课题组的两篇重要工作。"
math: true
---

> *本文是**结构图论**系列第二篇。[第一篇](../sgt-01-planar-minor-zh/)介绍了平面图、图子式（graph minor）和树宽（treewidth）。这一篇我们从理论走向实验：如何真正测量一个真实材料的图结构？*

---

## 1. 问题：没有周期性，如何描述结构？

晶体结构有晶胞（unit cell）。非晶和无序材料没有。那么，当我们面对一张银纳米线（AgNW）薄膜、芳纶纳米纤维（aramid nanofiber, ANF）、碳纳米管（carbon nanotube, CNT）薄膜的电镜图像时，如何定量描述其"结构"？

传统的结构描述子——晶格参数（lattice parameters）、配位壳（coordination shells）、对分布函数（pair distribution function, PDF）——都隐含地假设局部周期性（local periodicity）。它们对于**渗流纳米尺度网络（percolating nanoscale networks, PNNs）**失效：在 PNNs 中，决定宏观性质的是**扩展连通性（extended connectivity）**，而非局部有序性（local order）。

密歇根大学 Kotov 课题组的两篇论文给出了核心洞见：

> **把网络当成图（graph）。测量图。**

这个想法听起来简单，实现它——从一张无序纳米纤维网络的显微镜图像中自动提取图，再计算拓扑上有意义的参数——需要开发一个全新的开源工具：**StructuralGT**。

---

## 2. StructuralGT：从显微镜图像到图

**StructuralGT** 是由 Vecchio、Mahler、Hammig 和 Kotov 在 UMich COMPASS 实验室开发的 Python 软件包（同时提供 GUI 界面），首次发表于：

> Vecchio, D. A., Mahler, S. H., Hammig, M. D., & Kotov, N. A. (2021). Structural Analysis of Nanoscale Network Materials Using Graph Theory. *ACS Nano*, **15**(8), 12847–12859. [https://doi.org/10.1021/acsnano.1c04711](https://doi.org/10.1021/acsnano.1c04711) ✅

开源获取方式：
- PyPI: `pip install StructuralGT`
- GitHub: [compass-stc/StructuralGT](https://github.com/compass-stc/StructuralGT)
- 文档: [structuralgt.readthedocs.io](https://structuralgt.readthedocs.io)

### 2.1 图像到图的流程（Image-to-Graph Pipeline）

从显微镜图像到图模型的转换流程如下：

```
原始显微镜图像（SEM/TEM/AFM/共聚焦）
        ↓ 
   图像预处理（Preprocessing）
   裁剪、降噪（中值滤波 + 高斯模糊）
   对比度调整、阈值分割（全局/OTSU/自适应）
        ↓
   二值图像（Binary Image）
   纤维 = 白色，背景 = 黑色
        ↓
   骨架化（Skeletonization）
   纤维细化为1像素宽骨架
        ↓
   节点检测（Node Detection）
   分支点（branch points）= 节点（nodes）
   端点（endpoints）= 叶节点（leaf nodes）
        ↓
   边追踪（Edge Tracing）
   节点间的纤维段 = 边（edges）
        ↓
   后处理（Post-processing）
   去除不连通片段、修剪锯齿边（jagged edges）
   合并邻近节点、去除悬挂边（dangling edges）
        ↓
   图 G = (V, E)
   （含空间坐标 + 边权重）
```

几何解释有明确的物理含义：
- **节点（Nodes）** = 纤维分支点（junction）或纤维端点
- **边（Edges）** = 节点之间的纤维段
- **边权重（Edge weight）** = 纤维段的物理长度，或计算得到的电导值（conductance）

### 2.2 13个图论参数（Graph-Theoretic Parameters）

从图 $G = (V, E)$ 提取后，StructuralGT 计算以下13个结构参数：

| 参数 | 符号 | 物理含义 |
|:---|:---|:---|
| 度分布（Degree distribution） | $\{k_v\}$ | 每个节点有多少条纤维相交 |
| 图密度（Graph density） | $\rho = 2\|E\|/(\|V\|(\|V\|-1))$ | 实际连接占最大可能连接的比例 |
| 网络直径（Network diameter） | $d = \max_{u,v} d(u,v)$ | 最远两点间的最短路径长度 |
| 全局效率（Global efficiency） | $E_{glob} = \frac{1}{\|V\|(\|V\|-1)}\sum_{u \neq v} \frac{1}{d(u,v)}$ | 网络整体输运效率 |
| Wiener 指数 | $W = \sum_{u < v} d(u,v)$ | 所有节点对距离之和（与扩散相关） |
| 聚类系数（Clustering coefficient） | $C$ | 局部三角形密度（环结构丰富程度） |
| 平均节点连通性（Avg. nodal connectivity） | $\kappa$ | 节点对之间的平均最小顶点割 |
| 同配性系数（Assortativity coefficient） | $r$ | 度-度相关性 |
| 介数中心性（Betweenness centrality） | $BC_v$ | 节点作为"瓶颈"的程度 |
| 接近中心性（Closeness centrality） | $CC_v$ | 节点到所有其他节点的平均距离之倒数 |
| 特征向量中心性（Eigenvector centrality） | $EC_v$ | 邻居影响力加权的节点影响力 |
| *（+ 2个几何/空间参数）* | | |

**关键结构-性质关联：**

- **平均节点连通性** $\kappa$ ↔ 力学韧性（toughness）和抗剪切能力（shear resistance）
- **聚类系数** $C$ ↔ 压缩/拉伸强度（局部载荷重分配能力）
- **全局效率** $E_{glob}$ ↔ 电导率和离子输运
- **介数中心性分布** ↔ 电流集中点和失效模式（failure mode）
- **同配性** $r$ ↔ 网络韧性（正 $r$：高度节点连接高度节点；负 $r$：轮毂-辐条型拓扑）

---

## 3. Matter 2024：图论预测显微镜看不见的东西

这个方法的真正威力体现在2024年发表在 *Matter* 上的工作：

> Wu, W., Kadar, A., Lee, S. H., Glotzer, S. C., Goss, V., Kotov, N. A., et al. (2024). Layer-by-layer assembled nanowire networks enable graph-theoretical design of multifunctional coatings. *Matter*, **7**(10). [https://doi.org/10.1016/j.matt.2024.09.014](https://doi.org/10.1016/j.matt.2024.09.014) ✅

### 3.1 材料：逐层自组装（LBL）银/金纳米线薄膜

**逐层自组装（layer-by-layer assembly, LBL）**是一种薄膜制备技术，通过带相反电荷的材料交替沉积构建多层结构：

1. 喷涂聚乙烯亚胺（PEI，正电荷）到清洁基底
2. 冲洗干燥
3. 喷涂 AgNW 或 AuNW 悬浮液（负电荷）
4. 冲洗：去除松散附着的纳米线，保留导电结（conducting junction）
5. 重复：每次 LBL 循环（cycle）增加一层纳米线

关键结构特征：每层纳米线被限制在 $x$–$y$ 平面内，但层与层在 $z$ 方向堆叠。层间 NW–NW 接触点同样导电。最终形成一个**准二维多层纳米线网络（quasi-2D multilayer NW network）**，同时具有面内和面外导电路径。

### 3.2 核心问题：为什么随机棒模型（Random Stick Model）失败了？

纳米线网络的标准计算模型是**随机棒模型（random stick model, RSM）**：在盒子中随机放置固定长度的细棒（stick），为每个 stick–stick 接触定义电导，求解基尔霍夫方程（Kirchhoff's equations）。

RSM 易于处理，且与渗流理论有解析联系。但 Wu et al. 证明，RSM 对 LBL 薄膜**结构描述是错的**：

| 参数 | RSM 预测 | 真实 AgNW LBL | 差异 |
|:---|:---|:---|:---|
| 平均聚类系数（ACC） | 约2倍偏高 | 实验值 | RSM 高估 |
| 平均介数中心性（ABC） | 约3倍偏高（AuNW） | 实验值 | RSM 高估 |

原因：RSM 假设**随机（泊松式）无序（Poissonian disorder）**。而 LBL 纳米线具有**关联的、非随机无序（correlated, non-random disorder）**——喷涂方向导致纳米线轻微排列，产生各向异性（anisotropy）和体积排除效应（excluded-volume effects），RSM 无法捕获。

**结果：** 通过调整 RSM 参数使片电阻（sheet resistance）匹配实验，会给出错误的结构描述——并对所有其他性质给出错误预测。

**根本矛盾：** RSM 属于 Erdős–Rényi 随机图系（或其几何版本）的采样。真实 LBL 纳米线网络属于**空间关联随机图（spatially correlated random graph）**系综，更接近带各向异性和软核排斥的**随机几何图（random geometric graph）**。

### 3.3 基于图像的 GT 模型：预测了什么？

将真实 LBL 薄膜的 SEM/TEM/AFM 图像输入 StructuralGT，把多层薄膜建模为**垂直堆叠的单层图（vertically stacked single-layer graphs）**（以层间连接比例作为唯一拟合参数），GT 模型正确预测了：

**电学性质（Electrical properties）：**
- 片电阻 vs. LBL 循环数 $N$
- 电荷输运 vs. $N$ 的**非线性行为**——RSM 完全漏掉的意外发现
- 导电**各向异性**（$\sigma_x \neq \sigma_y$）——单靠显微镜无法检测，但图拉普拉斯（graph Laplacian）分析正确预测
- 承载电流（failure current）随 $N$ 的变化

**光学性质（Optical properties）：**
- 光学各向异性（偏振相关吸收）
- THz 吸收
- 光学旋转（optical rotation）

**力学性质（Mechanical properties）：**
- 弹性模量（elastic modulus）估算
- 表面粗糙度（surface roughness）

计算电学性质的核心工具是**图拉普拉斯伪逆（graph Laplacian pseudoinverse）** $L^+$：

$$\sigma \propto (L^+)_{ij}$$

其中 $L = D - A$（$D$ 为度矩阵，$A$ 为加权邻接矩阵），端点 $i$, $j$ 之间的电导通过 $L$ 的 Moore–Penrose 伪逆计算。

### 3.4 从实验室到无人机机翼：GT 指导的规模化涂层

基于 GT 优化的 LBL 涂层通过喷雾沉积在**大型曲面无人机机翼**（米量级）上，实现了：
- **雷击防护**（电导率）
- **除冰**（焦耳加热 / Joule heating）
- **隐身**（THz 吸收）
- **传感**（光学各向异性）

GT 设计框架——从图像提取图、计算参数、预测性质、优化配方——从毫米级实验室样品直接推广到米级航空表面。

---

## 4. 随机网络 vs. 关联网络：核心概念对比

这两篇论文最重要的概念教训，与无序网络（partially disordered network）研究直接相关：

```
随机网络（RSM）                    关联网络（真实 LBL）
━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━━━
• 泊松无序（Poissonian disorder）  • 非随机关联无序
• 各向同性（isotropic）            • 各向异性（spray bias）
• 无体积排除效应                   • 体积排除效应显著
• 高估 ACC（约2倍）                • 正确 ACC
• 高估 BC（约3倍）                 • 正确 BC 分布
• 结构错误 → 性质预测全错          • 结构正确 → 性质预测正确
```

**图论参数对这种差异极其敏感。** 在相同密度下，两种系综的平均聚类系数和介数中心性相差2–3倍。

---

## 5. 代码示例：用 StructuralGT 计算图论参数

```python
# 安装：pip install StructuralGT
# 或：conda install conda-forge::structuralgt

from StructuralGT.electronic import Electronic
from StructuralGT.networks import Network

# 加载 AgNW 网络的 SEM 图像
# options 控制图像预处理参数
agnwn_options = {
    "Thresh_method": 0,    # 0=全局, 1=Otsu, 2=自适应
    "gamma": 1.001,        # gamma 校正
    "md_filter": 0,        # 中值滤波核尺寸（0=关闭）
    "g_blur": 0,           # 高斯模糊（0=关闭）
    "autolvl": 0,          # 自动对比度（0=关闭）
    "fg_color": 0,         # 0=暗背景, 1=亮背景
    "thresh": 128.0,       # 二值化阈值
    "asize": 3,            # 形态学核尺寸
    "bsize": 1,
    "wsize": 1,
}

# 从图像目录初始化网络
AgNWN = Network('path/to/AgNW_image_directory')
AgNWN.binarize(options=agnwn_options)   # 图像 → 二值图
AgNWN.img_to_skel()                      # 二值图 → 骨架
AgNWN.set_graph(weight_type=['FixedWidthConductance'])  # 骨架 → 图

# 计算电学性质
# 定义源/漏端子区域
width = AgNWN.image.shape[0]
elec = Electronic()
elec.compute(AgNWN, 0, 0, [[0, 50], [width-50, width]])

# 访问 NetworkX 图对象进行自定义分析
G = AgNWN.Gr  # NetworkX graph
import networkx as nx

print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
print(f"全局效率: {nx.global_efficiency(G):.4f}")
print(f"平均聚类系数: {nx.average_clustering(G):.4f}")
print(f"平均节点连通性: {nx.average_node_connectivity(G):.4f}")
```

---

## 6. 对"部分无序网络"研究的启示

对于**部分无序网络（partially disordered networks, PDNs）**研究——介于完美有序和完全随机之间的体系——这两篇论文给出了几条关键启示：

1. **图论参数是结构指纹（structural fingerprints）：** ACC、直径、全局效率和介数中心性分布共同唯一地刻画网络组织，独立于局部对称性。

2. **RSM 是零假设（null model），不是基准真理：** 当实验图论参数与 RSM 预测偏离时，这种偏离*本身*就是有趣的物理——它编码了材料中的非随机关联。

3. **各向异性可以被图论检测，而无需方向分辨显微镜：** 图拉普拉斯特征值分析可以捕获各向异性电导，这对各向同性结构指标完全不可见。

4. **渗流阈值与图密度和聚类系数相关：** 在 $\phi_c$ 附近，图论参数（尤其是全局效率）出现急剧转变——图论为渗流提供了自然的数学语言。

---

## 7. 总结

| 论文 | 关键贡献 |
|:---|:---|
| Vecchio et al., ACS Nano 2021 | StructuralGT：图像→图→13个图论参数，适用于各类 PNNs |
| Wu et al., Matter 2024 | GT 预测电/光/力学性质；随机棒模型对关联无序失效 |

**StructuralGT** 现已成为 UMich Kotov 课题组纳米材料结构表征的标准工具，支持 SEM、TEM、AFM、共聚焦和电子断层摄影图像，覆盖二维和三维场景。

**系列下一篇：** 计算实验 AgNW 图的树宽，以及低树宽对接近渗流阈值的网络力学脆性意味着什么。

---

## 参考文献

1. Vecchio, D. A., Mahler, S. H., Hammig, M. D., & Kotov, N. A. (2021). Structural Analysis of Nanoscale Network Materials Using Graph Theory. *ACS Nano*, **15**(8), 12847–12859. [https://doi.org/10.1021/acsnano.1c04711](https://doi.org/10.1021/acsnano.1c04711) ✅

2. Wu, W., Kadar, A., Lee, S. H., Glotzer, S. C., Goss, V., Kotov, N. A., et al. (2024). Layer-by-layer assembled nanowire networks enable graph-theoretical design of multifunctional coatings. *Matter*, **7**(10). [https://doi.org/10.1016/j.matt.2024.09.014](https://doi.org/10.1016/j.matt.2024.09.014) ✅

3. StructuralGT 文档: [https://structuralgt.readthedocs.io](https://structuralgt.readthedocs.io) ✅

4. StructuralGT GitHub: [https://github.com/compass-stc/StructuralGT](https://github.com/compass-stc/StructuralGT) ✅

5. COMPASS 实验室（UMich）: [https://compass.engin.umich.edu/structuralgt-software/](https://compass.engin.umich.edu/structuralgt-software/) ✅

---

*标签：#structural-graph-theory #StructuralGT #nanowire #AgNW #percolation #graph-theory #materials-science #UMich*
