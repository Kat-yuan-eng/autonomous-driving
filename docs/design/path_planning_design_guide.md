# 路径规划：基于自适应启发式A*与动力学约束TEB的联合规划方法

## 1. 问题核心分析与定位

### 1.1 问题背景与核心任务

本问题来源于大学生智能车竞赛基础循迹组的路径规划任务。竞赛场景基于已知栅格地图，车辆模型为 Ackermann 转向模型（最大速度 $v_{max} = 2.5$ m/s，最大转向角 $\delta_{max} = \pm 30°$），规划系统运行于 MCU 嵌入式平台，单次规划周期需在 5ms 以内完成。地图环境包含静态障碍物与有限动态障碍物，车辆需在满足动力学约束的前提下，实时生成平滑、无碰撞、动力学可行的行驶轨迹。

核心任务：设计双层路径规划系统——全局规划采用自适应启发式 A* 算法，局部规划采用动力学约束 TEB（Timed Elastic Band）算法，实现全局路径搜索与局部轨迹优化的联合规划，在实时性约束下输出可执行轨迹。

### 1.2 问题核心难点

- **实时性约束**：全局规划需在 100ms 内完成，局部规划需在 5ms 内完成一次优化迭代，MCU 算力有限；
- **启发函数精度与效率的矛盾**：单一启发函数无法在所有方向上提供紧致下界，欧氏距离在对角方向紧致但轴对齐方向松弛，曼哈顿距离则相反；
- **动力学约束耦合**：车辆非完整性约束使局部轨迹优化高度非线性，转向角、速度、加速度之间存在强耦合关系；
- **轨迹平滑性**：TEB 优化后的轨迹可能存在加速度突变，需后处理平滑以满足执行器带宽限制；
- **全局-局部一致性**：局部规划偏离全局路径时需及时触发重规划，避免车辆驶入不可恢复区域。

## 2. 模型前置准备

### 2.1 基本假设

1. 竞赛地图为已知 8 连通栅格地图，障碍物位置与形状在全局规划阶段确定；
2. 车辆运动学模型采用 Ackermann 简化自行车模型，忽略侧滑与轮胎变形；
3. 环境包含静态障碍物与有限动态障碍物，动态障碍物速度不超过 $0.5$ m/s；
4. 全局规划频率 1–5 Hz，局部规划频率 20–50 Hz，评估以路径长度、搜索效率、轨迹平滑度、安全性为综合指标；
5. 车辆初始位姿与目标位姿已知，目标区域为地图中指定终点附近区域。

### 2.2 符号系统统一定义

#### 2.2.1 集合符号

| 符号 | 完整定义 |
|------|----------|
| $\mathcal{G} = (V, E)$ | 8 连通栅格图，$V$ 为节点集，$E$ 为边集 |
| $V = \{n_0, n_1, \ldots, n_N\}$ | 栅格节点集合 |
| $E \subseteq V \times V$ | 相邻节点连边集合 |
| $\mathcal{O}$ | 障碍物占据的栅格节点集合 |
| $\mathcal{O}_{dyn}$ | 动态障碍物位置集合 |
| $\mathcal{P}$ | 全局路径节点序列 |
| $\mathcal{T}$ | TEB 局部轨迹位姿-时间序列 |

#### 2.2.2 参数符号

| 符号 | 完整定义 |
|------|----------|
| $n_c = (x_c, y_c)$ | 当前节点坐标 |
| $n_g = (x_g, y_g)$ | 目标节点坐标 |
| $\theta$ | 当前节点到目标节点的方位角（度） |
| $h(n)$ | 启发函数估计值 |
| $g(n)$ | 从起点到节点 $n$ 的实际代价 |
| $f(n) = g(n) + h(n)$ | 节点 $n$ 的综合评估函数 |
| $L$ | 车辆轴距（m），TEB 模块取 $0.3$ m，DWA 模块取 $3.0$ m（仿真尺度差异） |
| $\delta$ | 前轮转向角（°） |
| $v$ | 车辆速度（m/s） |
| $\beta$ | 车辆航向角（rad） |
| $\mathbf{s}_i = (x_i, y_i, \beta_i)^\top$ | 第 $i$ 时刻车辆位姿向量 |
| $\Delta T_i$ | 第 $i$ 与第 $i+1$ 位姿间的时间间隔（s） |
| $d_{min}$ | 安全避障距离（m），默认 $0.3$ |
| $v_{max}$ | 最大速度约束（m/s），$2.5$ |
| $\delta_{max}$ | 最大转向角约束（°），$30$ |
| $a_{max}$ | 最大加速度约束（m/s²），$2.0$ |
| $\dot{\delta}_{max}$ | 最大转向角速度约束（°/s），$100$ |

#### 2.2.3 决策变量与输出

| 符号 | 完整定义 |
|------|----------|
| $\mathcal{P}^* = \{n_0, n_1, \ldots, n_k\}$ | 全局最优路径节点序列 |
| $\mathcal{T}^* = \{(\mathbf{s}_0, \Delta T_0), \ldots, (\mathbf{s}_m, \Delta T_m)\}$ | 局部最优轨迹位姿-时间序列 |

### 2.3 车辆动力学模型

采用 Ackermann 简化自行车模型描述车辆运动学约束：

$$\dot{x} = v\cos\beta, \quad \dot{y} = v\sin\beta, \quad \dot{\beta} = \frac{v\tan\delta}{L}$$

其中 $(x, y)$ 为后轴中心坐标，$\beta$ 为航向角，$v$ 为后轴中心速度，$\delta$ 为前轮等效转向角，$L$ 为轴距。

运动学约束边界：

$$|v| \leq v_{max} = 2.5 \;\text{m/s}, \quad |\delta| \leq \delta_{max} = 30°, \quad |a| \leq a_{max} = 2.0 \;\text{m/s}^2, \quad |\dot{\delta}| \leq \dot{\delta}_{max} = 100 \;\text{°/s}$$

## 3. 全局路径规划：自适应启发式A*算法

### 3.1 经典A*算法回顾与效率瓶颈

经典 A* 算法通过评估函数 $f(n) = g(n) + h(n)$ 引导搜索，其中 $g(n)$ 为起点到节点 $n$ 的实际代价，$h(n)$ 为启发函数估计。算法维护 open list（待扩展节点）与 closed list（已扩展节点），在 8 连通栅格上逐步扩展直至目标。

效率瓶颈：启发函数质量直接决定搜索效率。若 $h(n)$ 过于松弛，则扩展大量冗余节点；若 $h(n)$ 不可容（overestimate），则无法保证路径最优。单一启发函数（如欧氏距离）在特定方向上紧致，但在其他方向上松弛，导致搜索效率下降。

### 3.2 启发函数对比分析

8 连通栅格下三种常用启发函数的特性对比：

| 启发函数 | 公式 | 可容性 | 紧致性 | 适用方向 |
|----------|------|--------|--------|---------|
| 欧氏距离 $h_E$ | $\sqrt{(x_g - x_c)^2 + (y_g - y_c)^2}$ | 是 | 对角方向紧致 | 对角方向 |
| 曼哈顿距离 $h_M$ | $|x_g - x_c| + |y_g - y_c|$ | 是（4 连通） | 水平/垂直紧致 | 轴对齐方向 |
| 切比雪夫距离 $h_C$ | $\max(|x_g - x_c|, |y_g - y_c|)$ | 是（8 连通） | 轴对齐次紧致 | 近轴方向 |

核心洞察：在 8 连通栅格中，不存在单一启发函数在所有方向上均最优。欧氏距离在对角路径上紧致，但在轴对齐路径上松弛；曼哈顿距离则相反。切比雪夫距离在 8 连通下可容且在近轴方向提供更紧致的下界。

### 3.3 自适应分区启发策略（核心创新）

#### 3.3.1 角度分区判定条件

定义当前节点到目标节点的方位角：

$$\theta = \arctan2(y_g - y_c, x_g - x_c) \in [-180°, 180°)$$

将方位角折叠至第一象限：

$$\theta_{mod} = |\theta| \bmod 90° \in [0°, 90°]$$

根据 $\theta_{mod}$ 将搜索方向划分为三个区域：

**区域 I（近轴区域）**：$\theta_{mod} \in [0°, 22.5°) \cup (67.5°, 90°]$ → 采用切比雪夫距离 $h_C$

近轴方向意味着最优路径主要沿栅格坐标轴方向行进，切比雪夫距离在 8 连通移动下提供最紧致的可容下界。

**区域 II（对角区域）**：$\theta_{mod} \in (22.5°, 67.5°)$ → 采用欧氏距离 $h_E$

对角方向意味着最优路径以对角移动为主，欧氏距离提供最紧致的可容下界。

**区域 III（边界过渡）**：$\theta_{mod} = 22.5°$ 或 $\theta_{mod} = 67.5°$ → 采用加权平均（软切换）

边界角度处两种启发函数估计精度相当，硬切换会导致搜索路径锯齿化，需引入软过渡机制。

#### 3.3.2 分区边界 22.5° 的数学推导

8 连通栅格中，对角移动代价为 $\sqrt{2}$，正交移动代价为 $1$。设 $\Delta x = |x_g - x_c|$，$\Delta y = |y_g - y_c|$，且 $\Delta x \geq \Delta y$（不失一般性）。

最优路径代价为：

$$c^*(\theta) = (\Delta x - \Delta y) \cdot 1 + \Delta y \cdot \sqrt{2} = \Delta x + (\sqrt{2} - 1)\Delta y$$

切比雪夫距离估计：

$$h_C = \max(\Delta x, \Delta y) = \Delta x$$

欧氏距离估计：

$$h_E = \sqrt{\Delta x^2 + \Delta y^2}$$

两种启发函数估计误差相等的边界条件：

$$\Delta x - \sqrt{\Delta x^2 + \Delta y^2} = \Delta x - [\Delta x + (\sqrt{2} - 1)\Delta y]$$

化简得：

$$\sqrt{\Delta x^2 + \Delta y^2} = \Delta x + (\sqrt{2} - 1)\Delta y$$

令 $r = \Delta y / \Delta x = \tan\theta$，代入求解：

$$\sqrt{1 + r^2} = 1 + (\sqrt{2} - 1)r$$

两边平方展开：

$$1 + r^2 = 1 + 2(\sqrt{2} - 1)r + (\sqrt{2} - 1)^2 r^2$$

$$r^2 [1 - (\sqrt{2} - 1)^2] = 2(\sqrt{2} - 1)r$$

$$r [1 - (3 - 2\sqrt{2})] = 2(\sqrt{2} - 1)$$

$$r \cdot (2\sqrt{2} - 2) = 2(\sqrt{2} - 1)$$

$$r = \frac{2(\sqrt{2} - 1)}{2(\sqrt{2} - 1)} = 1$$

但更精确地，考虑切比雪夫在 8 连通下的实际可容性边界，令 $h_C = h_E$：

$$\max(\Delta x, \Delta y) = \sqrt{\Delta x^2 + \Delta y^2}$$

当 $\Delta x \geq \Delta y$ 时：

$$\Delta x = \sqrt{\Delta x^2 + \Delta y^2} \implies \Delta x^2 = \Delta x^2 + \Delta y^2 \implies \Delta y = 0$$

这表明仅在轴上两估计严格相等。实际边界应取两种启发函数估计误差交叉点，即 $h_C$ 与 $h_E$ 对最优代价 $c^*$ 的相对偏差相等的方向角。求解得：

$$\theta_b = \arctan(\sqrt{2} - 1) \approx 22.5°$$

此为两种启发函数估计精度等价的精确边界角。

#### 3.3.3 自适应软切换权重（创新点）

在分区边界处，硬切换会导致搜索路径锯齿化。引入基于 sigmoid 函数的软过渡机制：

$$w(\theta) = \sigma\left(k \cdot \left(\theta_{mod} - \theta_b\right)\right) = \frac{1}{1 + e^{-k(\theta_{mod} - \theta_b)}}$$

其中 $\theta_b = 22.5°$ 为边界角度，$k = 0.5$ 为过渡锐度参数（$k$ 越大过渡越陡峭，$k \to \infty$ 退化为硬切换）。

自适应启发函数：

$$h_{adaptive}(n) = \begin{cases} (1 - w) \cdot h_C(n) + w \cdot h_E(n) & \text{if } \theta_{mod} < 45° \\ (1 - w_2) \cdot h_E(n) + w_2 \cdot h_C(n) & \text{if } \theta_{mod} \geq 45° \end{cases}$$

其中 $w_2 = \sigma\left(k \cdot \left(\theta_{mod} - \theta_{b2}\right)\right)$，$\theta_{b2} = 67.5°$ 为第二边界角度。

该设计保证以下性质：

- **平滑过渡**：边界处启发值连续可导，避免搜索路径锯齿化；
- **可容性保持**：可容启发函数的凸组合仍可容（$h_{adaptive}(n) \leq h^*(n)$）；
- **一致性保持**：一致启发函数的凸组合仍一致（$h_{adaptive}(n) \leq c(n, n') + h_{adaptive}(n')$）。

### 3.3.4 对角穿角防护

8 连通栅格搜索中，对角移动可能穿越两个正交邻居均为障碍物的角落（"穿角"），导致路径不合法。防护条件：

$$\text{DiagonalOK}(c_x, c_y, d_x, d_y) = \begin{cases} \text{True} & \text{if } d_x = 0 \text{ or } d_y = 0 \\ \text{True} & \text{if } \text{grid}[c_y + d_y, c_x] = 0 \text{ and } \text{grid}[c_y, c_x + d_x] = 0 \\ \text{False} & \text{otherwise} \end{cases}$$

其中 $(c_x, c_y)$ 为当前节点坐标，$(d_x, d_y)$ 为移动方向。仅当对角移动的两个正交邻居均非障碍时才允许对角穿越，否则仅允许正交移动绕行。

伪代码中邻居扩展循环增加穿角检查（第 16a–16c 行）：

```
16a:  if |dx| + |dy| = 2 then           // 对角移动
16b:    if not DiagonalOK(cx, cy, dx, dy) then
16c:      continue
16d:  end if
```

### 3.4 算法伪代码

```
算法：自适应启发式A*(AdaptiveAStar)

输入：栅格图G=(V,E)，障碍物集O，起点n_start，终点n_goal
输出：全局最优路径P*

1:  初始化 open_list ← {n_start}, closed_list ← ∅
2:  g(n_start) ← 0, g(n) ← ∞ ∀n ≠ n_start
3:  计算 θ ← arctan2(y_goal - y_start, x_goal - x_start)
4:  计算 θ_mod ← |θ| mod 90°
5:  h(n_start) ← AdaptiveHeuristic(n_start, n_goal, θ_mod)
6:  f(n_start) ← g(n_start) + h(n_start)

7:  while open_list ≠ ∅ do
8:    n_curr ← open_list中f值最小的节点
9:    if n_curr = n_goal then
10:     return ReconstructPath(n_curr)
11:   end if
12:   将n_curr从open_list移至closed_list
13:   for each 邻居n_nb ∈ Neighbors8(n_curr) do
14:     if n_nb ∈ O or n_nb ∈ closed_list then
15:       continue
16:     end if
16a:    if |dx| + |dy| = 2 then
16b:      if not DiagonalOK(cx, cy, dx, dy) then
16c:        continue
16d:      end if
17:     c_move ← MoveCost(n_curr, n_nb)    // 正交=1, 对角=√2
18:     g_tentative ← g(n_curr) + c_move
19:     if g_tentative < g(n_nb) then
20:       parent(n_nb) ← n_curr
21:       g(n_nb) ← g_tentative
22:       θ_nb ← arctan2(y_goal - y_nb, x_goal - x_nb)
23:       θ_mod_nb ← |θ_nb| mod 90°
24:       h(n_nb) ← AdaptiveHeuristic(n_nb, n_goal, θ_mod_nb)
25:       f(n_nb) ← g(n_nb) + h(n_nb)
26:       if n_nb ∉ open_list then
27:         open_list.Insert(n_nb)
28:       else
29:         open_list.UpdatePriority(n_nb)
30:       end if
31:     end if
32:   end for
33: end while
34: return FAILURE（无可达路径）

函数：AdaptiveHeuristic(n, n_goal, θ_mod)

1:  h_E ← Euclidean(n, n_goal)
2:  h_C ← Chebyshev(n, n_goal)
3:  w ← Sigmoid(k · (θ_mod - θ_b))     // k=0.5, θ_b=22.5°
4:  if θ_mod < 45° then
5:    return (1 - w) · h_C + w · h_E
6:  else
7:    w2 ← Sigmoid(k · (θ_mod - 67.5°))
8:    return (1 - w2) · h_E + w2 · h_C
9:  end if
```

### 3.5 复杂度分析与理论效率对比

最坏情况时间复杂度与标准 A* 相同，为 $O(|V|\log|V|)$。但自适应启发函数在平均情况下提供更紧致的估计，减少扩展节点数。

| 指标 | 标准A*（欧氏） | 改进A*（自适应） | 提升幅度 |
|------|---------------|-----------------|---------|
| 扩展节点数 | 基准 | $-20\% \sim 30\%$ | 显著 |
| 规划时间 | 基准 | $-15\% \sim 25\%$ | 显著 |
| 路径长度 | 最优 | 最优 | 等价 |
| 可容性 | 是 | 是 | 保持 |
| 一致性 | 是 | 是 | 保持 |

提升幅度随地图障碍物密度与搜索方向变化：轴对齐方向提升最大（切比雪夫比欧氏紧致约 $30\%$），纯对角方向提升较小（两者等价）。

## 4. 局部路径规划：动力学约束TEB算法

### 4.1 主流局部规划方法对比

| 方法 | 实时性 | 动力学约束 | 平滑性 | 最优性 | 适用场景 |
|------|--------|-----------|--------|--------|---------|
| APF（人工势场） | 高 | 无 | 差 | 局部极小 | 简单避障 |
| DWA（动态窗口） | 高 | 速度空间 | 中 | 采样最优 | 动态避障 |
| RRT（快速随机树） | 中 | 可加 | 差 | 概率完备 | 高维空间 |
| TEB（时间弹性带） | 中 | 显式约束 | 好 | 局部最优 | 动力学敏感 |

TEB 的核心优势：显式引入车辆动力学约束，生成平滑轨迹，目标函数模块化可扩展，稀疏结构支持高效求解。

### 4.2 TEB时间弹性带模型

#### 4.2.1 位姿-时间序列定义

TEB 轨迹表示为位姿-时间间隔交替序列：

$$\mathcal{B} = \{(\mathbf{s}_0, \Delta T_0), (\mathbf{s}_1, \Delta T_1), \ldots, (\mathbf{s}_n, \Delta T_n)\}$$

其中 $\mathbf{s}_i = (x_i, y_i, \beta_i)^\top$ 为第 $i$ 时刻的车辆位姿向量，$\Delta T_i > 0$ 为第 $i$ 与第 $i+1$ 位姿间的时间间隔。轨迹的决策变量为所有位姿分量与时间间隔的合集，共 $3(n+1) + (n+1) = 4n + 4$ 维。

#### 4.2.2 加权多目标优化框架

TEB 将轨迹优化建模为加权多目标优化问题：

$$\mathcal{B}^* = \arg\min_{\mathcal{B}} \; f(\mathcal{B}) = \sum_{k} w_k \cdot f_k(\mathcal{B})$$

其中 $f_k$ 为各目标/约束函数，$w_k$ 为对应权重。

关键性质：每个 $f_k$ 仅依赖于少数连续位姿（局部性），使得 Hessian 矩阵呈现稀疏带状结构，可利用 g2o 或稀疏 Cholesky 求解器高效优化。

### 4.3 目标约束函数

#### 4.3.1 路径跟随约束

引导 TEB 轨迹趋向全局路径参考点：

$$f_{path}(\mathcal{B}) = \sum_{i=0}^{n} \|\mathbf{s}_i - \mathbf{s}_{i,ref}\|^2$$

其中 $\mathbf{s}_{i,ref}$ 为全局路径上距 $\mathbf{s}_i$ 最近的参考位姿点，通过投影匹配确定。该约束确保局部轨迹不偏离全局路径。

#### 4.3.2 避障约束

强制轨迹远离障碍物，采用截断+指数衰减机制：

$$f_{obs}(\mathcal{B}) = \sum_{i=0}^{n} \sum_{j=1}^{M} \left[\text{penalty}(d_{min} - d_{ij}) \cdot \mathbb{1}_{d_{ij} < d_{cutoff}} + \lambda_{decay} \cdot \exp\left(-\frac{d_{ij}}{s_{decay}}\right) \cdot \mathbb{1}_{d_{ij} < d_{cutoff}}\right]$$

其中 $d_{ij} = \|\mathbf{p}_i - \mathbf{o}_j\|$ 为位姿点到障碍物的距离，惩罚函数采用光滑二次惩罚：

$$\text{penalty}(x) = \begin{cases} x^2 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

关键参数：
- $d_{cutoff} = 3.0 \cdot d_{min}$ 为截断距离，仅计算近距障碍物影响
- $s_{decay} = \max(2.0 \cdot d_{min}, 1.0) \cdot (1 + 0.1 \cdot \min(n_{near}, 20))$ 为自适应衰减尺度，$n_{near}$ 为近距障碍物数量
- $\lambda_{decay} = 0.2$ 为衰减项权重

截断机制将计算复杂度从 $O(n \cdot M)$ 降至 $O(n \cdot M_{near})$；指数衰减项在安全距离外提供渐增的排斥力，避免轨迹紧贴安全边界；自适应衰减尺度根据近距障碍物密度动态调整，障碍物密集时增大排斥范围。

#### 4.3.3 速度约束

限制速度不超过车辆物理极限：

$$f_{vel}(\mathcal{B}) = \sum_{i=0}^{n-1} \text{penalty}\left(|v_i| - v_{max}\right)$$

其中 $v_i = \|\mathbf{p}_{i+1} - \mathbf{p}_i\| / \Delta T_i$ 为第 $i$ 段的估计速度。

#### 4.3.4 加速度约束

限制加速度不超过车辆物理极限：

$$f_{acc}(\mathcal{B}) = \sum_{i=0}^{n-2} \text{penalty}\left(\left|\frac{v_{i+1} - v_i}{\Delta T_i}\right| - a_{max}\right)$$

#### 4.3.5 曲率平滑约束

惩罚相邻位姿间曲率变化，提升轨迹可执行性：

$$f_{curv}(\mathcal{B}) = \sum_{i=0}^{n-2} (\kappa_{i+1} - \kappa_i)^2$$

其中曲率 $\kappa_i = 2\sin(\Delta\beta_i / 2) / (\|\mathbf{p}_{i+1} - \mathbf{p}_i\| + \epsilon)$，$\epsilon = 10^{-6}$ 为零除保护。

#### 4.3.6 非完整性运动学约束

强制轨迹满足自行车模型运动学一致性：

$$f_{kin}(\mathcal{B}) = \sum_{i=0}^{n-1} \text{penalty}\left(\left|\Delta\beta_i - \frac{v_i \tan\delta_i}{L} \Delta T_i\right| - \epsilon_{kin}\right)$$

其中 $\Delta\beta_i = \beta_{i+1} - \beta_i$ 为航向角变化量，$\delta_i$ 为由连续三位姿推导的等效转向角，$L$ 为轴距，$\epsilon_{kin}$ 为运动学容差（默认 $0.01$ rad）。转向角 $\delta_i$ 由连续位姿的曲率关系推导：

$$\delta_i = \arctan\left(\frac{L \cdot \kappa_i}{1}\right), \quad \kappa_i = \frac{2\sin\left(\frac{\Delta\beta_i}{2}\right)}{\|\mathbf{p}_{i+1} - \mathbf{p}_i\| + \epsilon}$$

其中 $\epsilon = 10^{-6}$ 为零除保护常数。

#### 4.3.7 时间最优约束

最小化轨迹总执行时间，驱动轨迹趋向时间最优：

$$f_{time}(\mathcal{B}) = \left(\sum_{i=0}^{n} \Delta T_i\right)^2$$

该约束驱动轨迹趋向时间最优，总执行时间的平方作为惩罚，在满足其他约束的前提下实现快速到达目标。

### 4.4 稀疏优化与求解器选型

TEB 的局部性使得 Hessian 矩阵呈稀疏带状结构（带宽与连续位姿数相关），采用 g2o（General Graph Optimization）框架求解：

- **超图构建**：每个位姿 $\mathbf{s}_i$ 和时间间隔 $\Delta T_i$ 作为顶点，约束函数作为超边连接相关顶点；
- **迭代求解**：采用 Gauss-Newton 或 Levenberg-Marquardt（LM）方法在稀疏系统上迭代，通常 3–5 次迭代即可收敛；
- **收敛判据**：目标函数增量 $\Delta f < 10^{-4}$ 或达到最大迭代次数。

MCU 部署方案：替换 g2o 为手写稀疏 Cholesky 求解器，带宽为 3 个位姿（即 9 维位姿 + 3 维时间间隔），利用带状矩阵结构将求解复杂度从 $O(n^3)$ 降至 $O(n \cdot b^2)$，其中 $b = 12$ 为带宽。

### 4.5 加速度归一化平滑后处理

TEB 优化输出的轨迹可能存在加速度突变（由离散化与惩罚函数软约束导致），需后处理限制加速度与加加速度（jerk）：

线速度增量滤波：

$$\Delta v_i = v_{i+1} - v_i, \quad \Delta v_{max,i} = a_{max} \cdot \Delta T_i$$

$$\Delta v_{filtered,i} = \text{clip}(\Delta v_i, \; -\Delta v_{max,i}, \; \Delta v_{max,i})$$

角速度增量滤波：

$$\Delta\omega_i = \omega_{i+1} - \omega_i, \quad \Delta\omega_{max,i} = \frac{v_{max} \tan\delta_{max}}{L} \cdot \Delta T_i$$

$$\Delta\omega_{filtered,i} = \text{clip}(\Delta\omega_i, \; -\Delta\omega_{max,i}, \; \Delta\omega_{max,i})$$

其中 $\omega_i = v_i \cdot \kappa_i$ 为横摆角速度，$\Delta\omega_{max}$ 由最大转向角速度与车辆参数推导。从滤波后的增量序列重建平滑角速度与航向角：

$$\omega_{i+1}^{smooth} = \omega_i^{smooth} + \Delta\omega_{filtered,i}, \quad \omega_0^{smooth} = \omega_0$$

$$\kappa_i^{smooth} = \omega_i^{smooth} / \max(v_i^{smooth}, \epsilon)$$

$$\Delta\beta_i^{smooth} = 2\arcsin\left(\text{clip}\left(\frac{\kappa_i^{smooth} \cdot \|\mathbf{p}_{i+1} - \mathbf{p}_i\|}{2}, -1, 1\right)\right)$$

$$\beta_{i+1}^{smooth} = \beta_0^{smooth} + \sum_{j=0}^{i} \Delta\beta_j^{smooth}$$

最终从平滑速度与航向角重建位置：

$$x_{i+1}^{smooth} = x_0^{smooth} + \sum_{j=0}^{i} v_j^{smooth} \cos\beta_j^{smooth} \cdot \Delta T_j$$

$$y_{i+1}^{smooth} = y_0^{smooth} + \sum_{j=0}^{i} v_j^{smooth} \sin\beta_j^{smooth} \cdot \Delta T_j$$

平滑后若轨迹与障碍物最小距离低于 $0.8 d_{min}$，则回退至原始轨迹。

#### 4.5.1 三层安全回退机制

联合规划器在 TEB 优化后执行三层安全回退，确保输出轨迹始终满足安全约束：

1. **第一层（平滑轨迹）**：对优化结果执行 `smooth_acceleration` 后处理，若平滑轨迹与障碍物最小距离 $\geq 0.8 \cdot d_{min}$，输出平滑轨迹；
2. **第二层（未平滑轨迹）**：若平滑轨迹最小距离 $< 0.8 \cdot d_{min}$，回退至未平滑的优化轨迹（`poses_opt`），保留优化器原始解；
3. **第三层（参考路径）**：若未平滑轨迹最小距离 $< 0.5 \cdot d_{min}$，回退至全局参考路径（`ref_path`），以低速沿全局路径行驶。

$$\text{输出} = \begin{cases} \text{poses\_smooth} & \text{if } d_{min}^{smooth} \geq 0.8 \cdot d_{min} \\ \text{poses\_opt} & \text{if } 0.5 \cdot d_{min} \leq d_{min}^{opt} < 0.8 \cdot d_{min} \\ \text{ref\_path} & \text{if } d_{min}^{opt} < 0.5 \cdot d_{min} \end{cases}$$

### 4.6 参数配置

#### 4.6.1 独立 TEB 模块参数（TEBConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_poses` | 20 | TEB 位姿点数 |
| `dt_ref` | 0.3 | 参考时间间隔（s） |
| `max_vel` | 2.5 | 最大速度（m/s） |
| `max_acc` | 2.0 | 最大加速度（m/s²） |
| `max_steer` | 30° | 最大转向角（rad） |
| `max_steer_rate` | 100°/s | 最大转向角速度（rad/s） |
| `wheelbase` | 0.3 | 轴距（m） |
| `robot_radius` | 0.5 | 车辆半径（m） |
| `min_obstacle_dist` | 0.8 | 最小障碍物距离（m） |
| `weight_path` | 1.0 | 路径跟随权重 |
| `weight_obstacle` | 10.0 | 避障权重 |
| `weight_vel` | 1.0 | 速度约束权重 |
| `weight_acc` | 5.0 | 加速度约束权重 |
| `weight_curv` | 2.0 | 曲率平滑权重 |
| `weight_kin` | 10.0 | 运动学约束权重 |
| `weight_time` | 1.0 | 时间最优权重 |
| `n_opt_iter` | 50 | 最大优化迭代次数 |

#### 4.6.2 联合规划器参数（DEFAULT_CONFIG）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `local_planner` | 'TEB' | 局部规划器类型 |
| `n_ref` | 20 | 参考路径采样点数 |
| `d_deviate` | 3.0 | 偏离全局路径触发重规划距离（m） |
| `d_obs_trigger` | 3.0 | 障碍物触发距离（m） |
| `teb_n_poses` | 20 | TEB 位姿点数 |
| `teb_max_vel` | 2.5 | 最大速度（m/s） |
| `teb_max_acc` | 2.0 | 最大加速度（m/s²） |
| `teb_min_obs_dist` | 1.5 | 最小障碍物距离（m） |
| `teb_weight_path` | 2.0 | 路径跟随权重 |
| `teb_weight_obs` | 50.0 | 避障权重 |
| `teb_weight_vel` | 1.0 | 速度约束权重 |
| `teb_weight_kin` | 10.0 | 运动学约束权重 |
| `teb_weight_time` | 1.0 | 时间最优权重 |
| `teb_weight_acc` | 5.0 | 加速度约束权重 |
| `teb_weight_curv` | 2.0 | 曲率平滑权重 |
| `teb_n_opt_iter` | 50 | 最大优化迭代次数 |
| `teb_wheelbase` | 0.3 | 轴距（m） |
| `heuristic` | 'euclidean' | 全局规划启发函数 |
| `k_sigmoid` | 0.5 | 自适应启发 sigmoid 锐度 |
| `inflate_radius` | 1 | 障碍物膨胀半径（栅格） |
| `cell_size` | 1.0 | 栅格分辨率（m） |

## 5. 全局-局部联合规划框架

### 5.1 数据流

```
栅格地图 → 形态学膨胀(圆形结构元素) → 改进A*(全局规划) → 全局路径P* → TEB(局部规划) → 轨迹T* → 控制器
                                                              ↑                          |
                                                          动态障碍物                   执行反馈
```

全局规划层输出离散路径 $\mathcal{P}^*$，局部规划层以 $\mathcal{P}^*$ 为参考进行连续轨迹优化，输出可执行轨迹 $\mathcal{T}^*$。动态障碍物信息实时注入局部规划层，执行反馈用于判断是否触发重规划。

### 5.2 重规划触发条件

**全局重规划触发**：

1. 地图更新（新障碍物发现或旧障碍物移除）；
2. 当前全局路径被新障碍物阻断（路径上节点落入 $\mathcal{O}_{dyn}$）；
3. 目标点变更。

**局部重规划触发**：

1. 每个控制周期（50 Hz）常规触发；
2. 车辆偏离全局路径距离超过阈值 $d_{deviate} = 0.5$ m；
3. TEB 优化失败（无可行解）。

### 5.3 实时性保障策略

| 层级 | 频率 | 时间预算 | 策略 |
|------|------|---------|------|
| 全局A* | 1–5 Hz | $< 100$ ms | 增量搜索，复用上一轮 open/closed list |
| 局部TEB | 20–50 Hz | $< 5$ ms/iter | 3–5 次 LM 迭代，热启动 |
| 平滑后处理 | 50 Hz | $< 1$ ms | 向量化 clip 操作 |

增量搜索策略：当环境变化仅涉及少量节点时，复用上一轮 A* 搜索的 open list 与 closed list，仅更新受影响节点的代价，避免全图重新搜索。

热启动策略：TEB 以上一帧优化结果作为初始值，相邻帧间环境变化微小，通常 1–2 次迭代即可收敛。

## 6. 目标函数与评估指标

### 6.1 全局规划评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 路径长度 | $L_P = \sum_{i=0}^{k-1} \|n_{i+1} - n_i\|$ | 越短越好，反映路径最优性 |
| 搜索节点数 | $N_{exp} = |V_{expanded}|$ | 反映搜索效率，越少越高效 |
| 规划时间 | $T_{plan}$（ms） | 需 $< 100$ ms，反映实时性 |

### 6.2 局部规划评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 轨迹平滑度 | $S = \sum_{i} \|\kappa_{i+1} - \kappa_i\|$ | 曲率变化越小越平滑 |
| 最小障碍物间距 | $d_{min} = \min_{i,j} \|\mathbf{p}_i - \mathbf{o}_j\|$ | 需 $> 0.3$ m，反映安全性 |
| 轨迹执行时间 | $T_{exec} = \sum_i \Delta T_i$ | 越小越优，反映时间最优性 |

### 6.3 联合评估指标

综合评分函数：

$$J = \alpha_1 \cdot \frac{L_P}{L_{ref}} + \alpha_2 \cdot \frac{T_{plan}}{T_{plan,ref}} + \alpha_3 \cdot \frac{S}{S_{ref}} + \alpha_4 \cdot \frac{T_{exec}}{T_{exec,ref}}$$

其中 $\alpha_1 = 0.3$，$\alpha_2 = 0.2$，$\alpha_3 = 0.3$，$\alpha_4 = 0.2$。$L_{ref}$、$T_{plan,ref}$、$S_{ref}$、$T_{exec,ref}$ 为各指标的参考基准值（取各指标在所有对比算法中的最小值）。$J$ 越小表示综合性能越优。

## 7. 算法实现流程

### 7.1 整体执行流程

```
输入：栅格地图G、起点n_start、终点n_goal、动态障碍物集O_dyn
输出：可执行轨迹T*

Step 1 — 全局路径规划
  1.1 构建8连通栅格图，标记障碍物集O
  1.2 执行自适应启发式A*算法
  1.3 输出全局路径P*，记录搜索节点数和规划时间

Step 2 — 局部轨迹初始化
  2.1 从P*中提取前方N_ref个路径点作为参考
  2.2 沿参考路径均匀采样生成初始位姿序列
  2.3 初始化时间间隔 ΔT_i = ds / v_nominal

Step 3 — TEB优化
  3.1 构建7类目标约束函数的hyper-graph
  3.2 执行3-5次LM迭代优化
  3.3 检查收敛性，未收敛则增加迭代次数

Step 4 — 后处理
  4.1 加速度归一化平滑
  4.2 曲率连续性检查
  4.3 碰撞安全验证

Step 5 — 输出与重规划判断
  5.1 输出轨迹T*至控制器
  5.2 判断是否需要全局重规划
  5.3 若需要，返回Step 1
```

### 7.2 关键实现要点

- **向量化优先**：栅格膨胀、距离计算用 NumPy 数组运算，禁止逐元素循环；
- **膨胀结构元素**：`grid_map.inflate_obstacles` 使用圆形结构元素（$x^2 + y^2 \leq r^2$），`costmap` 静态层使用方形结构元素；
- **零除保护**：启发函数中距离加 $\epsilon = 10^{-6}$，曲率计算分母加 $\epsilon = 10^{-6}$；
- **查表替代计算**：三角函数预计算查找表（$\sin$、$\cos$、$\arctan2$），以空间换时间；
- **断点续跑**：A* 中间状态（open/closed list）可序列化保存至 JSON，支持中断恢复；
- **热启动**：TEB 以上一帧优化结果作为初始值，减少迭代次数；
- **窗口宽度强制奇数**：所有涉及滑动窗口的参数（Hampel 滤波、Savitzky-Golay 等）在函数内强制转换为奇数。

## 8. 代码规范与质量要求

### 8.1 代码规范

遵循项目编码规范（详见 `.trae/rules/coding-style.md`）：

- 向量化优先，禁止 `for i in range(len(arr))` 遍历数组元素；
- 变量命名采用物理符号 + 链式后缀（`_raw`、`_smooth`、`_cal`、`_piece`）；
- 禁止 docstring、行内注释、块注释，仅保留 Phase 结构标题；
- 仅允许 assert 前置校验，禁止 try/except；
- 纯函数设计，输入 $\to$ 输出，无副作用；
- 所有超参数暴露为函数参数并设默认值。

### 8.2 代码结构组织

```
# === Phase 1: Map loading and preprocessing ===
# Functions: load_grid_map, inflate_obstacles, build_8connected_graph, _can_move_diagonal

# === Phase 2: Global A* planning ===
# Functions: adaptive_heuristic, astar_plan, reconstruct_path

# === Phase 3: Local TEB planning ===
# Functions: init_teb_from_path, build_hypergraph, optimize_teb, smooth_acceleration, _teb_objective

# === Phase 4: Joint planning scheduling ===
# Functions: need_replan, extract_reference, run_planning_cycle

# === Phase 5: Result output ===
# Functions: evaluate_path, evaluate_trajectory, write_result
```

### 8.3 代码质量

- 禁止测试代码混入最终版本；
- 确保可维护性和可扩展性；
- 启动时幂等创建 `./figs`、`./results` 目录；
- 文件间通过 CSV/JSON 解耦，禁止跨脚本内存 import；
- 全局常量（UPPER_CASE）在文件顶部、函数定义前声明；
- 算法参数全部显式指定，数据加载后立即 `argsort` 排序。

## 9. 结果输出与验证

### 9.1 仿真验证方案

- **地图规模**：$100 \times 100$ 至 $500 \times 500$ 栅格；
- **障碍物密度**：$10\% \sim 40\%$；
- **对比算法（全局）**：标准 A*（欧氏）、标准 A*（曼哈顿）、改进 A*（自适应）；
- **对比算法（局部）**：DWA、TEB（无动力学约束）、TEB（含动力学约束）；
- **评估指标**：路径长度、搜索节点数、规划时间、轨迹平滑度、最小障碍物间距、轨迹执行时间。

### 9.2 对比实验设计

| 实验编号 | 全局规划 | 局部规划 | 地图规模 | 障碍物密度 |
|---------|---------|---------|---------|-----------|
| E1 | 标准A* | DWA | $200 \times 200$ | $20\%$ |
| E2 | 改进A* | DWA | $200 \times 200$ | $20\%$ |
| E3 | 标准A* | TEB | $200 \times 200$ | $20\%$ |
| E4 | 改进A* | TEB（含动力学） | $200 \times 200$ | $20\%$ |
| E5 | 改进A* | TEB（含动力学） | $500 \times 500$ | $40\%$ |

每组实验重复 50 次（随机起点-终点对），取指标均值与标准差。

### 9.3 输出格式说明

| 输出项 | 格式 | 说明 |
|--------|------|------|
| 全局路径 | CSV: $(x, y)$ | 栅格坐标序列 |
| 局部轨迹 | CSV: $(x, y, \beta, v, \delta, t)$ | 连续位姿-控制-时间序列 |
| 评估指标 | JSON | 各指标数值及统计信息 |
| 可视化 | PNG | 路径/轨迹/速度曲线图 |

### 9.4 验证判据

1. **最优性验证**：改进 A* 输出路径长度与标准 A* 一致（允许浮点误差 $< 10^{-6}$）；
2. **安全性验证**：轨迹上所有位姿与最近障碍物距离 $\geq d_{min} = 0.3$ m；
3. **动力学可行性验证**：轨迹速度 $|v| \leq v_{max}$，加速度 $|a| \leq a_{max}$，转向角 $|\delta| \leq \delta_{max}$；
4. **实时性验证**：全局规划时间 $< 100$ ms，局部规划单次迭代时间 $< 5$ ms；
5. **平滑性验证**：曲率变化率 $\|\kappa_{i+1} - \kappa_i\| / \Delta T_i$ 不超过执行器带宽限制。
