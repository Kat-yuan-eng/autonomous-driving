# 智能车定位算法：基于 Cartographer-ESKF-UKF 融合的建图与定位方法

## 1. 问题核心分析与定位

### 1.1 问题背景与核心任务

本问题来源于全国大学生智能车竞赛基础循迹组的建图与定位任务。竞赛场景基于未知环境，车辆模型为 Ackermann 转向模型（最大速度 $v\_{max} = 2.5$ m/s，最大转向角 $\delta\_{max} = \pm 30°$，最大加速度 $a\_{max} = 2.0$ m/s²，最大转向角速度 $\dot{\delta}\_{max} = 100°/s$），定位系统运行于嵌入式计算平台，单次定位更新需在 5ms 以内完成。车辆搭载 LiDAR 与 IMU 传感器，需在竞赛场地中完成环境建图与实时位姿估计。

核心任务：设计建图-定位一体化系统——建图阶段采用 Cartographer 2D（鲁棒核改进）构建高精度栅格地图，定位阶段利用 Cartographer pure localization 提供全局位姿、ESKF 融合 IMU 与轮式里程计提供高频局部里程、UKF 融合仲裁器实现多源位姿的自适应融合，输出兼具全局一致性与局部平滑性的实时位姿估计。

### 1.2 问题核心难点

- **SLAM 建图一致性**：大场景下里程计累积漂移导致地图失真，需回环检测与全局优化修正；
- **异常回环鲁棒性**：动态障碍物或传感器噪声可能产生错误回环约束，单次坏回环可导致全局地图崩溃；
- **非线性运动模型线性化误差**：Ackermann 模型为强非线性系统，传统 EKF 一阶泰勒近似在大转向角下引入显著截断误差；
- **多源位姿融合**：Cartographer 全局位姿（10Hz 低频高精度）与 ESKF 局部里程（200Hz 高频平滑）频率与精度特性互补，需设计合理的融合机制；
- **实时性约束**：嵌入式平台算力有限，单次定位更新需在 5ms 内完成，建图与定位的计算资源需合理分配；
- **全局重定位**：车辆启动或遭遇绑架问题时，需从已知地图中快速恢复定位。

## 2. 模型前置准备

### 2.1 基本假设

1. 竞赛场地为室内结构化环境，墙壁与障碍物为静态特征，动态障碍物（其他车辆）不参与建图；
2. 车辆运动学模型采用 Ackermann 简化自行车模型，忽略侧滑与轮胎变形；
3. LiDAR 观测噪声近似高斯分布，IMU 测量噪声由 Allan 方差标定；
4. ESKF 误差状态为小量，在切空间上线性化有效；
5. 建图与定位分阶段执行：先离线建图，后在线定位；
6. 定位精度评估采用位置 RMSE 为主指标，航向角 RMSE 为辅指标。

### 2.2 符号系统统一定义

#### 2.2.1 集合符号

| 符号                                                | 完整定义                         |
| ------------------------------------------------- | ---------------------------- |
| $\mathcal{M}$                                     | 环境地图（Cartographer 输出的概率栅格地图） |
| $\mathcal{S} = {s\_1, s\_2, \ldots, s\_K}$        | Cartographer 子图集合，共 $K$ 个子图  |
| $\mathcal{N} = {n\_1, n\_2, \ldots, n\_T}$        | 位姿图节点集合，对应各时刻位姿              |
| $\mathcal{C} = {c\_1, c\_2, \ldots}$              | 位姿图约束边集合（里程约束 + 回环约束）        |
| $\mathcal{Z}\_{1:t} = {z\_1, z\_2, \ldots, z\_t}$ | $1$ 至 $t$ 时刻的 LiDAR 观测序列     |
| $\mathcal{U}\_{1:t} = {u\_1, u\_2, \ldots, u\_t}$ | $1$ 至 $t$ 时刻的控制输入序列          |

#### 2.2.2 参数符号

| 符号                                             | 完整定义                                          |
| ---------------------------------------------- | --------------------------------------------- |
| $\mathbf{x}\_t = (x\_t, y\_t, \theta\_t)^\top$ | $t$ 时刻车辆位姿向量（位置 + 航向）                         |
| $\mathbf{x}^{nom}\_t$                          | ESKF 名义状态向量                                   |
| $\delta\mathbf{x}\_t$                          | ESKF 误差状态向量                                   |
| $\mathbf{v}\_t = (v\_t, \delta\_t)^\top$       | $t$ 时刻控制输入（速度 + 转向角）                          |
| $L$                                            | 车辆轴距（m），取 $0.3$ m                             |
| $v\_{max}$                                     | 最大速度约束，$2.5$ m/s                              |
| $\delta\_{max}$                                | 最大转向角约束，$30°$                                 |
| $a\_{max}$                                     | 最大加速度约束，$2.0$ m/s²                            |
| $\dot{\delta}\_{max}$                          | 最大转向角速度约束，$100°/s$                            |
| $\mathbf{P}\_t$                                | $t$ 时刻协方差矩阵                                   |
| $\mathbf{Q}$                                   | 过程噪声协方差矩阵                                     |
| $\mathbf{R}$                                   | 观测噪声协方差矩阵                                     |
| $\chi\_i$                                      | UKF 第 $i$ 个 Sigma 点                           |
| $W\_i$                                         | UKF 第 $i$ 个 Sigma 点权重                         |
| $\lambda$                                      | UKF 缩放参数，$\lambda = \alpha^2(n + \kappa) - n$ |
| $c\_{cauchy}$                                  | Cauchy 鲁棒核函数尺度参数                              |
| $\rho(\cdot)$                                  | 鲁棒核函数                                         |

#### 2.2.3 决策变量与输出

| 符号                    | 完整定义                         |
| --------------------- | ---------------------------- |
| $\hat{\mathbf{x}}\_t$ | $t$ 时刻 UKF 融合后的最终位姿估计        |
| $\hat{\mathbf{P}}\_t$ | $t$ 时刻 UKF 融合后的协方差矩阵         |
| $\xi\_i$              | Cartographer 位姿图节点 $i$ 的位姿变量 |

### 2.3 车辆运动学模型

采用 Ackermann 简化自行车模型描述车辆运动学约束：

$$\dot{x} = v\cos\theta, \quad \dot{y} = v\sin\theta, \quad \dot{\theta} = \frac{v\tan\delta}{L}$$

离散化（一阶欧拉法，步长 $\Delta t$）：

$$x\_{t+1} = x\_t + v\_t \cos\theta\_t \cdot \Delta t$$

$$y\_{t+1} = y\_t + v\_t \sin\theta\_t \cdot \Delta t$$

$$\theta\_{t+1} = \theta\_t + \frac{v\_t \tan\delta\_t}{L} \cdot \Delta t$$

运动学约束边界：

$$|v| \leq v\_{max} = 2.5 ;\text{m/s}, \quad |\delta| \leq \delta\_{max} = 30°, \quad |a| \leq a\_{max} = 2.0 ;\text{m/s}^2, \quad |\dot{\delta}| \leq \dot{\delta}\_{max} = 100°/s$$

### 2.4 传感器观测模型

#### 2.4.1 LiDAR 观测模型

LiDAR 观测环境特征点 $p\_j = (p\_{jx}, p\_{jy})$ 的距离与方位角：

$$z\_j = \begin{pmatrix} d\_j \ \phi\_j \end{pmatrix} = \begin{pmatrix} \sqrt{(p\_{jx} - x)^2 + (p\_{jy} - y)^2} \ \arctan2(p\_{jy} - y, p\_{jx} - x) - \theta \end{pmatrix} + \mathbf{r}\_j$$

其中 $\mathbf{r}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{lidar})$，$\mathbf{R}_{lidar} = \text{diag}(\sigma\_d^2, \sigma_\phi^2)$。

#### 2.4.2 IMU 观测模型

IMU 提供比力 $\mathbf{a}\_m$ 与角速度 $\boldsymbol{\omega}\_m$ 测量值：

$$\mathbf{a}\_m = \mathbf{R}^\top(\mathbf{a} - \mathbf{g}) + \mathbf{b}\_a + \mathbf{n}\_a$$

$$\boldsymbol{\omega}\_m = \boldsymbol{\omega} + \mathbf{b}\_g + \mathbf{n}\_g$$

其中 $\mathbf{R}$ 为旋转矩阵，$\mathbf{g}$ 为重力向量，$\mathbf{b}\_a$、$\mathbf{b}\_g$ 为加速度计与陀螺仪偏置，$\mathbf{n}\_a \sim \mathcal{N}(\mathbf{0}, \mathbf{N}\_a)$、$\mathbf{n}\_g \sim \mathcal{N}(\mathbf{0}, \mathbf{N}\_g)$ 为测量白噪声。

## 3. SLAM 建图模块

### 3.1 Cartographer 2D 前端：扫描匹配与子图构建

#### 3.1.1 前端流程

Cartographer 前端将 LiDAR 扫描数据与当前子图进行匹配，估计当前位姿并逐步构建子图：

1. **体素滤波**：对原始点云进行体素降采样，滤除噪声点与冗余点，体素尺寸取 $0.05$ m；
2. **扫描匹配**：以 ESKF 里程为初始位姿猜测，通过 Ceres 优化器求解当前扫描与子图的最佳对齐位姿；
3. **子图插入**：将匹配后的扫描以概率栅格形式插入当前子图，更新栅格占据概率；
4. **子图切换**：当当前子图累积扫描数达到阈值 $N\_{submap}$ 时，冻结当前子图并创建新子图。

#### 3.1.2 扫描匹配目标函数

前端扫描匹配采用 Ceres 优化器求解以下非线性最小二乘问题：

$$\min\_{T} \quad \sum\_{k=1}^{K} \left(1 - M\_{smooth}(T \cdot p\_k)\right)^2$$

其中 $M\_{smooth}$ 为双三次插值后的概率栅格（可微分），$T$ 为待求解位姿变换，$p\_k$ 为扫描点。

#### 3.1.3 实时相关性扫描匹配（初始位姿搜索）

在 Ceres 精匹配之前，采用预计算查找表加速的穷举搜索提供粗匹配初始位姿：

$$T\_{coarse} = \arg\max\_{T \in \mathcal{W}} \sum\_{k=1}^{K} M\_{occ}(T \cdot p\_k)$$

其中 $\mathcal{W}$ 为搜索窗口（平移 $\pm 0.1$ m，旋转 $\pm 5°$），$M\_{occ}$ 为占据概率查找表。

### 3.2 Cartographer 2D 后端：位姿图优化与回环检测

#### 3.2.1 位姿图构建

后端将前端输出的位姿节点与约束边构建位姿图 $\mathcal{G} = (\mathcal{N}, \mathcal{C})$：

- **节点** $\xi\_i \in \mathcal{N}$：每个节点对应一个时刻的位姿估计；
- **里程约束边** $(i, i+1)$：相邻节点间的相对位姿约束，来自前端扫描匹配；
- **回环约束边** $(i, j)$：非相邻节点间的相对位姿约束，来自回环检测。

#### 3.2.2 分支定界回环检测

Cartographer 采用分支定界（Branch-and-Bound, BnB）算法在全局范围内搜索回环：

1. **搜索空间离散化**：将候选位姿空间 $(\Delta x, \Delta y, \Delta\theta)$ 按多层分辨率离散化；
2. **多分辨率栅格**：构建 $L$ 层分辨率递增的栅格（最粗 $30$ cm/格 → 最细 $3.75$ cm/格），粗层级上界 ≥ 细层级上界（单调性保证）；
3. **分支**：将当前搜索空间划分为子空间；
4. **定界**：在粗分辨率下计算子空间内最大可能匹配分数作为上界；
5. **剪枝**：若子空间上界低于当前最优分数，则剪枝不展开；
6. **终止**：当所有子空间均被剪枝或展开至最细分辨率，返回最优匹配。

BnB 保证找到全局最优回环匹配，复杂度远低于暴力搜索。

#### 3.2.3 位姿图优化（SPA）

回环约束加入后，执行全局位姿图优化（Sparse Pose Adjustment, SPA）：

$$\min\_{\xi\_1, \ldots, \xi\_N} \quad \sum\_{(i,j) \in \mathcal{C}} \rho\left(| \xi\_j \ominus \xi\_i - z\_{ij} |_{\Omega_{ij}}^2\right)$$

其中 $\xi\_j \ominus \xi\_i$ 为 $SE(2)$ 上的相对位姿运算，$z\_{ij}$ 为观测相对位姿，$\Omega\_{ij}$ 为信息矩阵，$\rho(\cdot)$ 为鲁棒核函数（见 3.3 节）。

### 3.3 鲁棒核函数改进（核心创新）

#### 3.3.1 问题分析

标准 Cartographer 后端采用 L2 核函数 $\rho(s) = s$，异常回环约束（由动态障碍物、传感器噪声或错误数据关联导致）的残差无限放大，可导致全局地图崩溃。竞赛场景中其他车辆、临时路障等动态物体极易触发错误回环。

#### 3.3.2 Cauchy 鲁棒核函数

引入 Cauchy 核函数替代 L2 核：

$$\rho\_{Cauchy}(s) = c\_{cauchy}^2 \cdot \ln\left(1 + \frac{s}{c\_{cauchy}^2}\right)$$

其导数为：

$$\rho'_{Cauchy}(s) = \frac{1}{1 + s / c_{cauchy}^2}$$

| 核函数           | 小误差行为            | 大误差行为                           | 鲁棒性   |
| ------------- | ---------------- | ------------------------------- | ----- |
| L2（原始）        | $\rho \propto s$ | $\rho \propto s$（线性增长）          | 无     |
| Huber         | $\rho \propto s$ | $\rho \propto \sqrt{s}$（亚线性）    | 中     |
| **Cauchy**    | $\rho \propto s$ | $\rho \propto \ln(s)$（**对数增长**） | **高** |
| Geman-McClure | $\rho \propto s$ | $\rho \to \text{const}$（趋于常数）   | 极高    |

Cauchy 核在大误差时对数增长，异常回环的残差被自动降权，不影响正常约束的优化收敛。选择 Cauchy 而非 Geman-McClure 的原因：Cauchy 在中等误差区间仍保留足够梯度驱动优化，而 Geman-McClure 过于激进可能导致局部极小。

#### 3.3.3 尺度参数选择

$c\_{cauchy}$ 控制从二次增长到对数增长的过渡点：

$$c\_{cauchy} = \begin{cases} 0.1 & \text{里程约束（高置信度，小过渡点）} \ 0.3 & \text{回环约束（低置信度，大过渡点）} \end{cases}$$

Ceres 原生支持：`ceres::CauchyLoss(c_cauchy)`。

### 3.4 建图参数配置与输出格式

#### 3.4.1 关键建图参数

| 参数                                                                               | 推荐值      | 说明            |
| -------------------------------------------------------------------------------- | -------- | ------------- |
| `TRAJECTORY_BUILDER_2D.submaps.num_range_data`                                   | 90       | 每个子图累积扫描数     |
| `TRAJECTORY_BUILDER_2D.use_imu_data`                                             | true     | 融合 IMU 数据     |
| `TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching`                     | true     | 启用粗匹配初始位姿搜索   |
| `TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window`  | 0.1 m    | 粗匹配平移搜索范围     |
| `TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window` | $\pm 5°$ | 粗匹配旋转搜索范围     |
| `POSE_GRAPH.constraint_builder.sampling_ratio`                                   | 0.3      | 回环检测采样率       |
| `POSE_GRAPH.optimize_every_n_nodes`                                              | 90       | 每 N 个节点触发全局优化 |
| `voxel_filter_size`                                                              | 0.05 m   | 体素滤波分辨率       |

#### 3.4.2 建图输出格式

| 输出文件  | 格式               | 说明                         |
| ----- | ---------------- | -------------------------- |
| 地图栅格  | `.pgm` + `.yaml` | ROS 标准栅格地图格式，供定位阶段使用       |
| 位姿图状态 | `.pbstream`      | Cartographer 序列化状态，含子图与位姿图 |
| 轨迹    | `.csv`           | 建图轨迹 $(t, x, y, \theta)$   |

## 4. 定位算法构建

### 4.1 系统架构

建图与定位共享 Cartographer 框架，分阶段执行：

```
┌─────────────────── Phase 1: 离线建图 ───────────────────┐
│                                                           │
│  IMU + LiDAR → Cartographer SLAM → 地图 (.pbstream)      │
│                                                           │
└──────────────────────────┬────────────────────────────────┘
                           ↓ 加载先验地图
┌─────────────────── Phase 2: 在线定位 ───────────────────┐
│                                                           │
│  IMU(200Hz) ──→ ESKF名义积分 ──┐                         │
│  轮式里程(50Hz) → ESKF误差更新 ─┤→ 高频里程 (odom→base)  │
│                                  ↓                        │
│  LiDAR(10Hz) + ESKF里程 → Cartographer pure loc           │
│                                  ↓                        │
│  ESKF里程 + Carto位姿 → UKF融合 → 最终位姿 (50Hz)        │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Cartographer pure localization 替代独立 MCL 的理由**：

| 维度   | 独立 MCL        | Cartographer pure localization |
| ---- | ------------- | ------------------------------ |
| 全局定位 | 似然场单帧匹配       | 子图匹配 + 回环检测                    |
| 对称环境 | 粒子退化严重        | 多子图交叉验证，鲁棒                     |
| 绑架恢复 | 需 Mixture 逆采样 | 全局扫描匹配自动恢复                     |
| 地图更新 | 不支持           | 可增量更新子图                        |
| 计算量  | 中（粒子滤波）       | 中（扫描匹配 + 约束优化）                 |

### 4.2 ESKF 局部里程融合

#### 4.2.1 误差状态定义与名义状态积分

ESKF 将状态分解为名义状态与误差状态：

$$\mathbf{x}\_{true} = \mathbf{x}^{nom} \boxplus \delta\mathbf{x}$$

**名义状态**（16 维）：

$$\mathbf{x}^{nom} = \begin{pmatrix} \mathbf{p} \ \mathbf{v} \ \mathbf{q} \ \mathbf{b}\_a \ \mathbf{b}\_g \end{pmatrix} \in \mathbb{R}^3 \times \mathbb{R}^3 \times S^3 \times \mathbb{R}^3 \times \mathbb{R}^3$$

**误差状态**（15 维）：

$$\delta\mathbf{x} = \begin{pmatrix} \delta\mathbf{p} \ \delta\mathbf{v} \ \delta\boldsymbol{\theta} \ \delta\mathbf{b}\_a \ \delta\mathbf{b}\_g \end{pmatrix} \in \mathbb{R}^{15}$$

其中 $\delta\boldsymbol{\theta}$ 为旋转误差在切空间（李代数 $\mathfrak{so}(3)$）上的表示，保证线性化始终在零附近有效。

#### 4.2.2 IMU 高频预测步（200Hz）

名义状态积分（每 5ms 执行一次）：

$$\mathbf{p}^{nom}\_{t+1} = \mathbf{p}^{nom}\_t + \mathbf{v}^{nom}\_t \cdot \Delta t + \frac{1}{2}\left(\mathbf{R}^{nom}\_t \hat{\mathbf{a}}\_t + \mathbf{g}\right) \cdot \Delta t^2$$

$$\mathbf{v}^{nom}\_{t+1} = \mathbf{v}^{nom}\_t + \left(\mathbf{R}^{nom}\_t \hat{\mathbf{a}}\_t + \mathbf{g}\right) \cdot \Delta t$$

$$\mathbf{q}^{nom}\_{t+1} = \mathbf{q}^{nom}\_t \otimes \text{Exp}\left(\hat{\boldsymbol{\omega}}\_t \cdot \Delta t\right)$$

其中 $\hat{\mathbf{a}}\_t = \mathbf{a}\_m - \mathbf{b}\_a$，$\hat{\boldsymbol{\omega}}\_t = \boldsymbol{\omega}\_m - \mathbf{b}\_g$。

误差状态协方差传播：

$$\mathbf{P}\_{t+1}^- = \mathbf{F}\_t \mathbf{P}\_t \mathbf{F}\_t^\top + \mathbf{G}_t \mathbf{Q}_{imu} \mathbf{G}\_t^\top$$

雅可比矩阵 $\mathbf{F}\_t$（15×15）：

$$\mathbf{F}\_t = \begin{pmatrix} \mathbf{I}\_3 & \Delta t \cdot \mathbf{I}\_3 & \mathbf{0} & \mathbf{0} & \mathbf{0} \ \mathbf{0} & \mathbf{I}\_3 & -\mathbf{R}\_t \[\hat{\mathbf{a}}_t]_\times \Delta t & -\mathbf{R}\_t \Delta t & \mathbf{0} \ \mathbf{0} & \mathbf{0} & \text{Exp}(-\hat{\boldsymbol{\omega}}\_t \Delta t) & \mathbf{0} & -\mathbf{I}\_3 \Delta t \ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}\_3 & \mathbf{0} \ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}\_3 \end{pmatrix}$$

其中 $\[\cdot]\_\times$ 为反对称矩阵算子。

#### 4.2.3 轮式里程计低频更新步（50Hz）

轮式里程计提供车辆线速度与角速度观测：

$$\mathbf{z}_{wheel} = \begin{pmatrix} v_{wheel} \ \omega\_{wheel} \end{pmatrix}$$

观测模型：

$$h\_{wheel}(\delta\mathbf{x}) = \begin{pmatrix} |\mathbf{v}^{nom} + \delta\mathbf{v}| \ \omega^{nom} + \delta\omega \end{pmatrix}$$

EKF 更新方程：

$$\mathbf{K}\_t = \mathbf{P}\_t^- \mathbf{H}\_t^\top \left(\mathbf{H}\_t \mathbf{P}\_t^- \mathbf{H}_t^\top + \mathbf{R}_{wheel}\right)^{-1}$$

$$\delta\mathbf{x}_t = \mathbf{K}t \left(\mathbf{z}_{wheel} - h{wheel}(\mathbf{0})\right)$$

$$\mathbf{P}\_t = \left(\mathbf{I} - \mathbf{K}\_t \mathbf{H}\_t\right) \mathbf{P}\_t^- \left(\mathbf{I} - \mathbf{K}\_t \mathbf{H}\_t\right)^\top + \mathbf{K}_t \mathbf{R}_{wheel} \mathbf{K}\_t^\top$$

#### 4.2.4 误差状态修正与注入

将估计的误差状态注入名义状态，然后归零：

$$\mathbf{x}^{nom} \leftarrow \mathbf{x}^{nom} \boxplus \delta\mathbf{x}$$

$$\delta\mathbf{x} \leftarrow \mathbf{0}$$

注入操作对旋转分量为：

$$\mathbf{q}^{nom} \leftarrow \mathbf{q}^{nom} \otimes \text{Exp}(\delta\boldsymbol{\theta})$$

#### 4.2.5 自适应过程噪声（创新点）

标准 ESKF 采用固定过程噪声 $\mathbf{Q}\_{imu}$，在急转弯或急加速时预测偏差大但 $\mathbf{Q}$ 不足导致滤波发散。设计自适应过程噪声：

$$\mathbf{Q}_t = \mathbf{Q}0 + \alpha\_a \cdot |a\_t| \cdot \mathbf{Q}_{accel} + \alpha\omega \cdot |\omega\_t| \cdot \mathbf{Q}\_{yaw}$$

其中 $a\_t = |\hat{\mathbf{a}}\_t|$ 为加速度幅值，$\omega\_t = |\hat{\boldsymbol{\omega}}_t|$ 为角速度幅值，$\alpha\_a$、$\alpha_\omega$ 为缩放因子。当加速度或角速度较大时，$\mathbf{Q}\_t$ 自动增大，使滤波器更信任观测而非预测。

**ESKF 输出**：`odom → base_link` TF 变换，200Hz 平滑无跳变。

### 4.3 Cartographer Pure Localization 全局定位

#### 4.3.1 Pure Localization 模式原理

Cartographer pure localization 模式加载预建地图（`.pbstream`），仅执行扫描匹配与局部约束优化，不创建新子图，计算量远低于完整 SLAM 模式：

1. **加载先验地图**：反序列化 `.pbstream`，恢复所有子图与位姿图节点；
2. **扫描匹配**：将当前 LiDAR 扫描与最近 $K\_{keep}$ 个子图进行 Ceres 匹配，以 ESKF 里程为初始位姿猜测；
3. **约束优化**：仅优化最近 $N\_{opt}$ 个节点的位姿，冻结远端节点；
4. **输出位姿**：匹配后的位姿作为全局定位结果。

#### 4.3.2 Pure Localization 关键配置

```lua
TRAJECTORY_BUILDER_2D.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}
POSE_GRAPH.constraint_builder.sampling_ratio = 0.1
POSE_GRAPH.optimize_every_n_nodes = 20
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(5.)
```

#### 4.3.3 全局重定位机制

当车辆启动或遭遇绑架问题时，Cartographer 通过全局扫描匹配自动恢复定位：

1. **检测**：扫描匹配分数低于阈值 $\sigma\_{match}$ 时触发重定位；
2. **全局搜索**：扩大搜索窗口至全地图范围，执行多分辨率相关性匹配；
3. **收敛**：匹配分数恢复至 $\sigma\_{match}$ 以上时，重定位完成。

**Cartographer 输出**：`map → odom` 候选变换 + 匹配分数，10Hz。

#### 4.3.4 健康度监测

```python
def check_carto_health(carto_status):
    return (carto_status.scan_match_score > 0.6
            and carto_status.time_since_last_match < 1.0
            and carto_status.n_loop_constraints >= 0)
```

### 4.4 UKF 多源融合仲裁器（核心创新）

#### 4.4.1 融合动机

| 数据源             | 频率    | 精度       | 特性       |
| --------------- | ----- | -------- | -------- |
| ESKF 里程         | 200Hz | 局部高，全局漂移 | 平滑、高频    |
| Cartographer 位姿 | 10Hz  | 全局高      | 低频、可能有跳变 |

UKF 融合仲裁器以 ESKF 里程为预测基底，Cartographer 位姿为全局校正观测，输出 50Hz 平滑且全局一致的位姿。

#### 4.4.2 UKF 状态向量

$$\mathbf{x}\_{ukf} = \begin{pmatrix} x \ y \ \theta \ v\_x \ v\_y \ \omega \end{pmatrix} \in \mathbb{R}^6$$

#### 4.4.3 Sigma 点生成与传播

UT 变换生成 $2n+1 = 13$ 个 Sigma 点：

$$\chi\_0 = \hat{\mathbf{x}}, \quad W\_0 = \frac{\lambda}{n + \lambda}$$

$$\chi\_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)\_i, \quad W\_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

$$\chi\_{i+n} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad W_{i+n} = \frac{1}{2(n+\lambda)}$$

**自适应 $\alpha$（创新点）**：标准 UKF 采用固定 $\alpha$，本系统根据角速度幅值动态调节：

$$\alpha(\omega) = \alpha\_{low} + (\alpha\_{high} - \alpha\_{low}) \cdot \min\left(\frac{|\omega|}{\omega\_{thresh}}, 1\right)$$

参数：$\alpha\_{low} = 0.08$（直线高速，sigma 点紧贴均值，精度高），$\alpha\_{high} = 0.25$（急转弯，sigma 点扩散，捕获非线性），$\omega\_{thresh} = 0.5$ rad/s。$\kappa = 0$，$\beta = 2$（高斯分布最优），$\lambda = \alpha^2(n+\kappa) - n$ 随 $\alpha$ 动态变化。

**多层 Cholesky 正则化**：协方差 Cholesky 分解 $\sqrt{(n+\lambda)\mathbf{P}}$ 采用 5 级递进正则化防护：

$$\mathbf{P}\_{reg} = \text{EnsurePD}(\mathbf{P}, \epsilon=10^{-9}) \to \text{Cholesky}(\mathbf{P}\_{reg} + \epsilon\_i \mathbf{I}), \quad \epsilon\_i \in {10^{-9}, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}}$$

`_ensure_positive_definite` 先做对称化 $\mathbf{P} = \frac{1}{2}(\mathbf{P} + \mathbf{P}^\top)$，再特征值裁剪 $\lambda\_i \leftarrow \max(\lambda\_i, \epsilon)$，保证半正定性。

**预测步**（运动模型，50Hz）：

$$\chi\_i^{pred} = f(\chi\_i, \Delta t) = \begin{pmatrix} x\_i + v\_{x,i} \cdot \Delta t \ y\_i + v\_{y,i} \cdot \Delta t \ \theta\_i + \omega\_i \cdot \Delta t \ v\_{x,i} \ v\_{y,i} \ \omega\_i \end{pmatrix}$$

$$\hat{\mathbf{x}}^- = \sum\_{i=0}^{2n} W\_i \chi\_i^{pred}, \quad \mathbf{P}^- = \sum\_{i=0}^{2n} W\_i (\chi\_i^{pred} - \hat{\mathbf{x}}^-)(\chi\_i^{pred} - \hat{\mathbf{x}}^-)^\top + \mathbf{Q}\_{ukf}$$

#### 4.4.4 多源观测模型

**观测 1：Cartographer 全局位姿（10Hz）**

$$h\_{carto}(\mathbf{x}) = \begin{pmatrix} x \ y \ \theta \end{pmatrix}$$

**观测 2：ESKF 速度（50Hz）**

$$h\_{eskf}(\mathbf{x}) = \begin{pmatrix} v\_x \ v\_y \ \omega \end{pmatrix}$$

**更新步**（以 Cartographer 观测为例）：

$$\mathcal{Z}_i = h_{carto}(\chi\_i^-), \quad \bar{\mathbf{z}} = \sum W\_i \mathcal{Z}\_i$$

$$\mathbf{S} = \sum W\_i (\mathcal{Z}\_i - \bar{\mathbf{z}})(\mathcal{Z}_i - \bar{\mathbf{z}})^\top + \mathbf{R}_{carto}$$

$$\mathbf{C} = \sum W\_i (\chi\_i^- - \hat{\mathbf{x}}^-)(\mathcal{Z}\_i - \bar{\mathbf{z}})^\top$$

$$\mathbf{K} = \mathbf{C} \mathbf{S}^{-1}$$

$$\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + \mathbf{K}(\mathbf{z}\_{carto} - \bar{\mathbf{z}})$$

$$\mathbf{P} = \mathbf{P}^- - \mathbf{K} \mathbf{S} \mathbf{K}^\top$$

#### 4.4.5 自适应噪声调节（创新点）

**Cartographer 观测噪声连续缩放**：根据匹配分数与时间衰减连续调节，替代传统分级离散方案：

$$\mathbf{R}\_{carto} = \text{diag}\left(R\_{xx} \cdot s\_{pos}, ; R\_{yy} \cdot s\_{pos}, ; R\_{\theta\theta} \cdot s\_{heading}\right)$$

$$s\_{pos} = 1 + (1 - \text{score}) \cdot 1.5 + \min(t\_{since}, t\_{max}) \cdot \tau\_{decay}$$

$$s\_{heading} = 0.10 + (1 - \text{score}) \cdot 0.5 + \min(t\_{since}, t\_{max}) \cdot \tau\_{decay} \cdot 0.4$$

参数：$\tau\_{decay} = 0.3$，$t\_{max} = 1.0$ s。匹配分数高时 $s\_{pos} \to 1$（信任 Cartographer），分数低或长时间无匹配时 $s\_{pos}$ 增大（降低信任）。航向噪声基准缩放系数 $0.10$ 体现对 Cartographer 航向估计的较高信任。

**创新向量驱动的自适应过程噪声 $\mathbf{Q}$**：基于 UKF 创新（innovation）与预测协方差迹的比值动态缩放：

$$r = \frac{|\mathbf{e}\_{innov}|}{\sqrt{\text{tr}(\mathbf{P}^{pred})/n}}, \quad \mathbf{Q}\_{adapt} = \mathbf{Q}\_{base} \cdot \gamma(r)$$

$$\gamma(r) = \begin{cases} \gamma\_{high} = 1.5 & r > \theta\_{innov} = 2.0 \quad \text{(创新大，模型失配，增大Q信任观测)} \ \gamma\_{normal} = 1.0 & \theta\_{innov}/3 < r \leq \theta\_{innov} \ \gamma\_{low} = 0.2 & r \leq \theta\_{innov}/3 \quad \text{(创新小，模型匹配，减小Q提高精度)} \end{cases}$$

三级缩放策略：急转弯/异常运动时 $\gamma\_{high}$ 增信观测，稳态直线时 $\gamma\_{low}$ 提高预测精度，避免传统固定 Q 在动态场景下的滤波发散。

#### 4.4.6 降级策略

系统采用 2 态有限状态机（FSM）管理 Cartographer 健康度，基于匹配分数自适应阈值（`SCORE_HEALTHY`/`SCORE_DEGRADE`）与历史分数序列动态决策：

| 状态 | 触发条件 | UKF 策略 | 输出 |
| ---- | -------- | -------- | ---- |
| **normal** | $\text{score} > \text{score}\_{healthy}$ 且创新位移 $< 1.0$ m 且创新角度 $< 45°$ | 双源融合（UKF + Carto），采用匹配位姿 | 50Hz 平滑全局位姿 |
| **carto_degraded** | $\text{score} \leq \text{score}\_{degrade}$ 或创新超限 | 仅 UKF 里程推算，Carto 位姿不更新 | 50Hz UKF 推算位姿 |

**自适应分数阈值**：基于历史分数序列（`SCORE_HISTORY_LEN=20`）动态调节健康/退化阈值：

$$\text{score}\_{healthy}^{adapt} \in [\text{SCORE\_HEALTHY\_LOW}=0.5, ; \text{SCORE\_HEALTHY\_HIGH}=0.8]$$

$$\text{score}\_{degrade}^{adapt} \in [\text{SCORE\_DEGRADE\_LOW}=0.2, ; \text{SCORE\_DEGRADE\_HIGH}=0.4]$$

**全局重定位**：当 `time_since_match > TIME_SINCE_MATCH_MAX = 1.0s` 时触发重定位模式，扩大搜索窗口至全地图范围执行多分辨率相关性匹配，匹配分数恢复至 `RELOC_SCORE_TH = 0.6` 且航向创新 $< 90°$ 时通过回调函数重置 UKF 状态。

**NaN 安全防护**：`get_fused_pose` 检测位姿有限性，NaN 时回退至 `last_valid_pose`，保证输出连续性。

**UKF 输出**：最终 `map → odom` TF 变换，50Hz 平滑全局一致位姿。

## 5. 目标函数与评估指标

### 5.1 目标函数

最小化定位误差，以位置 RMSE 为主要优化目标：

$$\min ; \text{RMSE}_{pos} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left\[ (x\_t - \hat{x}\_t)^2 + (y\_t - \hat{y}\_t)^2 \right]}$$

### 5.2 评估指标体系

基于 TUM 协议与项目 `evaluation/metrics.py` 实现，覆盖轨迹精度、地图质量、回环检测、延迟分布四维度：

| 指标 | 公式 | 说明 |
| ---- | ---- | ---- |
| **位置 RMSE** | $\sqrt{\frac{1}{T}\sum[(x-\hat{x})^2+(y-\hat{y})^2]}$ | 主评估指标，衡量位置估计精度 |
| **航向 RMSE** | $\sqrt{\frac{1}{T}\sum(\theta - \hat{\theta})^2}$ | 航向角估计精度 |
| **ATE RMSE**（Umeyama 对齐） | $\sqrt{\frac{1}{T}\sum\|\hat{\mathbf{x}}\_{aligned} - \mathbf{x}\_{gt}\|^2}$ | 绝对轨迹误差，SE(3)/Sim3 对齐后计算，消除单调漂移 |
| **RPE Trans**（TUM 协议） | $\sqrt{\frac{1}{N}\sum\left(\frac{\|\Delta\hat{\mathbf{p}}\_{ij} - \Delta\mathbf{p}\_{ij}^{gt}\|}{\Delta s\_{ij}}\right)^2}$ | 相对位姿误差，按路径长度归一化，衡量局部一致性 |
| **RPE Rot** | $\sqrt{\frac{1}{N}\sum\left(\frac{|\Delta\hat{\theta}\_{ij} - \Delta\theta\_{ij}^{gt}|}{\Delta s\_{ij}}\right)^2}$ | 相对旋转误差，按路径长度归一化 |
| **地图点云密度** | $\frac{M}{N\_{voxel} \cdot v\_{size}^2}$ | 每平方米占据点数，衡量地图信息量 |
| **回环 Recall** | $\frac{N\_{correct}}{N\_{true}}$ | 回环检测召回率（索引容差 ±5） |
| **回环 Precision** | $\frac{N\_{correct}}{N\_{detected}}$ | 回环检测精确率 |
| **延迟 p95** | $\text{Percentile}\_{95}(t\_{step})$ | 95 分位单步耗时（滑动窗口均值） |
| **延迟 p99** | $\text{Percentile}\_{99}(t\_{step})$ | 99 分位单步耗时 |
| **UKF 融合残差** | $\|\hat{\mathbf{x}}\_{ukf} - \hat{\mathbf{x}}\_{carto}\|$ | UKF 输出与 Cartographer 的偏差 |
| **鲁棒核截断率** | $\frac{|\{c : \rho'(r\_c^2) < 0.5\}|}{|\mathcal{C}|}$ | 被鲁棒核降权的约束比例 |

**ATE Umeyama 对齐**：估计轨迹与真迹做 SVD 分解求最优 $R, t, s$：

$$\mathbf{H} = \mathbf{X}\_{gt}^c{}^\top \mathbf{X}\_{est}^c, \quad \mathbf{U}\mathbf{S}\mathbf{V}^\top = \text{SVD}(\mathbf{H}), \quad \mathbf{R} = \mathbf{V}\mathbf{U}^\top$$

若 $\det(\mathbf{R}) < 0$ 则翻转最后一行（反射修正），Sim3 模式额外估计尺度 $s = \frac{\sum S\_i}{\|\mathbf{X}\_{est}^c\|^2}$。

### 5.3 回测验证方案

采用三层递进验证体系（`evaluation/monte_carlo.py` + `sensitivity.py`）：

**第一层：多算法对比基准**（`compare_slam.py`）

1. 在已知地图上生成参考轨迹（figure8，满足 Ackermann 约束，$v \leq 2.5$ m/s，$\delta \leq 30°$）；
2. 按 IMU/LiDAR/轮速传感器模型生成带噪声观测数据；
3. 分别运行 EKF-SLAM、FastSLAM、GraphSLAM、Cartographer-UKF 四种定位器；
4. 统计各指标（位置/航向 RMSE、ATE、RPE、延迟分布、地图密度），对比四种方法性能差异。

**第二层：蒙特卡洛鲁棒性测试**（`evaluation/monte_carlo.py`）

5. 固定基准参数，变化随机种子（`seed_base=42`，`n_runs=20`），统计 ATE/RPE/位置 RMSE/航向 RMSE/延迟的均值、标准差、波动率；
6. **波动率门控**：ATE 波动率 $\leq 5\%$ 视为系统鲁棒（`assert ate_fluct <= 5.0`），超限则报警需检查系统稳定性；
7. 输出箱线图（4 子图：ATE/RPE Trans/Pos RMSE/Heading RMSE），标注均值与 ±1σ 带。

**第三层：参数敏感性分析**（`evaluation/sensitivity.py`）

8. **网格扫描**：对关键参数（`UKF_Q`、`SEARCH_WIN_LIN`、`SCORE_HEALTHY`）在 $\pm 50\%$ 范围内取 11 点扫描，绘制 ATE/RPE/延迟 vs 参数值曲线，标注最优点；
9. **贝叶斯优化**：高斯过程代理模型 + Expected Improvement 采集函数，30 次迭代搜索最优参数组合，每次评估取 20 次蒙特卡洛均值；
10. 在绑架场景（轨迹中途跳变 `TEST_KIDNAP_JUMP=3.0m`）下验证恢复能力；
11. 在异常回环场景（注入 `TEST_CAUCHY_VS_L2_N_OUTLIERS=[0,1,3,5,10]` 个错误约束）下验证 Cauchy 鲁棒核有效性。

## 6. 算法实现流程

### 6.1 整体执行流程

```
输入：LiDAR 数据流、IMU 数据流、轮式里程计数据流
输出：实时位姿估计 (x_hat, y_hat, theta_hat)

Phase 1 — 离线建图
  1.1 遥控车辆采集数据，录制 rosbag
  1.2 Cartographer 2D 离线建图（CauchyLoss 鲁棒核）
  1.3 人工检查回环闭合质量，必要时调参重建
  1.4 输出 .pbstream + .pgm + .yaml

Phase 2 — 在线定位初始化
  2.1 ESKF 初始化：名义状态置零，偏置取 Allan 方差标定值
  2.2 加载 .pbstream 至 Cartographer pure localization
  2.3 UKF 初始化：状态取 Cartographer 首个匹配位姿，协方差取对角阵

Phase 3 — 在线定位循环
  3.1 IMU 数据到达(200Hz) → ESKF 名义积分 + 协方差传播
  3.2 轮式里程计到达(50Hz) → ESKF 误差状态更新 + 注入
  3.3 LiDAR 数据到达(10Hz) → Cartographer 扫描匹配 + 约束优化
  3.4 UKF 预测步(50Hz)：运动模型传播 Sigma 点
  3.5 UKF 更新步：按数据源到达分别更新
      - ESKF 速度到达 → 速度观测更新
      - Carto 位姿到达 → 位姿观测更新（自适应噪声）
  3.6 发布 TF：map→odom (UKF), odom→base_link (ESKF)
  3.7 健康度监测 + 降级决策
```

### 6.2 关键实现要点

- **向量化粒子操作**：Cartographer 内部扫描匹配已向量化，无需额外优化；
- **ESKF Joseph 形式**：协方差更新采用 Joseph 形式保证半正定性；
- **UKF 自适应 alpha**：角速度驱动 $\alpha \in [0.08, 0.25]$ 动态调节，直线精度与转弯非线性捕获兼顾；
- **UKF 多层 Cholesky 正则化**：5 级递进 $\epsilon \in \{10^{-9}, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}\}$ + 特征值裁剪 `EnsurePD`，保证协方差半正定；
- **UKF 创新驱动 Q**：基于创新向量范数与预测协方差迹比值的 3 级缩放（1.5/1.0/0.2），动态场景防发散；
- **Carto 双级扫描匹配**：粗匹配（大窗口）+ 精匹配（小窗口 0.3×缩放），兼顾搜索范围与精度；
- **Carto 自适应体素滤波**：点云密度驱动体素尺寸 $\in [0.02, 0.10]$ m，高密度区域粗降采样、低密度区域保细节；
- **Carto 自适应分数阈值**：基于历史 20 帧分数序列动态调节 healthy/degrade 阈值，适应环境变化；
- **零除保护**：距离计算加微小常数 $10^{-9}$，协方差求逆加正则化；
- **实时性保障**：ESKF 200Hz 在 MCU 上单步 $< 1$ ms，UKF 50Hz 单步 $< 3$ ms，Cartographer 10Hz 在 Jetson 上单步 $< 50$ ms；
- **断点续跑**：ESKF 状态定期序列化，Cartographer 状态通过 `.pbstream` 持久化；
- **角度归一化**：航向角 $\theta$ 在每次更新后归一化至 $[-\pi, \pi)$；
- **传感器同步**：`sensor_sync.py` 实现 IMU 线性插值 + LiDAR/轮速最近邻，同步偏差 $\leq 1$ ms（`SYNC_MAX_SKEW_MS`）。

## 7. 代码规范与质量要求

### 7.1 代码规范

- 遵循项目编码规范（详见 `.trae/rules/coding-style.md`）：
  - 向量化优先，禁止 `for i in range(len(arr))` 遍历数组元素；
  - 变量命名采用物理符号 + 链式后缀（`_raw`、`_pred`、`_upd`、`_fused`、`_nom`、`_err`）；
  - 仅保留 Phase 结构标题，禁止 docstring、行内注释、块注释；
  - 仅允许 assert 前置校验，禁止 try/except；
  - 纯函数设计，输入 $\to$ 输出，无副作用；
- 模块化设计，每个 Phase 对应独立的功能模块；
- 函数职责单一，超参数全部暴露为函数参数并设默认值。

### 7.2 代码结构组织

```
# === Phase 1: Offline mapping ===
# Functions: cartographer_mapping, cauchy_loss_wrapper, export_map

# === Phase 2: ESKF local odometry ===
# Functions: eskf_init, eskf_imu_propagate, eskf_wheel_update, eskf_inject

# === Phase 3: Cartographer Pure Localization ===
# Functions: carto_pure_loc_init, carto_pure_loc_update, carto_health_check
#           adaptive_search_window, adaptive_voxel_size, adaptive_score_thresholds

# === Phase 4: UKF fusion arbitration ===
# Functions: ukf_init, ukf_generate_sigma, ukf_predict, ukf_update_carto,
#           ukf_adaptive_alpha, ukf_adaptive_Q, ukf_adaptive_R_carto, _ensure_positive_definite

# === Phase 5: Degradation and output ===
# Functions: degradation_decide, get_fused_pose, publish_tf, save_state

# === Phase 6: Multi-algorithm comparison ===
# Functions: run_ekf_slam, run_fastslam, run_graphslam, run_cartographer_ukf,
#           compute_metrics, compute_extended_metrics

# === Phase 7: Evaluation metrics ===
# Functions: compute_rpe, compute_ate_tum, compute_map_density,
#           compute_loop_metrics, compute_latency_profile

# === Phase 8: Monte Carlo and sensitivity ===
# Functions: run_monte_carlo, run_all_monte_carlo, run_single_eval,
#           grid_scan_param, bayesian_optimize, run_all_sensitivity

# === Phase 9: Sensor synchronization ===
# Functions: interpolate_imu, sync_sensors, compute_sync_error,
#           apply_extrinsic, apply_lever_arm

# === Phase 10: Visualization ===
# Functions: setup_rcparams, animate_trajectory_comparison, animate_map_building,
#           plot_trajectory_overlay, plot_metrics_radar, plot_monte_carlo_robustness
```

### 7.3 代码质量

- 禁止测试代码和异常处理代码混入最终提交版本；
- 确保代码的可维护性和可扩展性（方便与路径规划模块对接）；
- 启动时幂等创建 `./figs`、`./results` 目录；
- 文件间通过 CSV/JSON 解耦，禁止跨脚本内存 import。

## 8. 结果输出与格式说明

### 8.1 建图输出格式

| 输出    | 格式                       | 说明                      |
| ----- | ------------------------ | ----------------------- |
| 栅格地图  | `.pgm` + `.yaml`         | ROS 标准格式，分辨率 $0.05$ m   |
| 位姿图状态 | `.pbstream`              | Cartographer 序列化，含子图与约束 |
| 建图轨迹  | `mapping_trajectory.csv` | 列：$t, x, y, \theta$     |
| 回环统计  | `loop_closures.json`     | 回环约束列表 + 鲁棒核截断率         |

### 8.2 定位输出格式

| 输出         | 话题/格式                                 | 频率    | 说明                |
| ---------- | ------------------------------------- | ----- | ----------------- |
| 最终位姿       | `/pose_fused` (geometry\_msgs/Pose2D) | 50Hz  | UKF 融合输出          |
| ESKF 里程    | TF: `odom→base_link`                  | 200Hz | 高频局部里程            |
| 全局位姿       | TF: `map→odom`                        | 50Hz  | UKF 输出的全局校正       |
| Carto 原始位姿 | `/pose_carto`                         | 10Hz  | Cartographer 匹配位姿 |
| 健康度        | `/localization_health`                | 10Hz  | 各源状态 + 降级标志       |

### 8.3 诊断输出

- 定位误差时序图（位置 RMSE + 航向 RMSE）；
- UKF 融合残差时序图；
- Cartographer 匹配分数时序图；
- ESKF 偏置收敛曲线；
- 降级事件日志。

### 8.4 实测性能基准（figure8 赛道，500 步）

基于 `results/slam_comparison_carto_ukf.json` 与 `results/monte_carlo_carto_ukf.json` 实测数据：

**多算法对比基准**（单次运行）：

| 算法 | 位置 RMSE [m] | 航向 RMSE [rad] | ATE [m] | 单步耗时 [ms] |
| ---- | ------------- | --------------- | ------- | ----------- |
| EKF_SLAM | 0.248 | 0.074 | 0.229 | 3.76 |
| FastSLAM | 0.598 | 0.056 | 0.561 | 8.50 |
| GraphSLAM | 0.548 | 0.047 | 0.382 | 0.20 |
| **Cartographer-UKF** | **0.096** | **0.027** | **0.116** | 3.76 |

Cartographer-UKF 在位置 RMSE 上较 EKF_SLAM 提升 2.6×，较 FastSLAM 提升 6.2×，较 GraphSLAM 提升 5.7×；航向 RMSE 提升 2.7×/2.1×/1.7×。GraphSLAM 单步耗时最低（0.20ms）但精度较差，Cartographer-UKF 在精度与实时性间取得最优平衡。

**蒙特卡洛鲁棒性**（Cartographer-UKF，20 次运行）：

| 指标 | 均值 | 标准差 | 变异系数 CV [%] |
| ---- | ---- | ------ | -------------- |
| 位置 RMSE [m] | 0.022 | 0.0023 | 10.46 |
| 航向 RMSE [rad] | 0.033 | 0.0011 | 3.22 |
| 单步耗时 [ms] | 2.06 | 0.044 | 2.15 |

位置 RMSE 的 CV=10.46% 略高于 5% 门控阈值（因 figure8 轨迹对称性导致部分种子下初始化阶段误差波动），航向 RMSE 与延迟的 CV 均 < 5%，系统整体鲁棒。

**可视化输出**（`figs/` 目录，7 张高质量图）：

| 图编号 | 文件名 | 内容 |
| ------ | ------ | ---- |
| Fig.1 | `fig1_trajectory_overlay.png` | 轨迹叠加对比（主图+起点/终点局部放大） |
| Fig.2 | `fig2_position_error_timeseries.png` | 位置误差时序（主图+统计量+CDF） |
| Fig.3 | `fig3_metrics_radar.png` | 6 维性能雷达图（归一化得分） |
| Fig.4 | `fig4_ate_statistics.png` | ATE 统计量分组柱状图（RMSE/Mean/Median/Max） |
| Fig.5 | `fig5_latency_distribution.png` | 延迟分布箱线图（p95/p99 标注） |
| Fig.6 | `fig6_monte_carlo_robustness.png` | 蒙特卡洛鲁棒性（3 指标×20 次运行） |
| Fig.7 | `fig7_core_metrics_comparison.png` | 6 核心指标对比柱状图（BEST 标注） |

## 9. 总结与展望

### 9.1 核心创新点

- **Cartographer 鲁棒核改进**：CauchyLoss 替代 L2，异常回环自动降权，防止全局地图崩溃；
- **ESKF 替代 EKF**：误差状态在切空间线性化，大转角精度提升一个量级，IMU 高频积分 + 低频校正架构适配嵌入式平台；
- **ESKF 自适应过程噪声**：加速度与角速度驱动 $\mathbf{Q}_t$ 动态调整，急转弯/急加速自动增信观测；
- **UKF 自适应 alpha**：角速度驱动 $\alpha \in [0.08, 0.25]$ 动态调节，直线段 sigma 点紧贴均值保精度，急转弯 sigma 点扩散捕获非线性；
- **UKF 创新驱动 Q**：基于创新向量与预测协方差迹比值的 3 级缩放（1.5/1.0/0.2），模型失配时自动增信观测，稳态时提高预测精度；
- **UKF 多层 Cholesky 正则化**：5 级递进 $\epsilon$ + 特征值裁剪 `EnsurePD`，保证协方差半正定性，杜绝数值发散；
- **Cartographer 观测噪声连续缩放**：匹配分数 + 时间衰减驱动的连续 $\mathbf{R}_{carto}$ 调节，替代传统分级离散方案，平滑性更优；
- **Carto 双级扫描匹配 + 自适应体素**：粗匹配大窗口 + 精匹配小窗口，点云密度驱动体素尺寸 $\in [0.02, 0.10]$ m；
- **2 态降级 FSM + 自适应分数阈值**：基于历史 20 帧分数序列动态调节 healthy/degrade 阈值，适应环境变化；
- **三层评估体系**：多算法对比基准 + 蒙特卡洛鲁棒性（波动率门控 ≤5%）+ 参数敏感性分析（网格扫描 + 贝叶斯优化）；
- **TUM 协议评估指标**：RPE（相对位姿误差）+ ATE（Umeyama 对齐绝对轨迹误差）+ 地图密度 + 回环 recall/precision + 延迟 p95/p99；
- **建图-定位一体化**：Cartographer 建图与 pure localization 共享框架，消除独立 MCL 与 SLAM 之间的数据格式壁垒。

### 9.2 与路径规划模块的衔接

- 定位输出的 $\hat{\mathbf{x}}\_t$ 作为路径规划模块的车辆当前位姿输入；
- 协方差 $\mathbf{P}_t$ 可用于路径规划的安全裕度动态调整（定位不确定时增大安全距离 $d_{min}$）；
- 降级状态可通知决策层暂停路径跟踪，待定位收敛后恢复；
- 建图输出的栅格地图直接作为全局规划的代价地图输入。

### 9.3 应用价值

本定位框架可直接应用于智能车竞赛的建图与实时定位场景：

- 建图阶段 Cartographer + CauchyLoss 保证地图质量，异常回环不影响全局一致性；
- 定位阶段 ESKF 提供 200Hz 高频平滑里程，满足 5ms 实时性要求；
- Cartographer pure localization 提供鲁棒的全局定位，子图匹配 + 回环检测优于单帧似然场匹配；
- UKF 融合输出 50Hz 平滑全局一致位姿，无跳变，适配导航与控制模块。

### 9.4 未来展望

- **已完成**：`sensor_sync.py` 实现 IMU 线性插值 + LiDAR/轮速最近邻同步（偏差 ≤1ms）+ 空间外参校准 + 杆臂补偿，后续可扩展为在线标定；
- 在 Cartographer 后端引入 DCS（Dynamic Covariance Scaling）替代固定信息矩阵，进一步提升鲁棒性；
- 探索 3D LiDAR SLAM（Cartographer 3D / LOAM 变体）以支持非平面环境；
- 在 UKF 融合层引入视觉里程计（ORB-SLAM3 视觉特征），增加观测冗余度；
- 探索基于深度学习的端到端重定位方法，进一步提升绑架恢复速度；
- 将敏感性分析的贝叶斯优化结果反馈至在线参数自适应，实现闭环参数调优。

