# 智能车系统可视化图表说明

本文档对感知模块与路径规划模块的全部可视化图表进行系统性解读，涵盖图表结构、数据含义、配色方案及分析要点。

---

## 一、路径规划模块图表

所有图表由 `compare_algorithms.py` 和 `joint_planner.py` 生成。

### 1.1 global_comparison.png — 全局规划器性能对比

**布局**：1×3 分组柱状图（`figsize=(18, 6)`）

| 子图 | Y 轴 | 含义 |
|------|------|------|
| 左 | Path Length (cell) | 路径长度，越短越优 |
| 中 | Expanded Nodes | 搜索扩展节点数，越少效率越高 |
| 右 | Planning Time (ms) | 规划耗时，越短实时性越好 |

- **X 轴**：障碍物密度（10% / 20% / 30%）
- **算法配色**：Dijkstra `#e74c3c`（红）、A\*(euclidean) `#3498db`（蓝）、A\*(manhattan) `#2ecc71`（绿）、A\*(chebyshev) `#f39c12`（橙）、AdaptiveA\* `#9b59b6`（紫）
- **柱顶标注**：精确数值（1 位小数）

**分析要点**：Dijkstra 无启发式，扩展节点最多、耗时最长；A\* 系列通过启发函数大幅缩减搜索空间；AdaptiveA\* 在轴对齐方向上利用 Chebyshev 紧致下界、在对角方向上利用 Euclidean 紧致下界，综合效率最优。路径长度五者一致（均为最优），差异体现在搜索效率。

---

### 1.2 search_process_comparison.png — 搜索过程可视化

**布局**：1×2 栅格地图叠加图（`figsize=(16, 8)`）

| 子图 | 标题 | 内容 |
|------|------|------|
| 左 | Dijkstra | 无启发式全向搜索 |
| 右 | AdaptiveA\* | 自适应启发式定向搜索 |

- **背景**：灰度栅格（`gray_r`），白色=自由空间，黑色=障碍物
- **扩展节点**：青色散点（`cyan`, s=2, α=0.4），图例 `expanded (N)`
- **最终路径**：红色实线（linewidth=2）
- **起终点**：绿色方块（start）、蓝色方块（goal）
- **标题动态标注**：扩展节点数 N 与耗时 T ms

**分析要点**：直观对比两种算法的搜索空间覆盖差异。Dijkstra 的青色散点大面积覆盖网格（全向扩展），AdaptiveA\* 的扩展节点集中在目标方向（定向扩展），标题中的 N 和 T 量化效率差距。此图验证自适应启发函数的核心价值：在不牺牲路径最优性的前提下大幅缩减搜索空间。

---

### 1.3 local_comparison.png — 局部规划器多维性能对比

**布局**：2×3 柱状图（`figsize=(16, 10)`）

| 面板 | 指标 | 含义 | 方向 |
|------|------|------|------|
| (a) Smoothness | 曲率变化率归一化 | 路径平滑度 | 越低越优 |
| (b) Safety | 最近障碍物距离归一化 | 安全裕度 | 越高越优 |
| (c) Speed | 计算时间归一化 | 实时性 | 越高越快 |
| (d) Adaptability | 速度/步长一致性 | 动态适应性 | 越高越优 |
| (e) Path Length (m) | 路径原始长度 | 路径效率 | 越短越优 |
| (f) Composite J | 加权综合代价 | 综合性能 | 越低越优 |

- **算法配色**：RRT `#3A86FF`（蓝）、DWA `#FF9E00`（橙）、TEB `#6C757D`（灰）
- **J 公式**：$J = 0.3 \cdot L/L_{ref} + 0.2 \cdot T_{plan}/T_{plan,ref} + 0.3 \cdot S/S_{ref} + 0.2 \cdot T_{exec}/T_{exec,ref}$，参考基准为各指标在所有对比算法中的最小值

**分析要点**：六维度全面对比三种局部规划器。TEB 在 Smoothness 和 Safety 上优势明显（显式动力学约束+避障代价），但 Speed 可能不及 DWA（迭代优化开销）。J 指标将多目标折中为单一排序值，便于算法选型决策。

---

### 1.4 trajectory_comparison.png — 轨迹形态与安全性对比

**布局**：单图叠加（`figsize=(12, 10)`），等比例坐标系

**图层结构**：

| 层 | 内容 | 样式 |
|----|------|------|
| 障碍物 | 8 个圆形障碍物 | 灰色填充，α=0.5 |
| 安全膨胀区 | 障碍物半径 + robot_radius(1.5) | 橙色虚线圆，α=0.12 |
| RRT 轨迹 | 路径线 + 机器人轮廓圆 | 蓝色 `#3498db`，lw=2 |
| DWA 轨迹 | 路径线 + 机器人轮廓圆 | 红色 `#e74c3c`，lw=2 |
| TEB 轨迹 | 路径线 + 机器人轮廓圆 | 绿色 `#2ecc71`，lw=2 |
| 碰撞点 | 碰撞位置（最多 5 个） | 红色 X 标记，ms=10 |
| 信息框 | 各算法 d_min、路径长度、PASS/FAIL | 小麦色圆角背景，monospace |

- **图例格式**：`RRT (d_min=X.XX)` / `DWA (d_min=X.XX)` / `TEB (d_min=X.XX)`

**分析要点**：同一场景下叠加三种算法的实际输出轨迹，视觉对比路径形态差异——RRT 锯齿形（随机采样）、DWA 蜿蜒形（速度空间采样）、TEB 平滑形（连续优化）。安全膨胀圈和碰撞标记直观评估安全裕度，信息框中的 PASS/FAIL 判定基于碰撞检测结果。

---

### 1.5 joint_planning.png — 全局-局部联合规划输出

**布局**：单图叠加（`figsize=(10, 10)`），Y 轴向上递增

**图层结构**：

| 层 | 内容 | 样式 |
|----|------|------|
| 膨胀底图 | `imshow(grid_inflated, cmap='Oranges', α=0.3)` | 橙色半透明 |
| 原始障碍物 | `imshow(grid, cmap='binary', α=0.7)` + 黑色散点 | 黑白二值 |
| 膨胀区域 | 仅膨胀新增部分 | 橙色方块，s=2，α=0.3 |
| 全局路径 | Adaptive A\* 输出 | 蓝色实线，lw=2 |
| 局部轨迹 | TEB 优化输出 | 红色实线，lw=2 |
| 碰撞点 | 距障碍物 <0.5 cell 的轨迹点 | 红色 X 标记 |
| 信息框 | d_min 值 | 小麦色圆角背景 |

- **X/Y 轴**：`x [cell]` / `y [cell]`

**分析要点**：展示联合规划框架的端到端输出。全局规划器在膨胀栅格上生成安全粗路径（蓝色），局部规划器在全局路径参考下精细优化生成可执行轨迹（红色）。膨胀区域（橙色散点）可视化安全膨胀策略的效果，碰撞点标记验证局部轨迹是否真正避障。核心目的是验证"全局引导 + 局部优化"分层架构的协同工作效果。

---

## 二、感知模块图表

所有图表由 `test_perception.py` Phase 3 生成，共享统一配色体系。

### 统一配色

| 元素 | 颜色 | 色值 |
|------|------|------|
| Proposed (SDF) | 亮蓝 | `#3A86FF` |
| Fixed Threshold | 橙色 | `#FF9E00` |
| Grid-NN | 中灰 | `#6C757D` |
| IDSW 标记 | 红色 | `#E63946` |
| 障碍物标记 | 橙色星号 | `#FF9E00` |
| 航迹标注（动态层） | 橙色 | `#FF9E00` |
| 航迹标注（融合层） | 绿色 | `#38B000` |

---

### 2.1 sdf_filter_comparison.png — SDF 自适应过滤效果对比

**布局**：1×3 散点图（`figsize=(15, 5)`）

| 子图 | 标题 | 内容 |
|------|------|------|
| (a) | Raw (after voxel) | 体素滤波后原始点云，无 SDF 过滤 |
| (b) | Fixed Threshold | 固定阈值过滤结果 |
| (c) | SDF Adaptive | 自适应 SDF 过滤结果 |

- **X/Y 轴**：x (m) / y (m)，范围 [0, 5.0]
- **背景**：灰度栅格地图（`Greys`, α=0.6）
- **点云**：蓝色 `#3A86FF`，s=8，α=0.6
- **障碍物**：橙色星号，ms=12
- **标题动态标注**：点数 N 及近墙保留率 `Near-wall retention: xx%`

**分析要点**：微观层对比三种过滤策略在近墙区域的表现。固定阈值在近墙处过度剔除动态点（近墙保留率低），SDF 自适应过滤通过 proximity 加权降低阈值（$\tau = \tau_{sdf} / (1 + \beta_{sdf} \cdot proximity)$），显著提升近墙保留率，同时仍能过滤静态墙壁点。此图是 SDF 自适应机制有效性的直接证据。

---

### 2.2 costmap_visualization.png — 代价地图三层架构可视化

**布局**：1×3 热力图（`figsize=(15, 5)`）

| 子图 | 标题 | 内容 |
|------|------|------|
| (a) | Static Layer | 静态膨胀代价层（方形结构元素，R_INFLATE=0.3m） |
| (b) | Dynamic Layer | 动态高斯代价层（SIGMA_DYN=0.2m，含预测分量 DT_PRED=0.5s, W_PRED=0.5） |
| (c) | Fused Costmap | 融合代价地图（`np.maximum(c_static, c_dyn)`） |

- **X/Y 轴**：x (m) / y (m)
- **色彩映射**：`viridis`，代价值 [0, 1]，附 colorbar
- **航迹标注**：`ID:x` 文字（黑底圆角框）+ 速度方向箭头（速度 >0.02 m/s 时绘制）
- **动态层/融合层标注色**：橙色 / 绿色

**分析要点**：中观层展示代价地图三层架构。静态膨胀层覆盖墙壁周围区域，动态层在移动障碍物当前位置及预测位置产生高斯代价，融合层取两者最大值确保安全。速度箭头直观展示预测方向，验证动态层对运动目标的时空预测能力。

---

### 2.3 tracking_trajectory_timeline.png — 跟踪轨迹时序对比

**布局**：1×2 时序折线图（`figsize=(14, 5)`）

| 子图 | 标题 | Y 轴 |
|------|------|------|
| (a) | Tracking Timeline — near_wall_cross (X) | X position (m) |
| (b) | Tracking Timeline — near_wall_cross (Y) | Y position (m) |

- **X 轴**：Frame（帧序号 0~39）
- **Ground Truth**：黑色实线，lw=2
- **Proposed (SDF)**：蓝色虚线 + 圆形标记（markevery=5）
- **Fixed Threshold**：橙色点线 + 方形标记
- **Grid-NN**：灰色点划线 + 菱形标记
- **IDSW 事件**：红色叉号（`#E63946`, s=60），标记身份切换时刻

**分析要点**：时域上对比三种算法的跟踪轨迹与真值的偏差，选取最具挑战性的近墙交叉场景。重点关注：(1) 近墙交叉场景下各算法的位置跟踪精度；(2) IDSW 事件的发生时机和频率——红色叉号标记身份切换时刻，Proposed 算法应显著减少 IDSW；(3) 航迹连续性——虚线/点线/点划线的断裂反映航迹丢失。

---

### 2.4 comprehensive_comparison.png — 综合性能全景对比

**布局**：2×3 混合图（`figsize=(16, 10)`）

| 面板 | 指标 | Y 轴范围 | 方向 |
|------|------|---------|------|
| (a) Recall | 召回率 | [0, 1.1] | 越高越优 |
| (b) MOTA | 多目标跟踪精度 | [-0.5, 1.1] | 越高越优 |
| (c) Position RMSE (m) | 位置均方根误差 | 截断 2.0m | 越低越优 |
| (d) IDSW Rate | 身份切换率 | — | 越低越优 |
| (e) Avg Time (ms) | 平均处理耗时 | — | 越低越优 |
| (f) Scene Difficulty vs Recall | 散点图 | X=Avg SDF, Y=Recall | — |

- **X 轴 (a-e)**：4 个动态场景（near_wall_single / multi_cross / near_wall_cross / high_speed_curve）
- **分组柱状图**：每场景 3 根柱（Proposed 蓝 / Fixed 橙 / Grid-NN 灰），柱宽 0.25
- **散点图 (f)**：场景难度（障碍物处平均 SDF 值）vs 召回率，每点旁标注场景名

**分析要点**：宏观层跨场景、跨指标的全面性能对比。柱状图直观展示 Proposed 算法在 Recall 和 MOTA 上的优势、RMSE 的降低、IDSW Rate 的减少；散点图 (f) 揭示场景难度（SDF 越小 = 越靠近墙壁 = 越难）与 Recall 的负相关关系，验证 SDF 自适应过滤在困难场景下的鲁棒性提升。

---

### 2.5 improvement_heatmap.png — 算法改善热力图

**布局**：单图热力图（`figsize=(8, 6)`）

- **行（Y 轴）**：4 个动态场景
- **列（X 轴）**：4 项指标（Recall / MOTA / 1÷RMSE / Speed）
- **色彩映射**：`RdBu`（红-白-蓝发散色标），对称色标
- **数值标注**：每格中心标注百分比改善值（`+xx.x%` / `-xx.x%`）
- **改善率计算**：$(proposed - fixed) / \max(|fixed|, 10^{-9}) \times 100\%$
- **指标变换**：RMSE 取倒数 $1/(1+rmse)$ 转为正向；Speed 取倒数 $1/(1+time/10)$ 转为正向

**分析要点**：决策层热力图一目了然地展示 Proposed 算法相对于 Fixed Threshold 基线的全方位改善幅度。蓝色 = 正向改善（Proposed 更优），红色 = 负向退化。重点关注：(1) 近墙场景在 Recall 和 MOTA 上的大幅蓝色改善；(2) Speed 列是否出现红色（SDF 查询额外开销可能稍慢）；(3) 1÷RMSE 列反映定位精度提升。

---

## 三、图表分析逻辑链

### 路径规划模块

```
global_comparison → search_process_comparison → local_comparison → trajectory_comparison → joint_planning
   算法选型          搜索空间验证            多维性能量化        轨迹形态直观对比       端到端集成验证
```

1. **global_comparison**：从路径长度、搜索效率、计算速度三维度筛选最优全局规划器
2. **search_process_comparison**：可视化验证 AdaptiveA\* 搜索空间缩减的物理机制
3. **local_comparison**：六维度量化对比局部规划器，J 指标提供综合排序
4. **trajectory_comparison**：同一场景下直观对比轨迹形态与安全性
5. **joint_planning**：验证全局-局部联合框架的端到端协同效果

### 感知模块

```
sdf_filter_comparison → costmap_visualization → tracking_trajectory_timeline → comprehensive_comparison → improvement_heatmap
     微观过滤              中观代价映射            时序跟踪质量              宏观全景对比              决策量化
```

1. **sdf_filter_comparison**：微观层证明 SDF 自适应过滤在近墙区域保留更多动态点
2. **costmap_visualization**：中观层展示过滤后的点如何转化为代价地图三层结构
3. **tracking_trajectory_timeline**：时序层在最具挑战性的近墙交叉场景中逐帧对比跟踪质量
4. **comprehensive_comparison**：宏观层跨场景、跨指标的全面性能对比
5. **improvement_heatmap**：决策层将所有改善量化为百分比，快速定位优势场景与潜在短板
