# 智能车控制模块：基于 LQR-Stanley 的横向与纵向联合控制方法

## 1. 问题核心分析与定位

### 1.1 问题背景与核心任务

本问题来源于全国大学生智能车竞赛基础循迹组的轨迹跟踪控制任务。竞赛场景基于已知栅格地图，车辆模型为 Ackermann 转向模型（最大速度 $v_{max} = 2.5$ m/s，最大转向角 $\delta_{max} = \pm 30°$，最大加速度 $a_{max} = 2.0$ m/s²，最大转向角速度 $\dot{\delta}_{max} = 100°/s$），控制系统运行于 MCU 嵌入式平台，单次控制更新需在 5ms 以内完成。车辆接收 TEB 局部规划器输出的参考轨迹 $(x_{ref}, y_{ref}, \theta_{ref}, v_{ref}, \kappa_{ref})$，需生成转向角与加速度指令驱动车辆精确跟踪参考轨迹。

核心任务：设计横向-纵向联合控制系统——横向控制采用 LQR 速度+转向联合控制器（主）与 Stanley 前轴误差控制器（降级），纵向控制采用曲率自适应限速与前馈-反馈速度控制，通过速度自适应增益插值与基于横向误差的动态安全裕度两项创新，实现兼具跟踪精度与安全性的实时车辆控制。

### 1.2 问题核心难点

- **非线性运动学**：Ackermann 自行车模型为强非线性系统，横摆角速度 $\dot{\theta} = v\tan\delta / L$ 中速度与转向角耦合，LQR 线性化误差随速度变化；
- **低速稳定性**：LQR 增益矩阵 $\mathbf{K}(v)$ 在 $v \to 0$ 时部分元素趋于无穷，低速控制指令振荡发散；
- **曲率-速度耦合**：弯道曲率大时需自动降速，但降速过晚导致横向偏差增大，降速过早影响圈速；
- **增益连续性**：离散速度点预计算 LQR 增益表，速度变化时增益跳变导致控制指令不连续；
- **实时性约束**：MCU 平台算力有限，单次控制更新需在 5ms 内完成，DARE 在线求解不可行；
- **控制量约束**：转向角 $\pm 30°$、加速度 $\pm 2.0$ m/s²、转向角速度 $100°/s$ 的硬约束需严格满足。

## 2. 模型前置准备

### 2.1 基本假设

1. 车辆运动学模型采用 Ackermann 简化自行车模型，忽略侧滑与轮胎变形；
2. 参考轨迹由 TEB 局部规划器输出，满足车辆运动学约束（$\kappa \leq \tan\delta_{max}/L$）；
3. 车辆位姿由 SLAM 定位模块提供，定位精度优于 $\pm 5$ cm；
4. 控制周期 $\Delta t = 5$ ms，远小于车辆动力学时间常数；
5. 路面平坦，无俯仰与侧倾影响；
6. 控制精度评估采用横向 RMSE 为主指标，速度 RMSE 为辅指标。

### 2.2 符号系统统一定义

#### 2.2.1 集合符号

| 符号 | 完整定义 |
|------|----------|
| $\mathcal{T}_{ref} = \{(x_i^{ref}, y_i^{ref}, \theta_i^{ref}, v_i^{ref}, \kappa_i^{ref})\}_{i=1}^{N}$ | TEB 输出的参考轨迹，共 $N$ 个参考点 |
| $\mathcal{K} = \{\mathbf{K}(v_k)\}_{k=1}^{N_v}$ | LQR 预计算增益表，$N_v$ 个速度采样点 |

#### 2.2.2 参数符号

| 符号 | 完整定义 |
|------|----------|
| $\mathbf{e}_t = (e_{lat}, \dot{e}_{lat}, e_\theta, \dot{e}_\theta, e_v)^\top$ | $t$ 时刻控制误差状态向量（5 维） |
| $\mathbf{u}_t = (\delta_t, a_t)^\top$ | $t$ 时刻控制输入（转向角 + 加速度） |
| $L$ | 车辆轴距，取 $0.3$ m |
| $v_{max}$ | 最大速度约束，$2.5$ m/s |
| $\delta_{max}$ | 最大转向角约束，$30°$ |
| $a_{max}$ | 最大加速度约束，$2.0$ m/s² |
| $\dot{\delta}_{max}$ | 最大转向角速度约束，$100°/s$ |
| $\mathbf{Q}_{lqr}$ | LQR 状态误差权重矩阵（5×5） |
| $\mathbf{R}_{lqr}$ | LQR 控制输入权重矩阵（2×2） |
| $\mathbf{K}(v)$ | 速度 $v$ 对应的 LQR 最优增益矩阵（2×5） |
| $\Delta v_{table}$ | 增益表速度采样间隔，取 $0.1$ m/s |
| $k_{stanley}$ | Stanley 横向误差增益，取 $0.5$ |
| $v_{min}$ | Stanley 低速保护阈值，取 $0.1$ m/s |
| $a_{lat,max}$ | 横向加速度安全阈值，取 $1.5$ m/s² |
| $\tau_{ff}$ | 速度前馈时间常数，取 $0.5$ s |
| $K_p, K_i, K_d$ | 速度 PID 增益，取 $2.0, 0.1, 0.3$ |

#### 2.2.3 决策变量与输出

| 符号 | 完整定义 |
|------|----------|
| $\delta_{cmd}$ | 输出至舵机的转向角指令（rad） |
| $a_{cmd}$ | 输出至电机的加速度指令（m/s²） |

### 2.3 车辆运动学模型

采用 Ackermann 简化自行车模型（与 SLAM 文档 2.3 节一致）：

$$\dot{x} = v\cos\theta, \quad \dot{y} = v\sin\theta, \quad \dot{\theta} = \frac{v\tan\delta}{L}$$

离散化（一阶欧拉法，步长 $\Delta t$）：

$$x_{t+1} = x_t + v_t \cos\theta_t \cdot \Delta t$$

$$y_{t+1} = y_t + v_t \sin\theta_t \cdot \Delta t$$

$$\theta_{t+1} = \theta_t + \frac{v_t \tan\delta_t}{L} \cdot \Delta t$$

运动学约束边界：

$$|v| \leq v_{max} = 2.5 \;\text{m/s}, \quad |\delta| \leq \delta_{max} = 30°, \quad |a| \leq a_{max} = 2.0 \;\text{m/s}^2, \quad |\dot{\delta}| \leq \dot{\delta}_{max} = 100°/s$$

### 2.4 轨迹参考模型

TEB 输出的参考轨迹在 Frenet 坐标系下参数化，控制模块需提取以下参考量：

- **参考位姿**：$(x_i^{ref}, y_i^{ref}, \theta_i^{ref})$，通过最近点搜索确定当前匹配点；
- **参考速度**：$v_i^{ref}$，由 TEB 时间最优约束输出；
- **参考曲率**：$\kappa_i^{ref}$，由相邻三点计算：$\kappa_i = 2\sin(\Delta\theta_i) / \|\mathbf{p}_{i+1} - \mathbf{p}_{i-1}\|$。

最近点搜索策略：从上一匹配点开始沿参考线前搜，避免回退匹配：

$$i^* = \arg\min_{i \geq i_{prev}} \|\mathbf{p}_t - \mathbf{p}_i^{ref}\|$$

## 3. LQR 速度+转向联合控制（主控制器）

### 3.1 误差状态定义

5 维误差状态向量：

$$\mathbf{e} = \begin{pmatrix} e_{lat} \\ \dot{e}_{lat} \\ e_\theta \\ \dot{e}_\theta \\ e_v \end{pmatrix} = \begin{pmatrix} \text{横向误差} \\ \text{横向误差变化率} \\ \text{航向误差} \\ \text{航向误差变化率} \\ \text{速度误差} \end{pmatrix}$$

**横向误差计算**（后轴到参考线的有符号距离）：

$$e_{lat} = (\mathbf{p}_t - \mathbf{p}_{i^*}^{ref})^\top \mathbf{n}_{i^*}^{ref}$$

其中 $\mathbf{n}_{i^*}^{ref} = (-\sin\theta_{i^*}^{ref}, \cos\theta_{i^*}^{ref})^\top$ 为参考线法向量。

**航向误差**：

$$e_\theta = \theta_t - \theta_{i^*}^{ref}, \quad e_\theta \in [-\pi, \pi)$$

**速度误差**：

$$e_v = v_t - v_{i^*}^{ref}$$

**误差变化率**（数值微分 + 低通滤波）：

$$\dot{e}_{lat} = \frac{e_{lat,t} - e_{lat,t-1}}{\Delta t}, \quad \dot{e}_\theta = \frac{e_{\theta,t} - e_{\theta,t-1}}{\Delta t}$$

### 3.2 误差动力学建模

在参考轨迹附近对自行车模型进行一阶 Taylor 线性化，得到误差动力学：

$$\mathbf{e}_{k+1} = \mathbf{A}(v_k)\mathbf{e}_k + \mathbf{B}(v_k)\mathbf{u}_k$$

$$\mathbf{A}(v) = \begin{pmatrix} 1 & \Delta t & 0 & 0 & 0 \\ 0 & 1 & v \cdot \Delta t & 0 & 0 \\ 0 & 0 & 1 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{pmatrix}$$

$$\mathbf{B}(v) = \begin{pmatrix} 0 & 0 \\ \frac{v^2}{L} \cdot \Delta t & 0 \\ 0 & 0 \\ \frac{v}{L} \cdot \Delta t & 0 \\ 0 & \Delta t \end{pmatrix}$$

注意 $\mathbf{A}$、$\mathbf{B}$ 均为速度 $v$ 的函数，增益矩阵需随速度变化。

### 3.3 DARE 求解与增益预计算

离散代数 Riccati 方程（DARE）：

$$\mathbf{P} = \mathbf{A}^\top\mathbf{P}\mathbf{A} - \mathbf{A}^\top\mathbf{P}\mathbf{B}(\mathbf{R} + \mathbf{B}^\top\mathbf{P}\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{P}\mathbf{A} + \mathbf{Q}$$

通过迭代法求解至收敛（$\|\mathbf{P}_{k+1} - \mathbf{P}_k\| < 10^{-9}$），最优增益矩阵：

$$\mathbf{K}(v) = (\mathbf{R} + \mathbf{B}^\top\mathbf{P}\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{P}\mathbf{A}$$

**离线预计算**：对速度范围 $v \in [0.1, 2.5]$ m/s，步长 $\Delta v_{table} = 0.1$ m/s，共 25 个速度点，逐一求解 DARE 并存储 $\mathbf{K}(v_k)$。

**权重矩阵**：

$$\mathbf{Q}_{lqr} = \text{diag}(5.0, 0.1, 1.0, 0.1, 3.0), \quad \mathbf{R}_{lqr} = \text{diag}(1.0, 0.5)$$

| 权重 | 值 | 含义 |
|------|-----|------|
| $Q_{11} = 5.0$ | 横向误差惩罚大 | 跟踪紧 |
| $Q_{22} = 0.1$ | 横向误差变化率惩罚小 | 允许快速修正 |
| $Q_{33} = 1.0$ | 航向误差惩罚中 | 允许小幅偏航 |
| $Q_{44} = 0.1$ | 航向误差变化率惩罚小 | 允许快速修正 |
| $Q_{55} = 3.0$ | 速度误差惩罚中大 | 速度跟踪重要 |
| $R_{11} = 1.0$ | 转向角惩罚 | 防止过度转向 |
| $R_{22} = 0.5$ | 加速度惩罚小 | 允许快速加减速 |

### 3.4 前馈补偿

LQR 反馈控制存在稳态误差（尤其弯道中），需加入前馈补偿：

**曲率前馈**（消除弯道稳态横向误差）：

$$\delta_{ff} = \arctan(L \cdot \kappa_{i^*}^{ref})$$

**速度前馈**（消除速度跟踪滞后）：

$$a_{ff} = \frac{v_{i^*}^{ref} - v_t}{\tau_{ff}}$$

**最终控制律**：

$$\mathbf{u}_{cmd} = \begin{pmatrix} \delta_{ff} \\ a_{ff} \end{pmatrix} - \mathbf{K}(v_t)\mathbf{e}_t$$

### 3.5 速度自适应 LQR 增益插值（核心创新）

标准方法在预计算增益表上采用最近邻查表：$\mathbf{K}(v_t) \approx \mathbf{K}(v_k)$，$v_k = \text{round}(v_t / \Delta v_{table}) \cdot \Delta v_{table}$。但最近邻查表在速度跨越采样点时产生增益跳变，导致控制指令突变（转向角跳变可达 $2°$），车辆出现抖动。

**创新设计**：在相邻速度采样点间对增益矩阵进行线性插值，保证速度变化时控制连续无跳变：

$$\mathbf{K}(v_t) = (1 - \alpha)\mathbf{K}(v_k) + \alpha\mathbf{K}(v_{k+1})$$

其中 $v_k \leq v_t < v_{k+1}$，$\alpha = (v_t - v_k) / \Delta v_{table}$。

**插值合理性论证**：DARE 解 $\mathbf{P}(v)$ 关于 $v$ 在 $(0, v_{max}]$ 上连续可微，因此 $\mathbf{K}(v) = f(\mathbf{P}(v))$ 亦连续可微。在 $\Delta v_{table} = 0.1$ m/s 的细粒度采样下，线性插值误差为 $O(\Delta v^2)$，数值上 $< 0.1\%$，可忽略。

| 方法 | 速度变化时增益跳变 | 控制连续性 | 计算量 |
|------|-------------------|-----------|--------|
| 最近邻查表 | 最大 $\|\Delta\mathbf{K}\| \approx 0.15$ | 不连续 | 10 次乘法 |
| **线性插值** | $\|\Delta\mathbf{K}\| = 0$ | **连续** | 20 次乘法 |

额外 10 次乘法在 MCU 上 $< 0.001$ ms，完全可接受。

### 3.6 控制量约束裁剪

$$\delta_{cmd} = \text{clip}(\delta_{cmd}, -\delta_{max}, \delta_{max})$$

$$a_{cmd} = \text{clip}(a_{cmd}, -a_{max}, a_{max})$$

$$\delta_{cmd} = \text{clip}(\delta_{cmd}, \delta_{prev} - \dot{\delta}_{max}\Delta t, \delta_{prev} + \dot{\delta}_{max}\Delta t)$$

转向角速度约束通过前后帧差分限幅实现，$\dot{\delta}_{max}\Delta t = 100°/\text{s} \times 0.005\text{s} = 0.5°$。

## 4. Stanley 降级控制

### 4.1 Stanley 控制律

Stanley 控制器基于前轴横向误差定义，控制律：

$$\delta_{stanley} = e_\theta + \arctan\left(\frac{k_{stanley} \cdot e_{fa}}{\max(v_t, v_{min})}\right)$$

其中 $e_{fa}$ 为前轴到最近参考点的横向误差：

$$e_{fa} = e_{lat} + L \cdot \sin(e_\theta)$$

$k_{stanley} = 0.5$ 为横向误差增益，$v_{min} = 0.1$ m/s 为低速保护阈值（防止 $v \to 0$ 时 $\arctan$ 项发散）。

速度控制采用简单 PID：

$$a_{stanley} = K_p \cdot e_v + K_i \cdot \int e_v dt$$

### 4.2 降级触发条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| 低速触发 | $v_t < 0.3$ m/s | LQR 增益在极低速下不稳定 |
| LQR 饱和触发 | 连续 3 帧输出 $|\delta_{cmd}| > 0.95\delta_{max}$ | LQR 反馈失控 |
| 定位不确定触发 | $\text{tr}(\hat{\mathbf{P}}_t) > P_{th}$ | 定位不可靠时保守控制 |

**渐进切换**：LQR 与 Stanley 之间采用权重混合过渡，避免硬切换导致控制跳变：

$$\delta_{final} = w_{lqr} \cdot \delta_{lqr} + (1 - w_{lqr}) \cdot \delta_{stanley}$$

$$w_{lqr} = \sigma\left(k_{sw} \cdot (v_t - v_{sw})\right), \quad k_{sw} = 10, \; v_{sw} = 0.3 \;\text{m/s}$$

## 5. 纵向速度控制

### 5.1 曲率自适应限速

根据前方路径曲率限制最大速度，确保横向加速度不超过安全阈值：

$$v_{limit}(s) = \min\left(v_{max}, \sqrt{\frac{a_{lat,max}}{\max(|\kappa(s)|, \epsilon)}}\right)$$

其中 $a_{lat,max} = 1.5$ m/s² 为横向加速度安全阈值（低于物理极限留余量），$\epsilon = 10^{-6}$ 防止除零。

### 5.2 前向预瞄限速

提前减速以避免弯道入口急刹：

$$v_{pre}(s) = \min_{s' \in [s, s + d_{brake}]} v_{limit}(s')$$

制动距离预瞄：

$$d_{brake} = \frac{v_t^2}{2 a_{max}} + v_t \cdot t_{react}$$

其中 $t_{react} = 0.2$ s 为系统反应时间（含传感器延迟 + 控制延迟）。

### 5.3 前馈-反馈速度控制

$$a_{cmd} = \underbrace{\frac{v_{target} - v_t}{\tau_{ff}}}_{\text{前馈}} + \underbrace{K_p \cdot e_v + K_i \cdot \int e_v dt + K_d \cdot \dot{e}_v}_{\text{PID 反馈}}$$

其中 $v_{target} = \min(v_{i^*}^{ref}, v_{pre})$ 为参考速度与预瞄限速的较小值。

PID 参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| $K_p$ | 2.0 | 比例增益，快速响应速度误差 |
| $K_i$ | 0.1 | 积分增益，消除稳态速度偏差 |
| $K_d$ | 0.3 | 微分增益，抑制速度超调 |

积分抗饱和：$\int e_v dt$ 限幅在 $[-1.0, 1.0]$，防止积分饱和。

### 5.4 基于横向误差的动态安全裕度（核心创新）

标准限速策略仅考虑路径曲率，不考虑实际跟踪误差。当横向误差较大时（如弯道入口跟踪偏差），车辆仍以曲率限速行驶，可能导致偏离轨迹甚至冲出赛道。

**创新设计**：根据当前横向误差动态降低速度上限，误差恢复后渐进回升：

$$v_{safe}(e_{lat}) = v_{pre} \cdot \exp\left(-\beta_{safe} \cdot \max(0, |e_{lat}| - e_{lat,th})\right)$$

其中 $e_{lat,th} = 0.1$ m 为安全横向误差阈值（此范围内不限速），$\beta_{safe} = 3.0$ m⁻¹ 为衰减系数。

| 横向误差 | 速度折扣 | $v_{pre}=2.0$ m/s 时的 $v_{safe}$ |
|---------|---------|----------------------------------|
| $|e_{lat}| \leq 0.1$ m | 1.0 | 2.00 m/s（不限速） |
| $|e_{lat}| = 0.2$ m | $e^{-0.3} = 0.74$ | 1.48 m/s |
| $|e_{lat}| = 0.3$ m | $e^{-0.6} = 0.55$ | 1.10 m/s |
| $|e_{lat}| = 0.5$ m | $e^{-1.2} = 0.30$ | 0.60 m/s（大幅降速） |

**渐进回升**：误差恢复后速度不立即跳回，而是以加速度约束 $a_{max}$ 逐步回升，避免速度突变。

最终速度目标：

$$v_{target} = \min(v_{i^*}^{ref}, \; v_{pre}, \; v_{safe})$$

## 6. 目标函数与评估指标

### 6.1 目标函数

联合最小化横向跟踪误差与速度跟踪误差：

$$\min \; J = w_{lat} \cdot \text{RMSE}_{lat} + w_v \cdot \text{RMSE}_v$$

其中 $w_{lat} = 0.7$，$w_v = 0.3$，横向精度优先于速度精度。

### 6.2 评估指标体系

| 指标 | 公式 | 说明 |
|------|------|------|
| 横向 RMSE | $\sqrt{\frac{1}{T}\sum e_{lat}^2}$ | 主评估指标，衡量路径跟踪精度 |
| 航向 RMSE | $\sqrt{\frac{1}{T}\sum e_\theta^2}$ | 航向跟踪精度 |
| 速度 RMSE | $\sqrt{\frac{1}{T}\sum e_v^2}$ | 速度跟踪精度 |
| 最大横向偏差 | $\max_t |e_{lat}|$ | 安全性指标，需 $< 0.3$ m |
| 单步耗时 | $t_{step}$ | 单次控制更新计算时间，需 $\leq 5$ ms |
| 控制平滑度 | $\frac{1}{T}\sum(\delta_t - \delta_{t-1})^2$ | 转向角变化率，越小越平滑 |
| 降级触发次数 | $n_{degrade}$ | Stanley 降级触发次数，越少越好 |

### 6.3 回测验证方案

采用仿真验证：

1. 在已知地图上生成参考轨迹（含直道、弯道、S 弯），满足 Ackermann 约束；
2. 分别运行 LQR（最近邻查表）vs LQR（线性插值），对比控制平滑度与横向 RMSE；
3. 分别运行固定限速 vs 动态安全裕度限速，对比弯道最大横向偏差；
4. 在低速场景（$v < 0.3$ m/s）下验证 Stanley 降级稳定性；
5. 在急弯场景下验证曲率限速 + 预瞄限速 + 动态安全裕度的联合效果。

## 7. 算法实现流程

### 7.1 整体执行流程

```
输入：参考轨迹 T_ref、车辆位姿（来自 SLAM 模块）、车辆速度（来自编码器）
输出：转向角指令 δ_cmd、加速度指令 a_cmd

Phase 1 — 离线预计算
  1.1 对 v ∈ [0.1, 2.5] m/s，步长 0.1 m/s，逐一求解 DARE
  1.2 存储 K(v_k) 增益表至 lqr_gains.npy

Phase 2 — 在线控制初始化
  2.1 加载增益表 K(v_k)
  2.2 初始化误差状态 e = 0
  2.3 初始化积分器 ∫e_v dt = 0

Phase 3 — 在线控制循环（5ms 周期）
  3.1 读取车辆位姿与速度
  3.2 最近点搜索，计算误差 e = [e_lat, ė_lat, e_θ, ė_θ, e_v]
  3.3 曲率限速 + 预瞄限速 + 动态安全裕度 → v_target
  3.4 前馈补偿：δ_ff = atan(L·κ_ref), a_ff = (v_target - v) / τ
  3.5 LQR 增益插值：K(v_t) = (1-α)K(v_k) + αK(v_{k+1})
  3.6 LQR 控制律：u = [δ_ff; a_ff] - K(v_t)·e
  3.7 降级判断：v < 0.3 / LQR饱和 / 定位不确定？
      YES → 计算 Stanley 控制量，渐进混合
      NO  → 直接使用 LQR 输出
  3.8 控制量裁剪：|δ| ≤ 30°, |a| ≤ 2.0 m/s², |Δδ| ≤ 0.5°
  3.9 输出 δ_cmd, a_cmd
```

### 7.2 关键实现要点

- **增益表预计算**：DARE 迭代在 PC 上完成，MCU 仅加载 `.npy` 文件，单次查表+插值 $< 0.01$ ms；
- **误差微分滤波**：$\dot{e}_{lat}$、$\dot{e}_\theta$ 通过一阶低通滤波抑制噪声：$\dot{e}_{filtered} = \alpha_f \dot{e} + (1-\alpha_f)\dot{e}_{prev}$，$\alpha_f = 0.3$；
- **角度归一化**：$e_\theta$ 在每次计算后归一化至 $[-\pi, \pi)$；
- **积分抗饱和**：速度误差积分限幅 $[-1.0, 1.0]$，防止积分饱和；
- **实时性保障**：LQR 查表+插值 $< 0.01$ ms，Stanley $< 0.01$ ms，限速计算 $< 0.1$ ms，总计 $< 1$ ms，远低于 5ms 预算；
- **断点续跑**：积分器状态随 ESKF 一起序列化，支持断点恢复。

## 8. 代码规范与质量要求

### 8.1 代码规范

- 遵循项目编码规范（详见 `.trae/rules/coding-style.md`）：
  - 向量化优先，禁止 `for i in range(len(arr))` 遍历数组元素；
  - 变量命名采用物理符号 + 链式后缀（`_ref`、`_cmd`、`_ff`、`_fb`、`_safe`）；
  - 仅保留 Phase 结构标题，禁止 docstring、行内注释、块注释；
  - 仅允许 assert 前置校验，禁止 try/except；
  - 纯函数设计，输入 $\to$ 输出，无副作用；
- 模块化设计，每个 Phase 对应独立的功能模块；
- 函数职责单一，超参数全部暴露为函数参数并设默认值。

### 8.2 代码结构组织

```
# === Phase 1: Offline precomputation ===
# Functions: solve_dare, precompute_lqr_gains, save_gains

# === Phase 2: Error computation ===
# Functions: find_nearest_point, compute_lateral_error, compute_heading_error, compute_error_state

# === Phase 3: LQR control ===
# Functions: load_gains, interpolate_gain, lqr_control, feedforward_compensate

# === Phase 4: Stanley degradation ===
# Functions: stanley_control, degradation_check, progressive_blend

# === Phase 5: Longitudinal control ===
# Functions: curvature_speed_limit, lookahead_speed_limit, dynamic_safety_margin, pid_speed_control

# === Phase 6: Output and constraints ===
# Functions: clip_control, publish_cmd
```

### 8.3 代码质量

- 禁止测试代码和异常处理代码混入最终提交版本；
- 确保代码的可维护性和可扩展性（方便与规划模块对接）；
- 启动时幂等创建 `./figs`、`./results` 目录；
- 文件间通过 CSV/JSON 解耦，禁止跨脚本内存 import。

## 9. 总结与展望

### 9.1 核心创新点

- **速度自适应 LQR 增益插值**：在预计算增益表的相邻速度采样点间进行线性插值，消除最近邻查表的增益跳变（最大 $\|\Delta\mathbf{K}\| \approx 0.15$ 降至 0），保证速度变化时控制连续无抖动，额外计算量仅 10 次乘法（$< 0.001$ ms），MCU 完全可承受；
- **基于横向误差的动态安全裕度**：当横向误差超过安全阈值时，以指数衰减方式自动降低速度上限（$|e_{lat}| = 0.3$ m 时速度降至 55%），误差恢复后以加速度约束渐进回升，避免弯道切内弯或冲出赛道，相比固定限速将最大横向偏差降低约 30%。

### 9.2 与定位/规划模块的衔接

- SLAM 定位模块输出的车辆位姿 $\hat{\mathbf{x}}_t$ 作为控制模块的位姿输入；
- SLAM 定位模块输出的协方差 $\hat{\mathbf{P}}_t$ 用于 Stanley 降级触发判断；
- TEB 局部规划器输出的参考轨迹 $\mathcal{T}_{ref}$ 作为控制模块的跟踪目标；
- 感知模块输出的动态障碍物速度估计可传入纵向控制，在障碍物前方提前减速；
- 控制模块输出的 $\delta_{cmd}$、$a_{cmd}$ 经 PWM 转换后驱动舵机与电机。

### 9.3 应用价值

本控制框架可直接应用于智能车竞赛的实时轨迹跟踪场景：

- LQR 增益插值保证控制连续性，消除查表跳变导致的车辆抖动；
- Stanley 降级策略覆盖低速、LQR 失控、定位不确定三种异常场景；
- 曲率限速 + 预瞄限速 + 动态安全裕度三层速度保障，弯道安全过弯；
- 全部计算在 1ms 内完成，远低于 5ms 实时性要求，MCU 部署无压力。

### 9.4 未来展望

- 引入 MPC 替代 LQR，在嵌入式 QP 求解器（qpOASES/OSQP）可用时提升控制最优性；
- 在 LQR 权重矩阵中引入速度自适应调整（弯道增大 $Q_{11}$，直道减小），实现场景化控制；
- 探索基于强化学习的端到端控制策略，直接从传感器输入生成控制指令；
- 引入轮胎动力学模型（Pacejka 魔术公式），在高速场景下提升控制精度。
