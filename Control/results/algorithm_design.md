# LQR-Stanley-PP 三融合复合控制器算法设计文档

## 1. 复合算法总体架构

### 1.1 架构概述

本控制模块采用 **LQR-Stanley-Pure Pursuit (PP) 三层融合架构**，以 LQR 曲率自适应控制器为主控层，Stanley 为降级控制层，Pure Pursuit 为低速后备层。三层之间通过基于速度与曲率的 Sigmoid 渐进混合机制实现平滑切换，而非硬切换，确保控制连续性。

整体控制信号流如下：

$$
\delta_{\text{final}} = w_{\text{lqr}} \cdot \delta_{\text{lqr}} + (1 - w_{\text{lqr}}) \cdot \delta_{\text{fallback}}
$$

其中 $w_{\text{lqr}} \in [0, 1]$ 为 LQR 权重，$\delta_{\text{fallback}}$ 根据降级条件在 Stanley 与 PP 之间选择。

### 1.2 三层架构拓扑

```
┌─────────────────────────────────────────────────────┐
│              Layer 1: LQR 主控制器                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ DARE 求解器   │  │ 双增益表插值  │  │ 前馈补偿  │  │
│  │ K(v,κ)       │  │ straight/curve│  │ δ_ff, a_ff│  │
│  └──────────────┘  └──────────────┘  └───────────┘  │
│         ↓                 ↓                ↓         │
│         └─────────→ δ_lqr ←──────────────┘         │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │  降级判定 & 渐进混合     │
              │  w_lqr = σ_v · σ_κ     │
              └────┬──────────────┬─────┘
                   │              │
    ┌──────────────▼──┐    ┌─────▼──────────────┐
    │ Layer 2: Stanley │    │ Layer 3: Pure Pursuit│
    │ 前轴横向误差反馈  │    │ 低速前瞻跟踪         │
    │ δ = e_θ + atan() │    │ δ = atan2(2Lsinα/Lf)│
    └─────────────────┘    └────────────────────┘
```

### 1.3 纵向控制并行通道

横向与纵向控制解耦，纵向通道独立运行：

$$
v_{\text{target}} = \min\bigl(v_{\text{safe}},\; v_{\text{ref}}\bigr)
$$

$$
a_{\text{cmd}} = a_{\text{ff}} + K_p \cdot e_v + K_i \cdot \int e_v \, dt + K_d \cdot \dot{e}_v
$$

---

## 2. 各组件功能分工

### 2.1 Layer 1: LQR 曲率自适应主控制器

#### 2.1.1 状态空间模型

基于自行车运动学模型，离散化状态方程为：

$$
\mathbf{e}_{k+1} = A(v) \, \mathbf{e}_k + B(v) \, \mathbf{u}_k
$$

5 维误差状态向量：

$$
\mathbf{e} = \begin{bmatrix} e_{\text{lat}} \\ \dot{e}_{\text{lat}} \\ e_{\theta} \\ \dot{e}_{\theta} \\ e_v \end{bmatrix}
$$

其中 $e_{\text{lat}}$ 为横向误差，$\dot{e}_{\text{lat}}$ 为横向误差变化率，$e_{\theta}$ 为航向误差，$\dot{e}_{\theta}$ 为航向误差变化率，$e_v = v - v_{\text{ref}}$ 为速度误差。

状态矩阵 $A(v)$ 与控制矩阵 $B(v)$：

$$
A(v) = \begin{bmatrix}
1 & \Delta t & 0 & 0 & 0 \\
0 & 0 & v & 0 & 0 \\
0 & 0 & 1 & \Delta t & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}, \quad
B(v) = \begin{bmatrix}
0 & 0 \\
0 & 0 \\
0 & 0 \\
v/L & 0 \\
0 & \Delta t
\end{bmatrix}
$$

其中 $L = 0.3\,\text{m}$ 为轴距，$\Delta t = 0.005\,\text{s}$ 为控制周期。

#### 2.1.2 DARE 迭代求解

离散代数 Riccati 方程 (DARE) 通过迭代法求解：

$$
P_{k+1} = A^\top P_k A - A^\top P_k B \left(R + B^\top P_k B\right)^{-1} B^\top P_k A + Q
$$

收敛判据：$\max|P_{k+1} - P_k| < \varepsilon$，其中 $\varepsilon = 10^{-9}$，最大迭代次数 10000。

最优反馈增益：

$$
K(v) = \left(R + B^\top P B\right)^{-1} B^\top P A
$$

#### 2.1.3 双增益表预计算与在线插值

离线预计算两组增益表：

- **gains_straight**：$\kappa = 0$（直道条件下的 Q/R 权重）
- **gains_curve**：$\kappa = \kappa_{\text{high}} + 1 = 4.0$（弯道条件下的 Q/R 权重）

速度维度以 $\Delta v = 0.1\,\text{m/s}$ 为步长，覆盖 $[0.1, 2.5]\,\text{m/s}$。

在线插值：首先对速度进行线性插值获取 $K_{\text{straight}}(v)$ 与 $K_{\text{curve}}(v)$，再按曲率混合：

$$
K(v, \kappa) = (1 - \alpha_\kappa) \cdot K_{\text{straight}}(v) + \alpha_\kappa \cdot K_{\text{curve}}(v)
$$

$$
\alpha_\kappa = \text{clip}\left(\frac{|\kappa| - \kappa_{\text{low}}}{\kappa_{\text{high}} - \kappa_{\text{low}}},\; 0,\; 1\right)
$$

其中 $\kappa_{\text{low}} = 0.5$，$\kappa_{\text{high}} = 3.0$。

#### 2.1.4 前馈补偿

转向前馈：

$$
\delta_{\text{ff}} = \arctan(L \cdot \kappa_{\text{la}})
$$

其中 $\kappa_{\text{la}}$ 为前瞻曲率，采用指数衰减加权：

$$
\kappa_{\text{la}} = \frac{\sum w_i \cdot \kappa_i}{\sum w_i}, \quad w_i = \exp\left(-\frac{2 s_i}{s_{\text{la}}}\right), \quad s_{\text{la}} = v \cdot t_{\text{la}} + l_{\min}
$$

加速度前馈：

$$
a_{\text{ff}} = \frac{v_{\text{ref}} - v}{\tau_{\text{ff}}}
$$

#### 2.1.5 LQR 控制律

$$
\mathbf{u} = \mathbf{u}_{\text{ff}} - K(v, \kappa) \cdot \mathbf{e}
$$

输出经物理约束裁剪：$\delta \in [-\delta_{\max},\, \delta_{\max}]$，$a \in [-a_{\max},\, a_{\max}]$。

### 2.2 Layer 2: Stanley 降级控制器

#### 2.2.1 控制律

Stanley 控制器以前轴横向误差为核心反馈量：

$$
\delta = e_\theta + \arctan\left(\frac{k_{\text{eff}} \cdot e_{\text{fa}}}{v}\right)
$$

前轴横向误差计算：

$$
e_{\text{fa}} = \left(\mathbf{p}_{\text{fa}} - \mathbf{p}_{\text{ref}}\right) \cdot \hat{\mathbf{n}}
$$

其中 $\mathbf{p}_{\text{fa}} = \mathbf{p} + L \cdot (\cos\theta, \sin\theta)$ 为前轴位置，$\hat{\mathbf{n}} = (-\sin\theta_{\text{ref}}, \cos\theta_{\text{ref}})$ 为参考轨迹法向量。

#### 2.2.2 低速增益放大

$$
k_{\text{eff}} = k_{\text{stanley}} \cdot \max\left(1,\; \frac{V_{\text{sw}}}{v}\right)
$$

当 $v < V_{\text{sw}} = 0.3\,\text{m/s}$ 时，增益随速度降低而反比放大，补偿低速下 $\arctan$ 项的饱和效应。

### 2.3 Layer 3: Pure Pursuit 低速后备

#### 2.3.1 触发条件

仅当 $v < V_{\text{sw}} = 0.3\,\text{m/s}$ 时激活，解决 Stanley 在极低速下 $\arctan(k \cdot e / v)$ 项截断导致的控制失效。

#### 2.3.2 控制律

前瞻距离：

$$
L_f = k_{\text{pp}} \cdot v + L_{\text{fc}}
$$

转向角：

$$
\delta = \arctan\left(\frac{2L \sin\alpha}{L_f}\right)
$$

其中 $\alpha = \text{atan2}(\Delta y_{\text{la}}, \Delta x_{\text{la}}) - \theta$ 为前瞻点方位角与当前航向之差。

### 2.4 纵向速度控制

#### 2.4.1 曲率限速

$$
v_{\text{limit}} = \min\left(v_{\max},\; \sqrt{\frac{a_{\text{lat,max}}}{|\kappa| + \epsilon}}\right)
$$

#### 2.4.2 前瞻限速

在制动距离 $d_{\text{brake}} = v^2 / (2 a_{\max}) + v \cdot t_{\text{react}}$ 范围内取最小限速：

$$
v_{\text{pre}} = \min\left\{v_{\text{limit}}(s) \mid s \leq d_{\text{brake}}\right\}
$$

#### 2.4.3 动态安全裕度

当横向误差超过阈值时指数衰减目标速度：

$$
v_{\text{safe}} = v_{\text{pre}} \cdot \exp\left(-\beta \cdot \max\left(0,\; |e_{\text{lat}}| - e_{\text{th}}\right)\right)
$$

#### 2.4.4 PID + 前馈

$$
a_{\text{cmd}} = a_{\text{ff}} + K_p \cdot e_v + K_i \cdot \int e_v \, dt + K_d \cdot \dot{e}_v
$$

积分项限幅 $|\int e_v \, dt| \leq 1.0$，防止积分饱和。

### 2.5 控制输出约束

执行器物理约束：

- 转向角限幅：$\delta \in [-\delta_{\max},\, \delta_{\max}]$，$\delta_{\max} = 0.5236\,\text{rad}\;(30°)$
- 转向角速率限幅：$|\dot{\delta}| \leq \dot{\delta}_{\max} = 1.7453\,\text{rad/s}\;(100°/\text{s})$
- 加速度限幅：$a \in [-a_{\max},\, a_{\max}]$，$a_{\max} = 2.0\,\text{m/s}^2$

---

## 3. 自适应机制详解

### 3.1 Q/R 矩阵曲率自适应

#### 3.1.1 设计动机

直道工况下横向误差容忍度较高，应弱化横向惩罚以减少不必要的转向修正；弯道工况下需强化横向与航向惩罚以保证轨迹跟踪精度，同时降低转向控制惩罚以允许更激进的转向输入。

#### 3.1.2 自适应律

曲率插值因子：

$$
\alpha_\kappa = \text{clip}\left(\frac{|\kappa| - \kappa_{\text{low}}}{\kappa_{\text{high}} - \kappa_{\text{low}}},\; 0,\; 1\right)
$$

Q 矩阵自适应：

$$
Q[0,0] = Q_{\text{lat,min}} + \alpha_\kappa \cdot (Q_{\text{lat,max}} - Q_{\text{lat,min}}) = 20 + \alpha_\kappa \cdot 40
$$

$$
Q[2,2] = Q_{\theta,\text{min}} + \alpha_\kappa \cdot (Q_{\theta,\text{max}} - Q_{\theta,\text{min}}) = 5 + \alpha_\kappa \cdot 10
$$

R 矩阵自适应（反向：弯道降低 R 以允许更大转向）：

$$
R[0,0] = R_{\delta,\text{max}} - \alpha_\kappa \cdot (R_{\delta,\text{max}} - R_{\delta,\text{min}}) = 0.15 - \alpha_\kappa \cdot 0.07
$$

| 曲率条件 | $\alpha_\kappa$ | $Q[0,0]$ | $Q[2,2]$ | $R[0,0]$ | 物理含义 |
|----------|:---:|:---:|:---:|:---:|------|
| $|\kappa| \leq 0.5$（直道） | 0 | 20 | 5 | 0.15 | 弱横向惩罚，强转向惩罚 |
| $0.5 < |\kappa| < 3.0$（过渡） | $(0,1)$ | $[20,60]$ | $[5,15]$ | $[0.08,0.15]$ | 渐进过渡 |
| $|\kappa| \geq 3.0$（急弯） | 1 | 60 | 15 | 0.08 | 强横向惩罚，弱转向惩罚 |

#### 3.1.3 实现方式

自适应并非在线求解 DARE，而是离线预计算直道/弯道两组增益表，在线通过线性插值实现，单步计算量仅为两次查表加一次线性混合，满足 $5\,\text{ms}$ 控制周期约束。

### 3.2 前瞻自适应

#### 3.2.1 误差前瞻

曲率越大，当前误差的预测价值越低，需增大前瞻权重以提前响应弯道：

$$
t_{\text{err}} = T_{\text{err,base}} + T_{\text{err,}\kappa} \cdot |\kappa| = 0.2 + 0.25 \cdot |\kappa|
$$

$$
w_{\text{err}} = W_{\text{err,base}} + W_{\text{err,}\kappa} \cdot |\kappa| = 0.3 + 0.6 \cdot |\kappa|
$$

混合误差：

$$
e_{\text{lat,blend}} = (1 - w_{\text{err}}) \cdot e_{\text{lat}} + w_{\text{err}} \cdot e_{\text{lat,la}}
$$

$$
e_{\theta,\text{blend}} = (1 - w_{\text{err}}) \cdot e_{\theta} + w_{\text{err}} \cdot e_{\theta,\text{la}}
$$

#### 3.2.2 曲率前瞻

前馈曲率采用指数衰减加权前瞻：

$$
s_{\text{la}} = v \cdot t_{\text{la,ff}} + l_{\text{la,min}} = 0.6v + 0.3
$$

$$
\kappa_{\text{la}} = \frac{\sum_{s_i \leq 2 s_{\text{la}}} \exp(-2 s_i / s_{\text{la}}) \cdot \kappa_i}{\sum \exp(-2 s_i / s_{\text{la}})}
$$

### 3.3 降级自适应

#### 3.3.1 降级触发条件

以下任一条件满足即触发降级：

| 编号 | 条件 | 降级目标 | 物理含义 |
|:---:|------|:---:|------|
| 1 | $v < V_{\text{sw}} = 0.3\,\text{m/s}$ | PP | 极低速下 Stanley $\arctan$ 项截断 |
| 2 | $|\delta_{\text{cmd}}| > 0.95 \cdot \delta_{\max}$ 连续 3 帧 | Stanley | LQR 输出饱和，反馈增益失效 |
| 3 | $|e_{\text{lat}}| > 0.10\,\text{m}$ | Stanley | 大横向误差超出 LQR 线性化域 |
| 4 | $|\dot{e}_{\text{lat}}| > 0.5\,\text{m/s}$ | Stanley | 误差发散速率过快 |
| 5 | $\text{tr}(P) > P_{\text{th}}$ | Stanley | 定位不确定性过高（预留接口） |

#### 3.3.2 渐进混合

速度 Sigmoid：

$$
w_{\text{lqr},v} = \frac{1}{1 + \exp\left(-K_{\text{sw}} \cdot (v - V_{\text{sw}})\right)} = \frac{1}{1 + \exp(-30 \cdot (v - 0.3))}
$$

曲率 Sigmoid：

$$
w_{\text{lqr},\kappa} = \frac{1}{1 + \exp\left(-5 \cdot (\kappa_{\text{sw}} - |\kappa|)\right)} = \frac{1}{1 + \exp(-5 \cdot (1.5 - |\kappa|))}
$$

联合权重：

$$
w_{\text{lqr}} = w_{\text{lqr},v} \cdot w_{\text{lqr},\kappa}
$$

硬切换保护：当 $v < V_{\text{sw}}$ 时，$w_{\text{lqr}} \leq 0.05$，确保低速下几乎完全由 PP 主导。

最终输出：

$$
\delta_{\text{final}} = w_{\text{lqr}} \cdot \delta_{\text{lqr}} + (1 - w_{\text{lqr}}) \cdot \delta_{\text{fallback}}
$$

#### 3.3.3 Sigmoid 参数选择

$K_{\text{sw}} = 30$ 的选择使速度过渡带宽度约为 $\pm 0.1\,\text{m/s}$，在 $v = 0.2\,\text{m/s}$ 时 $w_{\text{lqr},v} \approx 0.005$，$v = 0.4\,\text{m/s}$ 时 $w_{\text{lqr},v} \approx 0.995$，实现近似硬切换但无抖振。

---

## 4. 参数物理意义与调优范围

### 4.1 车辆物理参数

| 参数 | 符号 | 值 | 物理意义 | 调优约束 |
|------|:---:|:---:|------|------|
| 轴距 | $L$ | 0.3 m | 前后轴距离，决定转向几何 | 由车型固定 |
| 最大转向角 | $\delta_{\max}$ | 0.5236 rad (30°) | 舵机物理极限 | 由舵机规格固定 |
| 最大加速度 | $a_{\max}$ | 2.0 m/s² | 驱动/制动力极限 | 由电机扭矩与附着力决定 |
| 最大转向角速率 | $\dot{\delta}_{\max}$ | 1.7453 rad/s (100°/s) | 舵机响应速度 | 由舵机带宽决定 |
| 控制周期 | $\Delta t$ | 0.005 s | 控制环路频率 200 Hz | 由处理器算力决定 |

### 4.2 LQR 权重参数

| 参数 | 符号 | 值 | 物理意义 | 调优范围 | 调优方向 |
|------|:---:|:---:|------|:---:|------|
| 横向误差权重 | $Q[0,0]$ | 20~60 | 惩罚横向偏差 | [10, 100] | 增大→跟踪更紧，但转向更频繁 |
| 横向误差率权重 | $Q[1,1]$ | 1.0 | 惩罚横向误差变化率 | [0.1, 5] | 增大→阻尼振荡 |
| 航向误差权重 | $Q[2,2]$ | 5~15 | 惩罚航向偏差 | [1, 30] | 增大→航向收敛更快 |
| 航向误差率权重 | $Q[3,3]$ | 0.5 | 惩罚航向误差变化率 | [0.05, 3] | 增大→航向更平稳 |
| 速度误差权重 | $Q[4,4]$ | 3.0 | 惩罚速度偏差 | [0.5, 10] | 增大→速度跟踪更紧 |
| 转向控制权重 | $R[0,0]$ | 0.08~0.15 | 惩罚转向输入幅度 | [0.01, 1.0] | 增大→转向更保守 |
| 加速度控制权重 | $R[1,1]$ | 0.5 | 惩罚加速度输入幅度 | [0.1, 5.0] | 增大→加速更平缓 |

### 4.3 降级与混合参数

| 参数 | 符号 | 值 | 物理意义 | 调优范围 |
|------|:---:|:---:|------|:---:|
| Stanley 横向增益 | $k_{\text{stanley}}$ | 2.0 | 前轴横向误差反馈强度 | [1.0, 5.0] |
| 速度切换阈值 | $V_{\text{sw}}$ | 0.3 m/s | PP/Stanley 切换速度 | [0.1, 0.5] |
| Sigmoid 陡度 | $K_{\text{sw}}$ | 30.0 | 速度过渡带宽度 | [10, 50] |
| 曲率切换阈值 | $\kappa_{\text{sw}}$ | 1.5 | LQR/Stanley 曲率切换点 | [0.5, 3.0] |
| 横向误差降级阈值 | $e_{\text{lat,deg}}$ | 0.10 m | 触发 Stanley 的横向误差 | [0.05, 0.20] |
| 误差率降级阈值 | $\dot{e}_{\text{lat,deg}}$ | 0.5 m/s | 触发 Stanley 的误差率 | [0.2, 1.0] |
| LQR 饱和帧数 | $N_{\text{sat}}$ | 3 | 连续饱和触发降级 | [2, 5] |
| PP 低速增益 | $k_{\text{pp}}$ | 0.1 | PP 前瞻距离速度系数 | [0.05, 0.3] |
| PP 最小前瞻 | $L_{\text{fc}}$ | 0.5 m | PP 前瞻距离常数项 | [0.2, 1.0] |

### 4.4 纵向控制参数

| 参数 | 符号 | 值 | 物理意义 | 调优范围 |
|------|:---:|:---:|------|:---:|
| 最大侧向加速度 | $a_{\text{lat,max}}$ | 1.5 m/s² | 弯道限速依据 | [0.5, 3.0] |
| 前馈时间常数 | $\tau_{\text{ff}}$ | 0.5 s | 速度前馈惯性 | [0.2, 1.0] |
| PID 比例增益 | $K_p$ | 2.0 | 速度误差比例反馈 | [0.5, 5.0] |
| PID 积分增益 | $K_i$ | 0.1 | 稳态误差消除 | [0.01, 0.5] |
| PID 微分增益 | $K_d$ | 0.3 | 速度超调抑制 | [0.05, 1.0] |
| 安全裕度指数 | $\beta$ | 3.0 | 横向误差对限速的衰减率 | [1.0, 5.0] |
| 反应时间 | $t_{\text{react}}$ | 0.2 s | 前瞻制动预留时间 | [0.1, 0.5] |
| 误差滤波系数 | $\alpha_f$ | 0.3 | 误差微分低通滤波 | [0.1, 0.5] |

### 4.5 前瞻参数

| 参数 | 符号 | 值 | 物理意义 | 调优范围 |
|------|:---:|:---:|------|:---:|
| 误差前瞻时间基数 | $T_{\text{err,base}}$ | 0.2 s | 直道误差前瞻时间 | [0.1, 0.4] |
| 误差前瞻时间曲率系数 | $T_{\text{err,}\kappa}$ | 0.25 s | 弯道前瞻时间增量 | [0.1, 0.5] |
| 误差前瞻权重基数 | $W_{\text{err,base}}$ | 0.3 | 直道前瞻误差混合权重 | [0.1, 0.5] |
| 误差前瞻权重曲率系数 | $W_{\text{err,}\kappa}$ | 0.6 | 弯道前瞻权重增量 | [0.2, 1.0] |
| 曲率前瞻时间 | $t_{\text{la,ff}}$ | 0.6 s | 前馈曲率前瞻距离系数 | [0.3, 1.0] |
| 最小前瞻距离 | $l_{\text{la,min}}$ | 0.3 m | 前馈曲率最小前瞻 | [0.1, 0.5] |

---

## 5. SMC 替代 MPC 的设计考量

### 5.1 SMC 控制律

滑模控制器作为四算法对比中的基准之一，其设计如下：

滑模面：

$$
s = \dot{e}_{\text{lat}} + \lambda \cdot e_{\text{lat}}, \quad \lambda = 3.0
$$

等效控制：

$$
\delta_{\text{eq}} = -\frac{L}{v} \left(\lambda \cdot \dot{e}_{\text{lat}} + v \cdot \sin(e_\theta)\right)
$$

切换控制（边界层抑制抖振）：

$$
\delta_{\text{sw}} = -\eta \cdot \text{sat}\left(\frac{s}{\varphi}\right), \quad \eta = 0.8,\; \varphi = 0.05
$$

其中 $\text{sat}(x) = \text{clip}(x, -1, 1)$ 为饱和函数，替代符号函数 $\text{sgn}$ 以抑制抖振。

总控制律：

$$
\delta = \delta_{\text{eq}} + \delta_{\text{sw}} + e_\theta
$$

### 5.2 SMC 替代 MPC 的理由

| 维度 | MPC | SMC | 选择理由 |
|------|-----|-----|------|
| 计算复杂度 | 在线 QP 求解，$O(n^3)$ | 代数公式，$O(1)$ | 嵌入式平台 200 Hz 控制周期下 MPC 求解时间不可控 |
| 实时性保证 | 依赖求解器收敛 | 确定性计算时间 | 比赛场景不允许求解超时 |
| 鲁棒性 | 依赖模型精度 | 对模型不确定性天然鲁棒 | 自驾小车参数辨识精度有限 |
| 参数调优 | 预测时域、控制时域、权重矩阵 | $\lambda, \eta, \varphi$ 三个参数 | SMC 调参维度低，现场调试效率高 |
| 抖振问题 | 无 | 边界层 $\varphi=0.05$ 抑制 | $\varphi$ 在精度与抖振间取折中 |

### 5.3 边界层宽度 $\varphi$ 的选择

$\varphi = 0.05$ 的物理含义：当 $|s| < 0.05$ 时，切换控制从 bang-bang 退化为线性比例反馈，避免在滑模面附近的高频切换。该值需满足：

- $\varphi$ 过小（$< 0.01$）：抖振明显，转向执行器磨损加剧
- $\varphi$ 过大（$> 0.2$）：稳态精度下降，等效于降低切换增益

### 5.4 SMC 在对比实验中的定位

SMC 作为非优化类鲁棒控制器的代表参与四算法对比（LQR-Stanley、PP、Stanley-only、SMC），用于验证复合架构相对于单一鲁棒控制器的优势。SMC 的优势在于强鲁棒性与零调参成本，劣势在于稳态精度有限且无法像 LQR 那样通过权重矩阵精细调节各误差分量的收敛特性。

---

## 附录 A: 文件结构

| 文件 | 功能 |
|------|------|
| `config.py` | 全局参数集中管理 |
| `lqr_controller.py` | DARE 求解、增益预计算、自适应 Q/R、误差计算、LQR 控制 |
| `stanley_controller.py` | Stanley 控制、降级判定、渐进混合 |
| `speed_controller.py` | 曲率限速、前瞻限速、安全裕度、PID 速度控制 |
| `control_output.py` | 控制量裁剪与发布 |
| `reference_extractor.py` | 参考轨迹提取（曲率、弧长、参考速度） |
| `compare_controllers.py` | 四算法对比仿真与可视化 |

## 附录 B: 误差微分滤波

误差变化率通过一阶低通滤波消除数值微分噪声：

$$
\dot{e}_k = \alpha_f \cdot \frac{e_k - e_{k-1}}{\Delta t} + (1 - \alpha_f) \cdot \dot{e}_{k-1}, \quad \alpha_f = 0.3
$$

$\alpha_f$ 越小滤波越强但延迟越大，$\alpha_f = 0.3$ 在 $200\,\text{Hz}$ 采样率下截止频率约为 $10\,\text{Hz}$。

## 附录 C: 参考轨迹曲率计算

采用 Menger 曲率的三点公式：

$$
\kappa = \frac{2 \cdot |\text{Area}(\mathbf{p}_{i-1}, \mathbf{p}_i, \mathbf{p}_{i+1})|}{|\mathbf{p}_i - \mathbf{p}_{i-1}| \cdot |\mathbf{p}_{i+1} - \mathbf{p}_i| \cdot |\mathbf{p}_{i+1} - \mathbf{p}_{i-1}|}
$$

边界点采用外推填充，内部点取相邻 Menger 曲率的均值。
