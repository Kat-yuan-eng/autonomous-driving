"""
Global configuration parameters for the control module

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np

# === Phase 0: Vehicle Physical Constants ===

WHEELBASE = 0.3  # [m]
V_MAX = 2.5  # [m/s]
DELTA_MAX_RAD = 0.5236  # [rad]
A_MAX = 2.0  # [m/s^2]
DELTA_DOT_MAX_RAD_S = 1.7453  # [rad/s]
DT = 0.005  # [s]
DV_TABLE = 0.1  # [m/s]
V_TABLE_MIN = 0.1  # [m/s]
V_TABLE_MAX = 2.5  # [m/s]

# === Phase 1: Stanley Degradation Parameters ===

K_STANLEY = 2.0  # [dimensionless]
V_MIN_STANLEY = 0.1  # [m/s]
K_SW = 30.0  # [dimensionless]
V_SW = 0.3  # [m/s]
N_SAT_TRIGGER = 3  # [count]
E_LAT_DEGRADE = 0.10  # [m]
KAPPA_SW = 1.214854  # [1/m]
DE_LAT_DEGRADE = 0.5  # [m/s]
K_KAPPA_SW = 5.0  # [dimensionless]
V_REF_STANLEY_GAIN = 0.3  # [m/s]
K_PP_LOW = 0.1  # [dimensionless]
LFC_LOW = 0.424430  # [m]

# === Phase 2: Longitudinal Speed Control Parameters ===

A_LAT_MAX = 1.5  # [m/s^2]
TAU_FF = 0.5  # [s]
KP_V = 2.0  # [dimensionless]
KI_V = 0.1  # [dimensionless]
KD_V = 0.3  # [dimensionless]
E_LAT_TH = 0.15  # [m]
BETA_SAFE = 2.0  # [dimensionless]
T_REACT = 0.180000  # [s]
INTEGRAL_LIMIT = 1.0  # [m]
ALPHA_F = 0.3  # [dimensionless]
T_LA_FF = 0.953814  # [s]
L_LA_MIN = 0.536489  # [m]

# === Phase 3: Curvature-Adaptive Lookahead Parameters ===

T_ERR_BASE = 0.287500  # [s]
T_ERR_KAPPA = 0.215141  # [s*m]
W_ERR_BASE = 0.3  # [dimensionless]
W_ERR_KAPPA = 0.228004  # [m]
LOOKAHEAD_RANGE_FACTOR = 2.0  # [dimensionless]
LOOKAHEAD_DECAY_RATE = 2.0  # [dimensionless]

# === Phase 4: LQR Weight Matrices ===

Q_LQR = np.diag([20.0, 1.0, 5.0, 0.5, 3.0])
R_LQR = np.diag([0.15, 0.5])

# === Phase 5: Curvature-Adaptive Q Weight Parameters ===

KAPPA_LOW = 0.5  # [1/m]
KAPPA_HIGH = 3.0  # [1/m]
Q_LAT_MIN = 17.816719  # [dimensionless]
Q_LAT_MAX = 60.0  # [dimensionless]
Q_THETA_MIN = 1.535111  # [dimensionless]
Q_THETA_MAX = 15.0  # [dimensionless]
R_DELTA_MIN = 0.08  # [dimensionless]
R_DELTA_MAX = 0.15  # [dimensionless]

# === Phase 6: SMC Parameters ===

LAM_SMC = 3.0  # [dimensionless]
ETA_SMC = 0.8  # [dimensionless]
PHI_SMC = 0.05  # [dimensionless]

# === Phase 7: Compatibility Aliases ===

DELTA_MAX = DELTA_MAX_RAD
DELTA_DOT_MAX = DELTA_DOT_MAX_RAD_S

# === Phase 8: Composite Scoring Weights ===

METRIC_WEIGHTS = np.array([0.35, 0.25, 0.15, 0.05, 0.10, 0.10])

# === Phase 9: Visualization Colors ===

COLORS = {
    'algo_1': '#1f77b4',
    'algo_2': '#ff7f0e',
    'algo_3': '#2ca02c',
    'algo_4': '#d62728',
    'algo_5': '#9467bd',
    'algo_6': '#8c564b',
    'grid': '#cccccc',
    'bg': '#ffffff',
}
COLOR_LQR = COLORS['algo_1']
COLOR_PP = COLORS['algo_2']
COLOR_STANLEY = COLORS['algo_6']
COLOR_SMC = COLORS['algo_3']
