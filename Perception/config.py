"""
Global configuration constants for the perception module

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np

# === Phase 0: Global configuration ===

V_MAX = 2.5  # [m/s]
DELTA_MAX = np.deg2rad(30.0)  # [rad]
A_MAX = 2.0  # [m/s^2]
WHEELBASE = 0.3  # [m]

R_VOXEL = 0.05  # [m]

TAU_SDF = 0.15  # [m]
BETA_SDF = 3.0
D_NEAR = 0.5  # [m]

EPSILON_CLUSTER = 0.2  # [m]
N_MIN = 3
R_MARGIN = 0.1  # [m]
A_MAX_CLUSTER = 2.0  # [m^2]

SIGMA_POS = 0.05  # [m]
SIGMA_VEL = 0.5  # [m/s]
SIGMA_OBS = 0.15  # [m]
D_ASSOC = 1.0  # [m]
N_CONFIRM = 3
N_DELETE = 5
ALPHA_NEW = 0.01
ALPHA_STABLE = 0.05

R_INFLATE = 0.3  # [m]
SIGMA_DYN = 0.2  # [m]
DT_PRED = 0.5  # [s]
W_PRED = 0.5
RESOLUTION = 0.05  # [m/cell]

W_DET = 0.4
W_TRK = 0.6

CHI2_NEW = 9.21
CHI2_STABLE = 5.99

V_DYNAMIC_THRESH = 0.1  # [m/s]
N_DYNAMIC_EXTRA = 3

DT = 0.1  # [s]
