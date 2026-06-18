"""SLAM module configuration parameters

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Sensor parameters ===
import numpy as np

# LiDAR
LIDAR_RANGE_MAX = 12.0  # [m]
LIDAR_N_BEAMS = 360
LIDAR_ANGLE_MIN = -np.pi  # [rad]
LIDAR_ANGLE_MAX = np.pi  # [rad]
LIDAR_SIGMA_RANGE = 0.02  # [m]
LIDAR_SIGMA_BEARING = np.deg2rad(0.5)  # [rad]
VOXEL_SIZE = 0.05  # [m]

# IMU
IMU_FREQ = 200  # [Hz]
IMU_DT = 1.0 / IMU_FREQ  # [s]
ACCEL_SIGMA = 0.1  # [m/s^2]
GYRO_SIGMA = np.deg2rad(0.5)  # [rad/s]
ACCEL_BIAS_WALK = 1e-4  # [m/s^3]
GYRO_BIAS_WALK = 1e-5  # [rad/s^2]
GRAVITY = 9.81  # [m/s^2]

# Wheel odometry
WHEEL_FREQ = 50  # [Hz]
WHEEL_SIGMA_V = 0.05  # [m/s]
WHEEL_SIGMA_W = np.deg2rad(1.0)  # [rad/s]

# === Phase 2: Vehicle model parameters ===
WHEELBASE = 0.3  # [m]
V_MAX = 2.5  # [m/s]
DELTA_MAX_RAD = np.deg2rad(30.0)  # [rad]
A_MAX = 2.0  # [m/s^2]
DELTA_DOT_MAX_RAD_S = np.deg2rad(100.0)  # [rad/s]

# === Phase 3: Mapping parameters ===
N_SUBMAP_SCANS = 90
OCC_PROB_OCCUPIED = 0.85
OCC_PROB_FREE = 0.15
PROB_OCCUPIED_INIT = 0.5
SEARCH_WIN_LIN = 0.2  # [m]
SEARCH_WIN_ANG = np.deg2rad(10.0)  # [rad]
VOXEL_FILTER_SIZE = 0.05  # [m]

# === Phase 4: Branch-and-bound loop closure detection parameters ===
BB_LAYERS = 4
BB_RES_COARSE = 0.30  # [m]
BB_RES_FINE = 0.0375  # [m]
LOOP_CLOSURE_SCORE_MIN = 0.65
CAUCHY_C_ODOM = 0.1
CAUCHY_C_LOOP = 0.3

# === Phase 6: UKF parameters ===
UKF_ALPHA = 0.1
UKF_BETA = 2.0
UKF_KAPPA = 0.0
UKF_DIM = 6
UKF_DT = 0.02  # [s]
UKF_Q = np.diag([0.0015, 0.0015, np.deg2rad(0.015), 0.008, 0.008, np.deg2rad(0.04)])**2  # [covariance]
UKF_R_CARTO_BASE = np.diag([0.005, 0.005, np.deg2rad(0.035)])**2  # [covariance]
UKF_BETA_A = 0.03  # [m/s^2]
UKF_BETA_W = 0.01  # [rad/s]
UKF_Q_SCALE_HIGH = 1.500000
UKF_Q_SCALE_LOW = 0.200000
UKF_INNOVATION_THRESH = 2.000000
UKF_Q_SCALE_NORMAL = 1.000000
UKF_R_CARTO_TIME_DECAY = 0.300000  # [1/s]
UKF_R_CARTO_TIME_MAX = 1.000000  # [s]
UKF_ALPHA_LOW = 0.080000
UKF_ALPHA_HIGH = 0.250000
UKF_OMEGA_HIGH_THRESH = 0.500000  # [rad/s]

# === Phase 7: Degradation strategy parameters ===
SCORE_HEALTHY = 0.8
SCORE_DEGRADE = 0.4
TIME_SINCE_MATCH_MAX = 1.0  # [s]
RELOC_SCORE_TH = 0.6
RELOC_TIMEOUT = 2.0  # [s]
RELPOS_EPS = 1e-9  # [m]
CARTO_K_COV = 0.500000
CARTO_K_INNOV = 2.000000
CARTO_COV_TRACE_MAX = 10.000000
VOXEL_DENSITY_LOW = 500.000000  # [pts/m^2]
VOXEL_DENSITY_HIGH = 2000.000000  # [pts/m^2]
VOXEL_SIZE_MIN = 0.020000  # [m]
VOXEL_SIZE_MAX = 0.100000  # [m]
SCORE_HISTORY_LEN = 20
SCORE_HEALTHY_LOW = 0.500000
SCORE_HEALTHY_HIGH = 0.800000
SCORE_DEGRADE_LOW = 0.200000
SCORE_DEGRADE_HIGH = 0.400000

# === Phase 8: EKF-SLAM comparison algorithm parameters ===
EKF_SLAM_Q_V = 0.1  # [m/s]
EKF_SLAM_Q_THETA = np.deg2rad(1.0)  # [rad/s]
EKF_SLAM_R_RANGE = 0.02  # [m]
EKF_SLAM_R_BEARING = np.deg2rad(0.5)  # [rad]
EKF_SLAM_MAX_RANGE = 10.0  # [m]

# === Phase 9: FastSLAM comparison algorithm parameters ===
FASTSLAM_N_PARTICLES = 100
FASTSLAM_Q_V = 0.1  # [m/s]
FASTSLAM_Q_THETA = np.deg2rad(1.0)  # [rad/s]
FASTSLAM_R_RANGE = 0.02  # [m]
FASTSLAM_R_BEARING = np.deg2rad(0.5)  # [rad]
FASTSLAM_MAX_RANGE = 10.0  # [m]
FASTSLAM_N_EFF_THRESHOLD = 0.5
FASTSLAM_LM_P_INIT = 1e4

# === Phase 10: GraphSLAM baseline algorithm parameters ===
GRAPHSLAM_LOOP_DIST_THRESHOLD = 1.5  # [m]
GRAPHSLAM_LOOP_MIN_INDEX_GAP = 30
GRAPHSLAM_ODOM_INFO_XY = 1.0 / 0.05**2  # [1/m^2]
GRAPHSLAM_ODOM_INFO_THETA = 1.0 / np.deg2rad(2.0)**2  # [1/rad^2]
GRAPHSLAM_LOOP_INFO_XY = 1.0 / 0.1**2  # [1/m^2]
GRAPHSLAM_LOOP_INFO_THETA = 1.0 / np.deg2rad(5.0)**2  # [1/rad^2]
GRAPHSLAM_N_OPTIM_ITER = 10

# === Phase 11: Test scenario parameters ===
TEST_NOISE_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
TEST_KIDNAP_STEP = 200
TEST_KIDNAP_JUMP = 3.0  # [m]
TEST_N_MONTE_CARLO = 20
TEST_REALTIME_BUDGET_MS = 6.0  # [ms]
TEST_CAUCHY_VS_L2_N_OUTLIERS = [0, 1, 3, 5, 10]
TEST_LOOP_CLOSURE_N_TRIALS = 50

# === Phase 12: Visualization parameters ===
VIS_SHOW_ANIMATION = True
VIS_FIGSIZE = (12, 8)  # [inch]
VIS_DPI = 100
VIS_TRAIL_ALPHA = 0.7
VIS_PARTICLE_ALPHA = 0.3
VIS_COV_SCALE = 3.0
VIS_ANIM_INTERVAL = 50  # [ms]
VIS_COLORS = {
    'gt': "#1f76b4da",
    'ekf': "#ff7e0ebe",
    'fastslam': "#2ca02c9d",
    'graphslam': "#9367bdbe",
    'cartographer_ukf': "#d627279d",
    'loop_edge': '#2ca02c',
    'odom_edge': "#cccccce1",
    'EKF_SLAM': "#ff7e0ec3",
    'FastSLAM': "#2ca02cd6",
    'GraphSLAM': "#9367bdbb",
    'Cartographer-UKF': "#d62727b9",
}

# === Phase 13: Sensor synchronization parameters ===
SYNC_IMU_RATE_HZ = 200.0  # [Hz]
SYNC_LIDAR_RATE_HZ = 50.0  # [Hz]
SYNC_WHEEL_RATE_HZ = 50.0  # [Hz]
SYNC_MAX_SKEW_MS = 1.000000  # [ms]
SYNC_INTERP_EPS = 1e-9  # [s]

# === Phase 14: Spatial calibration extrinsics ===
EXTRINSIC_LIDAR_IMU_R = np.eye(3)
EXTRINSIC_LIDAR_IMU_T = np.zeros(3)  # [m]
LEVER_ARM_WHEEL = np.zeros(3)  # [m]

# === Phase 15: Latency compensation parameters ===
COMPENSATE_MAX_DT = 0.100000  # [s]

# === Phase 5: carto_match performance optimization parameters ===
CARTO_WIN_COV_GAIN = 0.300000

# === Phase 5: Loop closure detection parameters ===
LOOP_DIST_THRESH = 1.0  # [m]
LOOP_ICP_THRESH = 0.1  # [m]
LOOP_CHECK_INTERVAL = 20