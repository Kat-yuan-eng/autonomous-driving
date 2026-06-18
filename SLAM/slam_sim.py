"""SLAM simulation data generation: trajectories, landmarks, sensor data

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from SLAM.config import (WHEELBASE, V_MAX, DELTA_MAX_RAD, LIDAR_RANGE_MAX,
    LIDAR_N_BEAMS, LIDAR_SIGMA_RANGE, ACCEL_SIGMA, GYRO_SIGMA, GRAVITY,
    WHEEL_SIGMA_V, WHEEL_SIGMA_W, IMU_DT, VOXEL_SIZE, UKF_DT,
    DELTA_DOT_MAX_RAD_S, WHEEL_FREQ)

# === Phase 1: Trajectory generation ===

def generate_reference_trajectory(course_type='figure8', dt=UKF_DT):
    """
    Generate reference trajectory for the specified course type.

    :param course_type: (str) Course type: 'figure8', 'circle', 'straight', or 'mixed'
    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Trajectory array, shape (N, 3) with columns [x, y, theta]
    """
    assert course_type in ('figure8', 'circle', 'straight', 'mixed'), f"unknown course_type={course_type}"
    ref_xy = {'figure8': _figure8, 'circle': _circle, 'straight': _straight, 'mixed': _mixed}
    return ref_xy[course_type](dt)


def _figure8(dt):
    """
    Generate smooth figure-8 trajectory with continuous heading.

    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Trajectory, shape (N, 3)
    """
    t = np.arange(0, 30, dt)
    a = 5.0
    x = a * np.sin(0.2 * t)
    y = a * np.sin(0.4 * t) * np.cos(0.2 * t)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    speed = np.sqrt(dx**2 + dy**2)
    theta = np.zeros_like(t)
    theta[0] = np.arctan2(dy[0], dx[0])
    for i in range(1, len(t)):
        if speed[i] < 0.5:
            theta[i] = theta[i-1]
        else:
            theta_raw = np.arctan2(dy[i], dx[i])
            diff = theta_raw - theta[i-1]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            theta[i] = theta[i-1] + diff
    return np.column_stack([x, y, theta])


def _circle(dt):
    """
    Generate circular trajectory.

    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Trajectory, shape (N, 3)
    """
    t = np.arange(0, 30, dt)
    r = 5.0
    omega = 0.2
    x = r * np.cos(omega * t)
    y = r * np.sin(omega * t)
    theta = omega * t + np.pi / 2
    return np.column_stack([x, y, angle_mod(theta)])


def _straight(dt):
    """
    Generate straight-line trajectory.

    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Trajectory, shape (N, 3)
    """
    t = np.arange(0, 15, dt)
    x = 1.0 * t
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    return np.column_stack([x, y, theta])


def _mixed(dt):
    """
    Generate mixed trajectory: straight + arc + straight segments.

    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Trajectory, shape (N, 3)
    """
    t = np.arange(0, 40, dt)
    seg_n = len(t) // 3
    x, y, theta = np.zeros(3), np.zeros(3), np.zeros(3)
    x1 = 1.0 * t[:seg_n]
    y1 = np.zeros(seg_n)
    t1 = np.zeros(seg_n)
    arc_len = 0.5 * np.pi * 3.0
    t_arc = t[seg_n:2*seg_n] - t[seg_n]
    x2 = x1[-1] + 3.0 * np.sin(t_arc / 5.0)
    y2 = t_arc * 0.5
    t2 = np.arctan2(np.gradient(y2, dt), np.gradient(x2, dt))
    t_end = t[2*seg_n:] - t[2*seg_n]
    x3 = x2[-1] + 1.0 * t_end
    y3 = y2[-1] * np.ones_like(t_end)
    t3 = np.zeros_like(t_end)
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    theta = np.concatenate([t1, t2, t3])
    return np.column_stack([x, y, theta])


# === Phase 2: Sensor observation generation ===

def generate_lidar_scan(x, y, theta, landmarks, range_max=LIDAR_RANGE_MAX,
                        n_beams=LIDAR_N_BEAMS, fov_deg=360.0, noise_std=LIDAR_SIGMA_RANGE):
    """向量化 LiDAR 扫描生成

    :param x: (float) 机器人 x 位置 [m]
    :param y: (float) 机器人 y 位置 [m]
    :param theta: (float) 机器人航向 [rad]
    :param landmarks: (np.ndarray) 路标位置, shape (M, 2)
    :param range_max: (float) 最大检测距离 [m]
    :param n_beams: (int) LiDAR 波束数
    :param fov_deg: (float) 视场角 [°]
    :param noise_std: (float) 距离噪声标准差 [m]
    :return: (np.ndarray) 距离测量, shape (n_beams,)
    """
    assert len(landmarks) > 0, "landmarks must be non-empty"
    dx = landmarks[:, 0] - x
    dy = landmarks[:, 1] - y
    dists = np.sqrt(dx**2 + dy**2)
    bearings = np.arctan2(dy, dx)

    angles = np.linspace(theta - np.deg2rad(fov_deg / 2),
                         theta + np.deg2rad(fov_deg / 2), n_beams, endpoint=False)

    ang_diff = angle_mod(bearings[np.newaxis, :] - angles[:, np.newaxis])
    beam_fov = np.deg2rad(1.5)
    mask_close = np.abs(ang_diff) < beam_fov

    scan_ranges = np.full(n_beams, range_max)
    # 向量化：用 np.where 将非命中距离设为 inf，再取 min
    masked_dists = np.where(mask_close, dists[np.newaxis, :], np.inf)
    hits = masked_dists.min(axis=1)
    valid = hits < range_max
    scan_ranges[valid] = hits[valid] + np.random.randn(valid.sum()) * noise_std

    scan_ranges = np.clip(scan_ranges, 0.0, range_max)
    return scan_ranges


def generate_imu(x, v, q, ba, bg, dt=IMU_DT):
    """
    Simulate IMU measurement with bias and Gaussian noise.

    :param x: (float) Position (unused in measurement model)
    :param v: (float) Velocity (unused in measurement model)
    :param q: (np.ndarray) Orientation quaternion, shape (4,)
    :param ba: (np.ndarray) Accelerometer bias, shape (3,)
    :param bg: (np.ndarray) Gyroscope bias, shape (3,)
    :param dt: (float) IMU sampling period in seconds
    :return: (tuple) (a_m, w_m) measured acceleration and angular velocity, each shape (3,)
    """
    accel_true = np.zeros(3)
    accel_true[2] = GRAVITY
    gyro_true = np.zeros(3)
    a_m = accel_true + ba + np.random.randn(3) * ACCEL_SIGMA
    w_m = gyro_true + bg + np.random.randn(3) * GYRO_SIGMA
    return a_m, w_m


def generate_imu_batch(n_steps, n_sub, dt_imu=IMU_DT, ref_traj=None):
    """批量生成 IMU 数据，从参考轨迹计算真实加速度和角速度

    :param n_steps: (int) 时间步数
    :param n_sub: (int) 每步 IMU 子步数
    :param dt_imu: (float) IMU 时间步长
    :param ref_traj: (np.ndarray) 参考轨迹 shape (n_steps, 3)，用于计算真实运动
    :return: (np.ndarray) IMU 数据，shape (n_steps, n_sub, 6)
    """
    total = n_steps * n_sub
    imu_data = np.zeros((n_steps, n_sub, 6))
    if ref_traj is not None and len(ref_traj) == n_steps:
        dt_traj = UKF_DT
        x_traj = ref_traj[:, 0]
        y_traj = ref_traj[:, 1]
        theta_traj = ref_traj[:, 2]
        vx_world = np.gradient(x_traj, dt_traj)
        vy_world = np.gradient(y_traj, dt_traj)
        ax_world = np.gradient(vx_world, dt_traj)
        ay_world = np.gradient(vy_world, dt_traj)
        omega_z = np.gradient(np.unwrap(theta_traj), dt_traj)
        for i in range(n_steps):
            theta_i = theta_traj[i]
            c, s = np.cos(theta_i), np.sin(theta_i)
            ax_body = c * ax_world[i] + s * ay_world[i]
            ay_body = -s * ax_world[i] + c * ay_world[i]
            for k in range(n_sub):
                a_m = np.array([ax_body, ay_body, GRAVITY])
                w_m = np.array([0.0, 0.0, omega_z[i]])
                a_m += np.random.randn(3) * ACCEL_SIGMA
                w_m += np.random.randn(3) * GYRO_SIGMA
                imu_data[i, k, 0:3] = a_m
                imu_data[i, k, 3:6] = w_m
    else:
        a_noise = np.random.randn(total, 3) * 0.01
        w_noise = np.random.randn(total, 3) * np.deg2rad(0.1)
        imu_data[:, :, 0:3] = a_noise.reshape(n_steps, n_sub, 3)
        imu_data[:, :, 3:6] = w_noise.reshape(n_steps, n_sub, 3)
        imu_data[:, :, 2] += GRAVITY
    return imu_data


def generate_wheel_odom(v, omega, dt=1.0/WHEEL_FREQ):
    """
    Simulate wheel odometry measurement with Gaussian noise.

    :param v: (float) Forward velocity in m/s
    :param omega: (float) Angular velocity in rad/s
    :param dt: (float) Sampling period in seconds
    :return: (np.ndarray) Measured [v_m, w_m], shape (2,)
    """
    v_m = v + np.random.randn() * WHEEL_SIGMA_V
    w_m = omega + np.random.randn() * WHEEL_SIGMA_W
    return np.array([v_m, w_m])


# === Phase 3: Landmark generation ===

def generate_landmarks(n_lm=50, map_size=10.0):
    """
    Generate random 2D landmark positions uniformly in a square area.

    :param n_lm: (int) Number of landmarks
    :param map_size: (float) Half-side of square area in meters
    :return: (np.ndarray) Landmark positions, shape (n_lm, 2)
    """
    rng = np.random.RandomState(42)
    lx = rng.uniform(-map_size, map_size, n_lm)
    ly = rng.uniform(-map_size, map_size, n_lm)
    return np.column_stack([lx, ly])


# === Phase 4: Utility functions ===

def angle_mod(x):
    """
    Wrap angle to [-pi, pi).

    :param x: (float or np.ndarray) Angle in radians
    :return: (float or np.ndarray) Wrapped angle
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def bicycle_dynamics(state, v, delta, dt, wheelbase=WHEELBASE):
    """
    Propagate bicycle model state by one time step.

    :param state: (np.ndarray) Current state [x, y, theta, ...], at least 3 elements
    :param v: (float) Forward velocity in m/s
    :param delta: (float) Steering angle in radians
    :param dt: (float) Time step in seconds
    :param wheelbase: (float) Wheelbase in meters
    :return: (np.ndarray) Updated state with same shape as input
    """
    x, y, theta = state[:3]
    x_n = x + v * np.cos(theta) * dt
    y_n = y + v * np.sin(theta) * dt
    theta_n = theta + v * np.tan(delta) / wheelbase * dt
    out = state.copy()
    out[0] = x_n
    out[1] = y_n
    out[2] = angle_mod(theta_n)
    return out


def generate_noisy_trajectory(ref_traj, v_profile, delta_profile, dt):
    """
    Propagate bicycle dynamics from initial pose using velocity and steering profiles.

    :param ref_traj: (np.ndarray) Reference trajectory for initial pose, shape (N, 3)
    :param v_profile: (np.ndarray) Forward velocity at each step, shape (N,)
    :param delta_profile: (np.ndarray) Steering angle at each step, shape (N,)
    :param dt: (float) Time step in seconds
    :return: (np.ndarray) Noisy trajectory, shape (N, 3)
    """
    n = len(ref_traj)
    states = np.zeros((n, 3))
    states[0] = ref_traj[0, :3]
    for i in range(1, n):
        v = v_profile[min(i, len(v_profile) - 1)]
        delta = delta_profile[min(i, len(delta_profile) - 1)]
        states[i] = bicycle_dynamics(states[i-1], v, delta, dt)
    return states