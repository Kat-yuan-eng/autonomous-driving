# === Phase 1: IMU time interpolation ===
# === Phase 2: Multi-sensor synchronization ===
# === Phase 3: Sync error statistics ===
# === Phase 4: Spatial extrinsic calibration ===
# === Phase 5: Lever arm compensation ===
import numpy as np

from SLAM.config import (SYNC_MAX_SKEW_MS, SYNC_INTERP_EPS,
    EXTRINSIC_LIDAR_IMU_R, EXTRINSIC_LIDAR_IMU_T, LEVER_ARM_WHEEL)


def interpolate_imu(imu_data, t_target):
    assert imu_data.shape[1] == 7, f"imu_data must have 7 columns, got {imu_data.shape[1]}"
    assert imu_data[0, 0] <= t_target <= imu_data[-1, 0], \
        f"t_target {t_target} out of range [{imu_data[0, 0]}, {imu_data[-1, 0]}]"
    idx = np.searchsorted(imu_data[:, 0], t_target)
    idx = max(1, min(idx, imu_data.shape[0] - 1))
    t_prev, t_next = imu_data[idx - 1, 0], imu_data[idx, 0]
    assert t_next - t_prev > SYNC_INTERP_EPS, \
        f"interp interval too small: {t_next - t_prev}"
    alpha = (t_target - t_prev) / (t_next - t_prev)
    imu_interp = imu_data[idx - 1] + alpha * (imu_data[idx] - imu_data[idx - 1])
    imu_interp[0] = t_target
    imu_interp[1:] = np.round(imu_interp[1:], 6)
    return imu_interp


def _nearest_neighbor(data, t_target):
    assert data.shape[0] >= 2, f"data must have at least 2 rows, got {data.shape[0]}"
    idx = np.searchsorted(data[:, 0], t_target)
    idx = np.clip(idx, 1, len(data) - 1)
    dist_prev = np.abs(data[idx - 1, 0] - t_target)
    dist_next = np.abs(data[idx, 0] - t_target)
    mask = dist_next < dist_prev
    result = np.where(mask[:, None], data[idx], data[idx - 1])
    return result


def sync_sensors(imu_raw, lidar_raw, wheel_raw, t_lidar):
    assert imu_raw.shape[1] == 7, f"imu_raw must have 7 columns, got {imu_raw.shape[1]}"
    assert lidar_raw.shape[1] == 4, f"lidar_raw must have 4 columns, got {lidar_raw.shape[1]}"
    assert wheel_raw.shape[1] == 4, f"wheel_raw must have 4 columns, got {wheel_raw.shape[1]}"
    L = len(t_lidar)
    assert L > 0, "t_lidar must not be empty"
    imu_sync = np.array([interpolate_imu(imu_raw, t) for t in t_lidar])
    lidar_sync = _nearest_neighbor(lidar_raw, t_lidar)
    wheel_sync = _nearest_neighbor(wheel_raw, t_lidar)
    sync_errors_ms = np.array([abs(t_lidar[i] - lidar_sync[i, 0]) * 1000 for i in range(L)])
    assert np.all(sync_errors_ms <= SYNC_MAX_SKEW_MS), \
        f"sync error exceeded: max={sync_errors_ms.max():.6f}ms"
    return {
        'imu_sync': imu_sync,
        'lidar_sync': lidar_sync,
        'wheel_sync': wheel_sync,
        'sync_errors_ms': sync_errors_ms,
    }


def compute_sync_error(t_imu, t_lidar, t_wheel):
    n_min = min(len(t_imu), len(t_lidar), len(t_wheel))
    assert n_min > 0, "timestamp arrays must not be empty"
    t_imu_a = t_imu[:n_min]
    t_lidar_a = t_lidar[:n_min]
    t_wheel_a = t_wheel[:n_min]
    skew_imu_lidar = np.abs(t_imu_a - t_lidar_a) * 1000
    skew_wheel_lidar = np.abs(t_wheel_a - t_lidar_a) * 1000
    skew = np.concatenate([skew_imu_lidar, skew_wheel_lidar])
    return {
        'max_skew_ms': round(float(skew.max()), 6),
        'mean_skew_ms': round(float(skew.mean()), 6),
        'std_skew_ms': round(float(skew.std()), 6),
    }


def apply_extrinsic(pose_imu, R_ext, t_ext):
    assert pose_imu.shape == (4, 4), f"pose_imu shape mismatch: {pose_imu.shape}"
    assert R_ext.shape == (3, 3), f"R_ext shape mismatch: {R_ext.shape}"
    assert t_ext.shape == (3,), f"t_ext shape mismatch: {t_ext.shape}"
    T_ext = np.eye(4)
    T_ext[:3, :3] = R_ext
    T_ext[:3, 3] = t_ext
    pose_target = pose_imu @ T_ext
    return pose_target


def apply_lever_arm(pose_imu, lever_arm):
    assert pose_imu.shape == (4, 4), f"pose_imu shape mismatch: {pose_imu.shape}"
    assert lever_arm.shape == (3,), f"lever_arm shape mismatch: {lever_arm.shape}"
    R_imu = pose_imu[:3, :3]
    v_imu = pose_imu[:3, 3]
    omega_imu = np.zeros(3)
    v_wheel = v_imu + np.cross(omega_imu, lever_arm)
    return v_wheel
