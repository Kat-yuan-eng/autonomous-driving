import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from SLAM.slam_sim import angle_mod


# === Phase 1: Relative pose error RPE (TUM protocol) ===

def compute_rpe(traj_est, traj_gt, delta_m=1.0):
    assert traj_est.shape == traj_gt.shape, f"shape mismatch est={traj_est.shape} gt={traj_gt.shape}"
    assert traj_est.ndim == 2 and traj_est.shape[1] == 3, f"expected (N,3), got {traj_est.shape}"
    n = traj_est.shape[0]
    assert n >= 2, f"trajectory must have >=2 points, got n={n}"
    assert delta_m > 0, f"delta_m must be positive, got {delta_m}"

    diff_gt = np.diff(traj_gt[:, :2], axis=0)
    seg_len = np.sqrt(np.sum(diff_gt ** 2, axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    target = cum_len + delta_m
    j_arr = np.searchsorted(cum_len, target, side='left')
    i_arr = np.arange(n)
    valid_mask = (j_arr > i_arr) & (j_arr < n)
    i_valid = i_arr[valid_mask]
    j_valid = j_arr[valid_mask]
    assert i_valid.size > 0, f"no valid frame pairs for delta_m={delta_m}, traj too short"

    d_pos_est = traj_est[j_valid, :2] - traj_est[i_valid, :2]
    d_pos_gt = traj_gt[j_valid, :2] - traj_gt[i_valid, :2]
    d_pos_err = d_pos_est - d_pos_gt
    trans_err = np.sqrt(np.sum(d_pos_err ** 2, axis=1))

    d_theta_est = traj_est[j_valid, 2] - traj_est[i_valid, 2]
    d_theta_gt = traj_gt[j_valid, 2] - traj_gt[i_valid, 2]
    rot_err = np.abs(angle_mod(d_theta_est - d_theta_gt))

    delta_safe = max(delta_m, 1e-9)
    trans_err_norm = trans_err / delta_safe
    rot_err_norm = rot_err / delta_safe

    return {
        'rpe_trans_rmse': float(np.sqrt(np.mean(trans_err_norm ** 2))),
        'rpe_trans_mean': float(np.mean(trans_err_norm)),
        'rpe_rot_rmse': float(np.sqrt(np.mean(rot_err_norm ** 2))),
        'rpe_rot_mean': float(np.mean(rot_err_norm)),
    }


# === Phase 2: Map point cloud density ===

def compute_map_density(pointcloud, voxel_size_m=0.05):
    assert pointcloud.ndim == 2 and pointcloud.shape[1] == 2, \
        f"expected (M,2), got shape={pointcloud.shape}"
    assert voxel_size_m > 0, f"voxel_size_m must be positive, got {voxel_size_m}"
    M = pointcloud.shape[0]
    assert M > 0, "pointcloud must be non-empty"

    voxels = np.floor(pointcloud / voxel_size_m).astype(np.int64)
    unique_voxels = np.unique(voxels, axis=0)
    n_occupied = unique_voxels.shape[0]

    density = M / max(n_occupied, 1) / (voxel_size_m ** 2)
    return float(density)


# === Phase 3: Loop closure detection evaluation ===

def compute_loop_metrics(loop_detected, loop_true, min_index_gap=50):
    loop_detected_arr = np.asarray(loop_detected, dtype=np.int64).reshape(-1, 2)
    loop_true_arr = np.asarray(loop_true, dtype=np.int64).reshape(-1, 2)

    n_detected = int(loop_detected_arr.shape[0])
    n_true = int(loop_true_arr.shape[0])

    if n_detected == 0 or n_true == 0:
        return {
            'loop_recall': float(n_correct_zero(n_detected, n_true, True)),
            'loop_precision': float(n_correct_zero(n_detected, n_true, False)),
            'n_detected': n_detected,
            'n_true': n_true,
        }

    i_d = loop_detected_arr[:, 0:1]
    j_d = loop_detected_arr[:, 1:2]
    i_t = loop_true_arr[:, 0]
    j_t = loop_true_arr[:, 1]

    match_i = np.abs(i_d - i_t[np.newaxis, :]) <= 5
    match_j = np.abs(j_d - j_t[np.newaxis, :]) <= 5
    match = match_i & match_j
    n_correct = int(np.sum(np.any(match, axis=1)))

    recall = n_correct / max(n_true, 1)
    precision = n_correct / max(n_detected, 1)

    return {
        'loop_recall': float(recall),
        'loop_precision': float(precision),
        'n_detected': n_detected,
        'n_true': n_true,
    }


def n_correct_zero(n_detected, n_true, is_recall):
    if is_recall:
        return 1.0 if (n_detected > 0 and n_true == 0) else 0.0
    return 1.0 if (n_true > 0 and n_detected == 0) else 0.0


# === Phase 4: Latency distribution statistics ===

def compute_latency_profile(step_times_ms, window=10):
    step_times = np.asarray(step_times_ms, dtype=float).flatten()
    assert step_times.size > 0, "step_times_ms must be non-empty"
    assert window >= 1, f"window must be >=1, got {window}"

    n = step_times.size
    if n >= window:
        cumsum = np.concatenate([[0.0], np.cumsum(step_times)])
        window_sums = cumsum[window:] - cumsum[:-window]
        window_means = window_sums / window
    else:
        window_means = step_times

    return {
        'latency_mean_ms': float(np.mean(window_means)),
        'latency_p95_ms': float(np.percentile(window_means, 95)),
        'latency_p99_ms': float(np.percentile(window_means, 99)),
        'latency_max_ms': float(np.max(window_means)),
    }


# === Phase 5: Absolute trajectory error ATE (Umeyama alignment) ===

def compute_ate_tum(traj_est, traj_gt, align='se3'):
    assert align in ('se3', 'sim3'), f"align must be 'se3' or 'sim3', got {align}"
    assert traj_est.shape == traj_gt.shape, f"shape mismatch est={traj_est.shape} gt={traj_gt.shape}"
    assert traj_est.ndim == 2 and traj_est.shape[1] == 3, f"expected (N,3), got {traj_est.shape}"
    n = traj_est.shape[0]
    assert n >= 2, f"trajectory must have >=2 points, got n={n}"

    est_xy = traj_est[:, :2]
    gt_xy = traj_gt[:, :2]

    mu_est = est_xy.mean(axis=0)
    mu_gt = gt_xy.mean(axis=0)
    est_c = est_xy - mu_est
    gt_c = gt_xy - mu_gt

    H = gt_c.T @ est_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if align == 'sim3':
        est_norm_sq = float(np.sum(est_c ** 2))
        scale = float(np.sum(S) / max(est_norm_sq, 1e-12))
    else:
        scale = 1.0

    est_aligned = scale * est_c @ R.T + mu_gt
    err = est_aligned - gt_xy
    err_norm = np.sqrt(np.sum(err ** 2, axis=1))

    return {
        'ate_rmse': float(np.sqrt(np.mean(err_norm ** 2))),
        'ate_mean': float(np.mean(err_norm)),
        'ate_median': float(np.median(err_norm)),
        'ate_max': float(np.max(err_norm)),
        'ate_std': float(np.std(err_norm)),
    }
