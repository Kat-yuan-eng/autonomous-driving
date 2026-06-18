"""Real-time correlative scan matching

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Real-time correlative scan matching ===
import numpy as np
from scipy.ndimage import map_coordinates

from SLAM.config import (SEARCH_WIN_LIN, SEARCH_WIN_ANG, VOXEL_FILTER_SIZE,
    LIDAR_N_BEAMS, LIDAR_RANGE_MAX, OCC_PROB_OCCUPIED, PROB_OCCUPIED_INIT)


def real_time_correlative_scan_match(scan_ranges, angles, grid, origin, resolution,
                                      init_pose=np.zeros(3),
                                      search_win_lin=SEARCH_WIN_LIN, search_win_ang=SEARCH_WIN_ANG,
                                      ang_step_deg=1.0, lin_step=None,
                                      early_stop_score=None):
    assert scan_ranges.ndim == 1, f"scan_ranges must be 1D, got shape {scan_ranges.shape}"
    assert grid.ndim == 2, f"grid must be 2D, got shape {grid.shape}"
    points = _scan_to_points(scan_ranges, angles)
    actual_lin_step = lin_step if lin_step is not None else resolution
    best_pose, best_score = _single_search(points, grid, origin, resolution,
                                            init_pose, search_win_lin, search_win_ang,
                                            ang_step_deg, actual_lin_step,
                                            early_stop_score)
    return best_pose, best_score


def _single_search(points, grid, origin, resolution, init_pose, search_win_lin, search_win_ang,
                   ang_step_deg=1.0, lin_step=None, early_stop_score=None):
    step = max(1, len(points) // 60)
    points_sub = points[::step]
    n_pts = len(points_sub)
    grid_h, grid_w = grid.shape
    actual_lin_step = lin_step if lin_step is not None else resolution
    n_xy = max(3, int(search_win_lin / actual_lin_step) * 2 + 1)
    ang_step = np.deg2rad(ang_step_deg)
    n_theta = max(3, int(search_win_ang / ang_step) * 2 + 1)
    best_score = -np.inf
    best_pose = init_pose.copy()
    di_vals = np.arange(-n_xy // 2, n_xy // 2 + 1)
    dj_vals = np.arange(-n_xy // 2, n_xy // 2 + 1)
    dk_vals = np.arange(-n_theta // 2, n_theta // 2 + 1)
    occ_mask = (grid > OCC_PROB_OCCUPIED).astype(np.float32)
    for dk in dk_vals:
        dtheta = init_pose[2] + dk * ang_step
        c, s = np.cos(dtheta), np.sin(dtheta)
        pts_x_rot = c * points_sub[:, 0] - s * points_sub[:, 1]
        pts_y_rot = s * points_sub[:, 0] + c * points_sub[:, 1]
        base_px = (pts_x_rot + init_pose[0] - origin[0]) / resolution
        base_py = (pts_y_rot + init_pose[1] - origin[1]) / resolution
        di_grid, dj_grid = np.meshgrid(di_vals, dj_vals, indexing='ij')
        di_flat = di_grid.ravel()
        dj_flat = dj_grid.ravel()
        n_shift = len(di_flat)
        px_all = np.round(base_px[None, :] + (di_flat * actual_lin_step / resolution)[:, None]).astype(np.int32)
        py_all = np.round(base_py[None, :] + (dj_flat * actual_lin_step / resolution)[:, None]).astype(np.int32)
        valid = (px_all >= 0) & (px_all < grid_w) & (py_all >= 0) & (py_all < grid_h)
        px_clipped = np.clip(px_all, 0, grid_w - 1)
        py_clipped = np.clip(py_all, 0, grid_h - 1)
        hits = occ_mask[py_clipped, px_clipped]
        scores = np.sum(hits * valid, axis=1)
        max_idx = int(np.argmax(scores))
        if scores[max_idx] > best_score:
            best_score = float(scores[max_idx])
            best_pose = np.array([
                init_pose[0] + di_flat[max_idx] * actual_lin_step,
                init_pose[1] + dj_flat[max_idx] * actual_lin_step,
                dtheta])
        if early_stop_score is not None and best_score >= early_stop_score:
            break
    return best_pose, best_score


def ceres_scan_match_placeholder(scan_ranges, angles, grid, origin, resolution, init_pose):
    points = _scan_to_points(scan_ranges, angles)
    wx = (init_pose[0] - origin[0]) / resolution
    wy = (init_pose[1] - origin[1]) / resolution
    rot = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), 0], [np.sin(init_pose[2]), np.cos(init_pose[2]), 0], [0, 0, 1]])
    pts_g = (rot[:2, :2] @ points.T).T + np.array([init_pose[0], init_pose[1]])
    pts_pix = np.column_stack([(pts_g[:, 0] - origin[0]) / resolution, (pts_g[:, 1] - origin[1]) / resolution])
    vals = map_coordinates(grid, [pts_pix[:, 1], pts_pix[:, 0]], order=3, mode='constant', cval=0.0)
    residual = np.sum((1.0 - vals)**2)
    return init_pose, float(residual)


def _scan_to_points(ranges, angles):
    mask = ranges < LIDAR_RANGE_MAX * 0.95
    px = np.where(mask, ranges * np.cos(angles), 0.0)
    py = np.where(mask, ranges * np.sin(angles), 0.0)
    return np.column_stack([px, py])


def _score_points(points, grid, origin, resolution):
    px = (points[:, 0] - origin[0]) / resolution
    py = (points[:, 1] - origin[1]) / resolution
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    mask = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
    if not np.any(mask):
        return 0.0
    hits = grid[iy[mask], ix[mask]]
    return float(np.sum(hits > OCC_PROB_OCCUPIED))
