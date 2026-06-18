"""Cartographer pure localization mode

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Cartographer Pure Localization mode wrapper ===
# === Phase 2: Global relocalization ===
# === Phase 3: Health monitoring ===
# === Phase 4: Adaptive search window and voxel filter ===
import numpy as np

from SLAM.config import (SCORE_HEALTHY, SCORE_DEGRADE, TIME_SINCE_MATCH_MAX,
    RELOC_SCORE_TH, RELOC_TIMEOUT, SEARCH_WIN_LIN, SEARCH_WIN_ANG,
    LIDAR_RANGE_MAX, CARTO_K_COV, CARTO_K_INNOV, CARTO_COV_TRACE_MAX,
    VOXEL_DENSITY_LOW, VOXEL_DENSITY_HIGH, VOXEL_SIZE_MIN, VOXEL_SIZE_MAX,
    SCORE_HISTORY_LEN, SCORE_HEALTHY_LOW, SCORE_HEALTHY_HIGH,
    SCORE_DEGRADE_LOW, SCORE_DEGRADE_HIGH, CARTO_WIN_COV_GAIN)
from SLAM.mapping.scan_matching import real_time_correlative_scan_match

SCORE_NORMALIZE_FACTOR = 10.0


# === Phase 4: Adaptive search window and voxel filter ===
def adaptive_search_window(innov_lin, innov_ang, P_pred):
    assert P_pred is not None, "P_pred must not be None for adaptive window"
    assert P_pred.shape[0] >= 3 and P_pred.shape[1] >= 3, \
        f"P_pred must be at least 3x3, got shape {P_pred.shape}"
    cov_trace = min(float(np.trace(P_pred[:3, :3])), CARTO_COV_TRACE_MAX)
    cov_ratio = CARTO_K_COV * cov_trace / CARTO_COV_TRACE_MAX
    cov_fine = CARTO_WIN_COV_GAIN * (cov_trace / max(CARTO_COV_TRACE_MAX, 1e-9)) ** 2
    adaptive_win_lin = SEARCH_WIN_LIN * (1.0 + CARTO_K_INNOV * min(innov_lin, 1.0) + cov_ratio + cov_fine)
    adaptive_win_ang = SEARCH_WIN_ANG * (1.0 + CARTO_K_INNOV * min(innov_ang, 1.0) + cov_ratio + cov_fine)
    return round(adaptive_win_lin, 6), round(adaptive_win_ang, 6)


def adaptive_voxel_size(point_density):
    assert point_density >= 0.0, f"point_density must be non-negative, got {point_density}"
    if point_density < VOXEL_DENSITY_LOW:
        voxel_size = VOXEL_SIZE_MIN
    elif point_density > VOXEL_DENSITY_HIGH:
        voxel_size = VOXEL_SIZE_MAX
    else:
        voxel_size = VOXEL_SIZE_MIN + (VOXEL_SIZE_MAX - VOXEL_SIZE_MIN) * (
            point_density - VOXEL_DENSITY_LOW) / (VOXEL_DENSITY_HIGH - VOXEL_DENSITY_LOW)
    return round(voxel_size, 6)


# === Phase 5: Adaptive voxel filter downsampling ===
def adaptive_voxel_filter(scan_points, density=None):
    assert scan_points.ndim == 2, f"scan_points must be 2D, got {scan_points.ndim}D"
    assert scan_points.shape[1] >= 2, f"scan_points must have >=2 cols, got {scan_points.shape[1]}"

    n_pts = scan_points.shape[0]
    if n_pts == 0:
        return scan_points

    if density is None:
        bbox = np.ptp(scan_points[:, :2], axis=0)
        area = max(float(np.prod(bbox)), 1e-6)
        density = n_pts / area

    voxel_size = adaptive_voxel_size(density)

    voxel_idx = np.floor(scan_points[:, :2] / max(voxel_size, 1e-9)).astype(np.int64)
    keys = voxel_idx[:, 0] * 100000 + voxel_idx[:, 1]
    _, unique_idx = np.unique(keys, return_index=True)
    filtered = scan_points[unique_idx]

    return filtered


def adaptive_score_thresholds(score_history):
    score_arr = np.asarray(score_history, dtype=float)
    assert score_arr.size > 0, "score_history must not be empty"
    recent = score_arr[-SCORE_HISTORY_LEN:]
    score_mean = float(np.mean(recent))
    score_std = float(np.std(recent))
    score_healthy = float(np.clip(score_mean - score_std,
                                   SCORE_HEALTHY_LOW, SCORE_HEALTHY_HIGH))
    score_degrade = float(np.clip(score_mean - 2.0 * score_std,
                                   SCORE_DEGRADE_LOW, SCORE_DEGRADE_HIGH))
    return {'score_healthy': round(score_healthy, 6),
            'score_degrade': round(score_degrade, 6)}


class CartoPureLoc:

    def __init__(self, prob_grid, origin, resolution):
        assert prob_grid.ndim == 2, f"prob_grid must be 2D, got shape {prob_grid.shape}"
        self.prob_grid = prob_grid
        self.origin = origin
        self.resolution = resolution
        self.last_pose = np.array([0.0, 0.0, 0.0])
        self.last_match_score = 0.0
        self.time_since_match = 0.0
        self.n_loop_constraints = 0
        self.reloc_mode = False
        self.reloc_timer = 0.0
        self.reloc_callback = None
        self.score_history = []
        self.current_voxel_size = VOXEL_SIZE_MIN
        self.score_healthy_adapt = SCORE_HEALTHY
        self.score_degrade_adapt = SCORE_DEGRADE

    def set_reloc_callback(self, callback):
        self.reloc_callback = callback

    def update(self, scan_ranges, angles, pred_pose, dt, P_pred=None):
        ref_pose = pred_pose[:3] if pred_pose is not None else self.last_pose[:3]
        innov_lin_pre = np.sqrt((self.last_pose[0]-ref_pose[0])**2
                                + (self.last_pose[1]-ref_pose[1])**2)
        innov_ang_pre = abs(np.arctan2(np.sin(self.last_pose[2]-ref_pose[2]),
                                        np.cos(self.last_pose[2]-ref_pose[2])))
        if P_pred is not None:
            adaptive_win_lin, adaptive_win_ang = adaptive_search_window(
                innov_lin_pre, innov_ang_pre, P_pred)
        else:
            adaptive_win_lin = round(SEARCH_WIN_LIN * (1.0 + min(innov_lin_pre * 2.0, 2.0)), 6)
            adaptive_win_ang = round(SEARCH_WIN_ANG, 6)
        if len(self.score_history) > 0:
            thresholds = adaptive_score_thresholds(self.score_history)
            self.score_healthy_adapt = thresholds['score_healthy']
            self.score_degrade_adapt = thresholds['score_degrade']
        else:
            self.score_healthy_adapt = SCORE_HEALTHY
            self.score_degrade_adapt = SCORE_DEGRADE
        valid_mask = (scan_ranges > 0.0) & (scan_ranges < LIDAR_RANGE_MAX)
        n_valid = int(np.sum(valid_mask))
        scan_area = np.pi * LIDAR_RANGE_MAX**2 + 1e-9
        point_density = n_valid / scan_area
        self.current_voxel_size = adaptive_voxel_size(point_density)
        ranges_valid = scan_ranges[valid_mask]
        angles_valid = angles[valid_mask]
        px_valid = ranges_valid * np.cos(angles_valid)
        py_valid = ranges_valid * np.sin(angles_valid)
        points_pre = np.column_stack([px_valid, py_valid])
        points_filt = adaptive_voxel_filter(points_pre, density=point_density)
        if points_filt.shape[0] > 0:
            ranges_match = np.sqrt(points_filt[:, 0]**2 + points_filt[:, 1]**2)
            angles_match = np.arctan2(points_filt[:, 1], points_filt[:, 0])
        else:
            ranges_match = scan_ranges
            angles_match = angles
        match_pose, score_raw = real_time_correlative_scan_match(
            ranges_match, angles_match, self.prob_grid, self.origin, self.resolution,
            init_pose=pred_pose[:3] if pred_pose is not None else np.zeros(3),
            search_win_lin=adaptive_win_lin,
            search_win_ang=adaptive_win_ang,
            early_stop_score=SCORE_NORMALIZE_FACTOR * self.score_healthy_adapt * 1.5)
        match_pose, score_raw = real_time_correlative_scan_match(
            ranges_match, angles_match, self.prob_grid, self.origin, self.resolution,
            init_pose=match_pose,
            search_win_lin=max(adaptive_win_lin * 0.3, self.resolution),
            search_win_ang=np.deg2rad(2.0),
            ang_step_deg=0.2,
            lin_step=self.resolution * 0.5)
        score = float(np.clip(score_raw / SCORE_NORMALIZE_FACTOR, 0.0, 1.0))
        self.score_history.append(score)
        if len(self.score_history) > SCORE_HISTORY_LEN * 2:
            self.score_history = self.score_history[-SCORE_HISTORY_LEN:]
        innovation = match_pose[:3] - ref_pose
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))
        innov_lin = np.sqrt(innovation[0]**2 + innovation[1]**2)
        innov_ang = abs(innovation[2])
        if score > self.score_healthy_adapt and innov_lin < 1.0 and innov_ang < np.deg2rad(45):
            self.last_pose = match_pose
            self.last_match_score = score
            self.time_since_match = 0.0
            self.reloc_mode = False
        elif score > self.score_degrade_adapt and innov_lin < 0.5 and innov_ang < np.deg2rad(30):
            blend = min(score / max(self.score_healthy_adapt, 1e-9), 1.0)
            self.last_pose = blend * match_pose + (1 - blend) * self.last_pose
            self.last_match_score = score
            self.time_since_match += dt
        else:
            self.time_since_match += dt
            if self.time_since_match > TIME_SINCE_MATCH_MAX:
                self.reloc_mode = True
        if self.reloc_mode:
            self.reloc_timer += dt
            reloc_pose, reloc_score = self._global_relocalization(scan_ranges, angles, prior_pose=pred_pose[:3] if pred_pose is not None else None)
            reloc_score_norm = float(np.clip(reloc_score / SCORE_NORMALIZE_FACTOR, 0.0, 1.0))
            yaw_innov = abs(np.arctan2(np.sin(reloc_pose[2] - self.last_pose[2]),
                                        np.cos(reloc_pose[2] - self.last_pose[2])))
            if reloc_score_norm > RELOC_SCORE_TH and yaw_innov < np.deg2rad(90):
                self.last_pose = reloc_pose
                self.last_match_score = reloc_score_norm
                self.time_since_match = 0.0
                self.reloc_mode = False
                self.reloc_timer = 0.0
                if self.reloc_callback is not None:
                    self.reloc_callback(self.last_pose.copy())
            elif self.reloc_timer > RELOC_TIMEOUT:
                self.reloc_mode = False
                self.reloc_timer = 0.0
        return self.last_pose.copy()

    def _global_relocalization(self, scan_ranges, angles, prior_pose=None):
        if prior_pose is not None:
            expanded_search = SEARCH_WIN_LIN * 5
            expanded_ang = np.deg2rad(60.0)
            reloc_pose, score = real_time_correlative_scan_match(
                scan_ranges, angles, self.prob_grid, self.origin, self.resolution,
                init_pose=prior_pose,
                search_win_lin=expanded_search, search_win_ang=expanded_ang,
                ang_step_deg=2.0, lin_step=self.resolution * 2)
        else:
            expanded_search = SEARCH_WIN_LIN * 20
            expanded_ang = SEARCH_WIN_ANG * 10
            reloc_pose, score = real_time_correlative_scan_match(
                scan_ranges, angles, self.prob_grid, self.origin, self.resolution,
                search_win_lin=expanded_search, search_win_ang=expanded_ang,
                ang_step_deg=5.0, lin_step=self.resolution * 4)
        return reloc_pose, score

    def check_health(self):
        if len(self.score_history) > 0:
            thresholds = adaptive_score_thresholds(self.score_history)
            score_healthy_adapt = thresholds['score_healthy']
            score_degrade_adapt = thresholds['score_degrade']
        else:
            score_healthy_adapt = SCORE_HEALTHY
            score_degrade_adapt = SCORE_DEGRADE
        healthy = (self.last_match_score > score_healthy_adapt
                   and self.time_since_match < TIME_SINCE_MATCH_MAX)
        degraded = (self.last_match_score > score_degrade_adapt
                    and self.time_since_match < TIME_SINCE_MATCH_MAX)
        failed = not degraded and not self.reloc_mode
        return {'healthy': healthy, 'degraded': degraded, 'failed': failed,
                'score': self.last_match_score, 'time_since_match': self.time_since_match,
                'reloc_mode': self.reloc_mode,
                'score_healthy': score_healthy_adapt,
                'score_degrade': score_degrade_adapt,
                'voxel_size': self.current_voxel_size}
