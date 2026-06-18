"""Branch-and-bound loop closure detection

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Branch-and-bound loop closure detection ===
# === Phase 2: Cauchy robust kernel ===
# === Phase 3: Match score feedback suppression ===
import numpy as np

from SLAM.config import (BB_LAYERS, BB_RES_COARSE, BB_RES_FINE,
    LOOP_CLOSURE_SCORE_MIN, CAUCHY_C_ODOM, CAUCHY_C_LOOP)


def cauchy_loss(s, c):
    """
    Cauchy robust loss function.

    :param s: (float) Squared residual
    :param c: (float) Cauchy scale parameter
    :return: (float) Robust loss value
    """
    assert c > 0, f"Cauchy scale must be positive, got {c}"
    return c**2 * np.log1p(s / c**2)


def cauchy_derivative(s, c):
    """
    Derivative of Cauchy robust loss.

    :param s: (float) Squared residual
    :param c: (float) Cauchy scale parameter
    :return: (float) Derivative value
    """
    return 1.0 / (1.0 + s / c**2)


def build_multi_resolution_grid(prob_grid, n_layers=BB_LAYERS):
    """
    Build multi-resolution grid pyramid by successive 2x max-pooling.

    :param prob_grid: (np.ndarray) Probability occupancy grid, shape (H, W)
    :param n_layers: (int) Number of pyramid layers
    :return: (list) List of grids from fine to coarse
    """
    assert prob_grid.ndim == 2, f"prob_grid must be 2D, got shape {prob_grid.shape}"
    layers = [prob_grid]
    for layer in range(1, n_layers):
        h, w = layers[-1].shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        coarse = np.zeros((h2, w2))
        for i in range(h2):
            for j in range(w2):
                block = layers[-1][i*2:min(i*2+2, h), j*2:min(j*2+2, w)]
                coarse[i, j] = block.max()
        layers.append(coarse)
    return layers


def branch_and_bound_search(scan_points, grid_layers, origin, resolution,
                             score_min=LOOP_CLOSURE_SCORE_MIN,
                             n_theta=72, search_radius=5.0):
    """
    Branch-and-bound search for best scan-to-map alignment.

    :param scan_points: (np.ndarray) Scan points, shape (N, 2)
    :param grid_layers: (list) Multi-resolution grid pyramid
    :param origin: (np.ndarray) Grid origin [ox, oy], shape (2,)
    :param resolution: (float) Grid resolution in meters
    :param score_min: (float) Minimum score threshold
    :param n_theta: (int) Number of angular search steps
    :param search_radius: (float) Search radius in meters
    :return: (tuple) (best_pose, best_score) best alignment pose and score
    """
    assert len(grid_layers) > 0, "must provide at least one grid layer"
    assert scan_points.ndim == 2 and scan_points.shape[1] >= 2, f"scan_points must be Nx2+, got shape {scan_points.shape}"
    n_xy = int(search_radius / resolution)
    best_score = score_min
    best_pose = np.array([0.0, 0.0, 0.0])
    angle_step = 2 * np.pi / n_theta
    for ai in range(n_theta):
        theta = ai * angle_step
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts_r = (rot @ scan_points.T).T
        nodes = []
        for xi in range(-n_xy, n_xy):
            for yi in range(-n_xy, n_xy):
                tx = xi * resolution
                ty = yi * resolution
                score = _score_at_layer_coarse(pts_r, tx, ty, grid_layers[-1], origin, resolution)
                if score >= best_score:
                    nodes.append((score, tx, ty, theta, len(grid_layers) - 1))
        nodes.sort(key=lambda x: -x[0])
        while nodes:
            score, tx, ty, theta, layer = nodes.pop(0)
            if layer == 0:
                if score > best_score:
                    best_score = score
                    best_pose = np.array([tx, ty, theta])
            else:
                sub_res = resolution * (2 ** layer)
                for dx in [0, sub_res / 2]:
                    for dy in [0, sub_res / 2]:
                        ntx = tx + dx
                        nty = ty + dy
                        sub_score = _score_at_layer_coarse(pts_r, ntx, nty, grid_layers[layer - 1], origin, resolution)
                        if sub_score >= best_score:
                            nodes.append((sub_score, ntx, nty, theta, layer - 1))
                nodes.sort(key=lambda x: -x[0])
    return best_pose, best_score


def _score_at_layer_coarse(points, tx, ty, grid, origin, resolution):
    """
    Score scan points against a single grid layer at given translation.

    :param points: (np.ndarray) Rotated scan points, shape (N, 2)
    :param tx: (float) Translation x in meters
    :param ty: (float) Translation y in meters
    :param grid: (np.ndarray) Grid at this layer
    :param origin: (np.ndarray) Grid origin, shape (2,)
    :param resolution: (float) Grid resolution in meters
    :return: (float) Sum of grid values at hit cells
    """
    px = (points[:, 0] + tx - origin[0]) / resolution
    py = (points[:, 1] + ty - origin[1]) / resolution
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    mask = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
    if not np.any(mask):
        return 0.0
    return float(np.sum(grid[iy[mask], ix[mask]]))


def detect_loop_closure_bb(current_scan, current_pose, submaps, grid_layers,
                         origin, resolution, n_theta=36, search_radius=1.0):
    """
    Detect loop closure by matching current scan against all submaps
    using branch-and-bound search.

    :param current_scan: (np.ndarray) Current scan points, shape (N, 2)
    :param current_pose: (np.ndarray) Current pose [x, y, theta], shape (3,)
    :param submaps: (list) List of submap probability grids
    :param grid_layers: (list) Grid layers (unused, kept for interface compatibility)
    :param origin: (np.ndarray) Grid origin, shape (2,)
    :param resolution: (float) Grid resolution in meters
    :param n_theta: (int) Number of angular search steps
    :param search_radius: (float) Search radius in meters
    :return: (tuple or None) (submap_idx, rel_pose, info_matrix, score) or None if no closure
    """
    best_score = LOOP_CLOSURE_SCORE_MIN
    best_constraint = None
    for submap_idx, prob_grid in enumerate(submaps):
        g_layers = build_multi_resolution_grid(prob_grid)
        pose, score = branch_and_bound_search(current_scan, g_layers, origin,
                                                resolution, LOOP_CLOSURE_SCORE_MIN,
                                                n_theta, search_radius)
        if score > LOOP_CLOSURE_SCORE_MIN:
            rel_pose = _compute_relative_pose(current_pose, pose)
            info_matrix = _compute_info_matrix(score)
            if score > best_score:
                best_score = score
                best_constraint = (submap_idx, rel_pose, info_matrix, score)
    return best_constraint


# === Phase 5: Lightweight loop closure detection ===
def detect_loop_closure(current_pose, keyframes, current_scan, keyframe_scans,
                        dist_thresh, icp_thresh):
    assert current_pose.shape[0] >= 3, f"current_pose must have >=3 elements, got {current_pose.shape[0]}"
    assert len(keyframes) > 0, "keyframes must not be empty"
    assert dist_thresh > 0.0, f"dist_thresh must be positive, got {dist_thresh}"
    assert icp_thresh > 0.0, f"icp_thresh must be positive, got {icp_thresh}"

    keyframes_arr = np.asarray(keyframes)
    dists = np.sqrt(np.sum((keyframes_arr[:, :2] - current_pose[:2]) ** 2, axis=1))

    candidates = np.where(dists < dist_thresh)[0]
    if len(candidates) == 0:
        return None

    best_idx = -1
    best_score = float('inf')
    for idx in candidates:
        if keyframe_scans[idx] is None or current_scan is None:
            continue
        score = _simple_icp_score(current_scan, keyframe_scans[idx])
        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0 or best_score >= icp_thresh:
        return None

    return {
        'keyframe_idx': int(best_idx),
        'relative_pose': keyframes_arr[best_idx].copy(),
        'icp_score': round(float(best_score), 6),
        'distance': round(float(dists[best_idx]), 6)
    }


def _simple_icp_score(scan_a, scan_b):
    assert scan_a.ndim == 2 and scan_b.ndim == 2, "scans must be 2D arrays"
    from scipy.spatial.distance import cdist
    if scan_a.shape[0] == 0 or scan_b.shape[0] == 0:
        return float('inf')
    dists = cdist(scan_a[:, :2], scan_b[:, :2])
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def _compute_relative_pose(pose_a, pose_b):
    """
    Compute relative pose from pose_a to pose_b with angle wrapping.

    :param pose_a: (np.ndarray) First pose [x, y, theta], shape (3,)
    :param pose_b: (np.ndarray) Second pose [x, y, theta], shape (3,)
    :return: (np.ndarray) Relative pose [dx, dy, dtheta], shape (3,)
    """
    dx = pose_b[0] - pose_a[0]
    dy = pose_b[1] - pose_a[1]
    dtheta = np.arctan2(np.sin(pose_b[2] - pose_a[2]), np.cos(pose_b[2] - pose_a[2]))
    return np.array([dx, dy, dtheta])


def _compute_info_matrix(score):
    """
    Compute information matrix scaled by match score above threshold.

    :param score: (float) Loop closure match score
    :return: (np.ndarray) Information matrix, shape (3, 3)
    """
    base_info = np.diag([50.0, 50.0, 100.0])
    return base_info * max(0.1, score - 0.6) * 5.0