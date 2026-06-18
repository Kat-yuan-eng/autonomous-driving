"""
DBSCAN + nearest-neighbor tracker baseline: voxel → fixed SDF → grid cluster → NN association

author: Kat-yuan-eng (RuiWen Liao)
"""
import time
from collections import defaultdict

import numpy as np

from config import (R_VOXEL, RESOLUTION, TAU_SDF, N_MIN,
                    R_MARGIN, D_ASSOC, N_CONFIRM, N_DELETE, DT)
from voxel_filter import voxel_filter, transform_to_global
from sdf_filter import compute_sdf, fixed_threshold_filter

# === Phase 0: Grid-NN pipeline initialization ===

def init_gridnn_pipeline(grid, grid_origin):
    """
    Initialize the grid-NN baseline pipeline state.

    :param grid: (np.ndarray) 2-D binary occupancy grid (1=occupied)
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :return: (dict) Pipeline state dict with 'sdf', 'grid_origin', 'trackers'
    """
    sdf = compute_sdf(grid, RESOLUTION)
    return {'sdf': sdf, 'grid_origin': grid_origin, 'trackers': [], '_next_id': 0}

# === Phase 1: Grid clustering ===

def grid_cluster(points, grid_size=0.3, n_min=N_MIN):
    """
    Cluster points using grid-based BFS connectivity.

    :param points: (np.ndarray) Nx2+ points to cluster
    :param grid_size: (float) Grid cell size for quantization in meters
    :param n_min: (int) Minimum cluster size
    :return: (list) List of np.ndarray, each Mx2 cluster points
    """
    if len(points) == 0:
        return []
    grid_idx = np.floor(points[:, :2] / grid_size).astype(np.int32)
    cell_map = defaultdict(list)
    for i, idx in enumerate(grid_idx):
        cell_map[tuple(idx)].append(i)
    visited_cells = set()
    clusters = []
    for cell, pt_indices in cell_map.items():
        if len(pt_indices) < n_min // 2:
            continue
        if cell in visited_cells:
            continue
        queue = [cell]
        visited_cells.add(cell)
        merged_indices = list(pt_indices)
        while queue:
            current = queue.pop(0)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in visited_cells or neighbor not in cell_map:
                        continue
                    if len(cell_map[neighbor]) >= n_min // 2:
                        visited_cells.add(neighbor)
                        queue.append(neighbor)
                        merged_indices.extend(cell_map[neighbor])
        if len(merged_indices) >= n_min:
            clusters.append(points[merged_indices])
    return clusters

# === Phase 2: Nearest-neighbor association ===

def _nn_associate(detections, trackers, d_thresh=D_ASSOC):
    """
    Associate detections to trackers using greedy nearest-neighbor matching.

    :param detections: (list) List of detection dicts with 'center' key
    :param trackers: (list) List of tracker dicts with 'pos' key
    :param d_thresh: (float) Maximum association distance in meters
    :return: (tuple) (associations, unmatched_det, unmatched_trk)
    """
    if len(detections) == 0 or len(trackers) == 0:
        return [], list(range(len(detections))), list(range(len(trackers)))
    det_centers = np.array([d["center"] for d in detections])
    trk_positions = np.array([t['pos'] for t in trackers])
    dists = np.linalg.norm(det_centers[:, None, :] - trk_positions[None, :, :], axis=2)
    associations = []
    used_det = set()
    used_trk = set()
    for _ in range(min(len(detections), len(trackers))):
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        if dists[idx] > d_thresh:
            break
        det_i, trk_i = int(idx[0]), int(idx[1])
        if det_i not in used_det and trk_i not in used_trk:
            associations.append((det_i, trk_i))
            used_det.add(det_i)
            used_trk.add(trk_i)
        dists[det_i, trk_i] = np.inf
    unmatched_det = [i for i in range(len(detections)) if i not in used_det]
    unmatched_trk = [i for i in range(len(trackers)) if i not in used_trk]
    return associations, unmatched_det, unmatched_trk

# === Phase 3-4: Grid-NN pipeline main loop ===

def run_gridnn_cycle(state, points_raw, pose, dt=None):
    """
    Execute one grid-NN baseline cycle: filter → grid cluster → NN track.

    :param state: (dict) Pipeline state dict from init_gridnn_pipeline
    :param points_raw: (np.ndarray) Nx2+ raw LiDAR points in local frame
    :param pose: (np.ndarray) Length-3 robot pose [x, y, theta]
    :param dt: (float) Time step in seconds; defaults to DT from config
    :return: (dict) Result dict with 'obstacles', 'timing', 'points_dyn', 'state'
    """
    dt = dt if dt is not None else DT
    t0 = time.perf_counter()

    points_voxel = voxel_filter(points_raw, R_VOXEL)
    points_global = transform_to_global(points_voxel, pose)
    points_dyn, _ = fixed_threshold_filter(
        points_global, state['sdf'], state['grid_origin'], RESOLUTION, TAU_SDF)
    t1 = time.perf_counter()

    clusters = grid_cluster(points_dyn, grid_size=0.3, n_min=N_MIN)
    detections = [{'center': cl[:, :2].mean(axis=0), 'radius': R_MARGIN,
                   'n_pts': len(cl)} for cl in clusters]
    t2 = time.perf_counter()

    trackers = state['trackers']
    associations, unmatched_det, _ = _nn_associate(detections, trackers, D_ASSOC)
    associated_trk_set = {c for _, c in associations}
    for det_i, trk_i in associations:
        trackers[trk_i]['pos'] = detections[det_i]["center"].copy()
        trackers[trk_i]['n_matched'] += 1
        trackers[trk_i]['n_lost'] = 0
        trackers[trk_i]['history'].append(tuple(trackers[trk_i]['pos']))
        if trackers[trk_i]['n_matched'] >= N_CONFIRM:
            trackers[trk_i]['confirmed'] = True
    for i, trk in enumerate(trackers):
        if i not in associated_trk_set:
            trk['n_lost'] += 1
    for det_i in unmatched_det:
        state['_next_id'] += 1
        trackers.append({
            'track_id': state['_next_id'] - 1,
            'pos': detections[det_i]["center"].copy(),
            'n_matched': 1,
            'confirmed': False,
            'n_lost': 0,
            'history': [tuple(detections[det_i]["center"])],
        })
    trackers = [t for t in trackers if t['n_lost'] < N_DELETE]
    state['trackers'] = trackers
    t3 = time.perf_counter()

    confirmed = [t for t in trackers if t['confirmed']]
    obstacles = [{'center': t['pos'].copy(), 'track_id': t['track_id'],
                  'confirmed': t['confirmed']}
                 for t in confirmed]
    timing = {'total_ms': (t3 - t0) * 1000}
    return {'obstacles': obstacles, 'timing': timing,
            'points_dyn': points_dyn, 'state': state}
