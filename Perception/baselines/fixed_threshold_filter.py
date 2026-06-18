"""
Fixed-threshold filter baseline: voxel → fixed SDF → DBSCAN → Kalman with Euclidean association

author: Kat-yuan-eng (RuiWen Liao)
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (R_VOXEL, RESOLUTION, TAU_SDF, EPSILON_CLUSTER, N_MIN,
                    R_MARGIN, A_MAX_CLUSTER, SIGMA_POS, SIGMA_VEL,
                    SIGMA_OBS, D_ASSOC, N_CONFIRM, N_DELETE, DT)
from voxel_filter import voxel_filter, transform_to_global
from sdf_filter import compute_sdf, fixed_threshold_filter
from euclidean_cluster import cluster_euclidean, compute_cluster_attrs
from kalman_tracker import (create_tracker, track_predict, track_update,
                            track_manage, reset_tracker_id)

# === Phase 0: Fixed-threshold pipeline initialization ===

def init_fixed_pipeline(grid, grid_origin):
    """
    Initialize the fixed-threshold baseline pipeline state.

    :param grid: (np.ndarray) 2-D binary occupancy grid (1=occupied)
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :return: (dict) Pipeline state dict with 'sdf', 'grid_origin', 'trackers'
    """
    sdf = compute_sdf(grid, RESOLUTION)
    reset_tracker_id()
    return {'sdf': sdf, 'grid_origin': grid_origin, 'trackers': [], '_next_id': 0}

# === Phase 1: Fixed Euclidean association ===

def _fixed_euclidean_associate(detections, trackers, d_assoc=D_ASSOC):
    """
    Associate detections to trackers using Hungarian algorithm with Euclidean distance gating.

    :param detections: (list) List of detection dicts with 'center' key
    :param trackers: (list) List of tracker dicts with 'x_trk' key
    :param d_assoc: (float) Maximum association distance in meters
    :return: (tuple) (associations, unmatched_det, unmatched_trk)
    """
    if len(detections) == 0 or len(trackers) == 0:
        return [], list(range(len(detections))), list(range(len(trackers)))
    det_centers = np.array([d["center"] for d in detections])
    trk_positions = np.array([t['x_trk'][:2] for t in trackers])
    cost = np.linalg.norm(det_centers[:, None, :] - trk_positions[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    associations = []
    unmatched_det = set(range(len(detections)))
    unmatched_trk = set(range(len(trackers)))
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= d_assoc:
            associations.append((r, c))
            unmatched_det.discard(r)
            unmatched_trk.discard(c)
    return associations, list(unmatched_det), list(unmatched_trk)

# === Phase 2-4: Fixed-threshold pipeline main loop ===

def run_fixed_cycle(state, points_raw, pose, dt=None):
    """
    Execute one fixed-threshold baseline cycle: filter → cluster → track.

    :param state: (dict) Pipeline state dict from init_fixed_pipeline
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

    clusters = cluster_euclidean(points_dyn, EPSILON_CLUSTER, N_MIN)
    detections = compute_cluster_attrs(clusters, R_VOXEL, R_MARGIN, A_MAX_CLUSTER)
    t2 = time.perf_counter()

    trackers = track_predict(state['trackers'], dt, SIGMA_POS, SIGMA_VEL)
    associations, _, _ = _fixed_euclidean_associate(detections, trackers, D_ASSOC)
    trackers = track_update(trackers, associations, detections, SIGMA_OBS)
    trackers = track_manage(trackers, associations, detections, N_CONFIRM, N_DELETE)
    state['trackers'] = trackers
    t3 = time.perf_counter()

    confirmed = [t for t in trackers if t['confirmed']]
    obstacles = [{'center': t['x_trk'][:2].copy(), 'velocity': t['x_trk'][2:4].copy(),
                  'track_id': t['track_id'], 'confirmed': t['confirmed'],
                  'is_dynamic': t['is_dynamic']}
                 for t in confirmed]
    timing = {'total_ms': (t3 - t0) * 1000}
    return {'obstacles': obstacles, 'timing': timing,
            'points_dyn': points_dyn, 'state': state}
