"""
Perception pipeline: voxel filter → SDF adaptive filter → clustering → Kalman tracking

author: Kat-yuan-eng (RuiWen Liao)
"""
import time

from config import *
from voxel_filter import voxel_filter, transform_to_global
from sdf_filter import compute_sdf, sdf_adaptive_filter
from euclidean_cluster import cluster_euclidean, compute_cluster_attrs
from kalman_tracker import create_tracker, track_predict, hungarian_associate, track_update, track_manage, reset_tracker_id

# === Phase 0: Perception pipeline initialization ===

def init_pipeline(grid, grid_origin):
    """
    Initialize the perception pipeline state with SDF and empty tracker list.

    :param grid: (np.ndarray) 2-D binary occupancy grid (1=occupied)
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :return: (dict) Pipeline state dict with 'sdf', 'grid_origin', 'trackers'
    """
    sdf = compute_sdf(grid, RESOLUTION)
    reset_tracker_id()
    return {'sdf': sdf, 'grid_origin': grid_origin, 'trackers': [], '_next_id': 0}

# === Phase 1-4: Perception pipeline main loop ===

def run_cycle(state, points_raw, pose, dt=None):
    """
    Execute one perception cycle: filter → cluster → track → build obstacle list.

    :param state: (dict) Pipeline state dict from init_pipeline or previous cycle
    :param points_raw: (np.ndarray) Nx2+ raw LiDAR points in local frame
    :param pose: (np.ndarray) Length-3 robot pose [x, y, theta]
    :param dt: (float) Time step in seconds; defaults to DT from config
    :return: (dict) Result dict with 'obstacles', 'timing', 'points_dyn', 'state'
    """
    dt = dt if dt is not None else DT
    t0 = time.perf_counter()

    # === Phase 1: Voxel filter + coordinate transform + SDF adaptive filter ===
    points_voxel = voxel_filter(points_raw, R_VOXEL)
    points_global = transform_to_global(points_voxel, pose)
    points_dyn, _ = sdf_adaptive_filter(
        points_global, state['sdf'], state['grid_origin'], RESOLUTION,
        TAU_SDF, BETA_SDF, D_NEAR)
    t1 = time.perf_counter()

    # === Phase 2: Euclidean clustering + cluster attributes ===
    clusters = cluster_euclidean(points_dyn, EPSILON_CLUSTER, N_MIN)
    detections = compute_cluster_attrs(clusters, R_VOXEL, R_MARGIN, A_MAX_CLUSTER)
    t2 = time.perf_counter()

    # === Phase 3: Kalman tracking ===
    trackers = track_predict(state['trackers'], dt, SIGMA_POS, SIGMA_VEL)
    associations, unmatched_det, unmatched_trk = hungarian_associate(
        detections, trackers, SIGMA_OBS, N_CONFIRM,
        CHI2_NEW, CHI2_STABLE)
    trackers = track_update(trackers, associations, detections, SIGMA_OBS)
    for det_idx, trk_idx in associations:
        trackers[trk_idx]['radius'] = detections[det_idx]['radius']
    trackers = track_manage(trackers, associations, detections, N_CONFIRM, N_DELETE)
    state['trackers'] = trackers
    t3 = time.perf_counter()

    # === Phase 4: Build obstacle list ===
    confirmed = [t for t in trackers if t['confirmed']]
    obstacles = [{'center': t['x_trk'][:2].copy(), 'velocity': t['x_trk'][2:4].copy(),
                  'radius': t.get('radius', R_MARGIN),
                  'track_id': t['track_id'], 'confirmed': t['confirmed'],
                  'is_dynamic': t['is_dynamic']}
                 for t in confirmed]
    t4 = time.perf_counter()

    timing = {
        'phase1_ms': (t1 - t0) * 1000,
        'phase2_ms': (t2 - t1) * 1000,
        'phase3_ms': (t3 - t2) * 1000,
        'phase4_ms': (t4 - t3) * 1000,
        'total_ms': (t4 - t0) * 1000,
    }
    return {'obstacles': obstacles, 'timing': timing,
            'points_dyn': points_dyn, 'state': state}
