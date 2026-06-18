"""SLAM main entry point: offline mapping + online localization pipeline

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Configuration and simulation data generation ===
# === Phase 2: Offline mapping (Cartographer mode) ===
# === Phase 3: Online localization (Cartographer + UKF) ===
# === Phase 4: Comparison experiments ===
# === Phase 5: Testing ===
# === Phase 6: Visualization and result saving ===
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from SLAM.config import (UKF_DT, IMU_DT, LIDAR_N_BEAMS, LIDAR_RANGE_MAX,
    VOXEL_SIZE, N_SUBMAP_SCANS, LOOP_CLOSURE_SCORE_MIN, UKF_DIM)
from SLAM.slam_sim import (generate_reference_trajectory, generate_landmarks,
    generate_lidar_scan, generate_imu, generate_wheel_odom, angle_mod, bicycle_dynamics)
from SLAM.mapping.scan_matching import real_time_correlative_scan_match, _scan_to_points
from SLAM.mapping.submap_builder import SubmapCollection, voxel_filter
from SLAM.mapping.loop_closure_detector import detect_loop_closure_bb
from SLAM.mapping.pose_graph_optimizer import build_pose_graph, optimize_pose_graph_spa
from SLAM.localization.carto_pure_loc import CartoPureLoc
from SLAM.localization.ukf_fusion import (ukf_init, ukf_generate_sigma, ukf_predict,
    ukf_update_carto, ukf_adaptive_R_carto)
from SLAM.localization.degradation_manager import DegradationManager
from SLAM.compare_slam import run_all_algorithms, print_comparison, save_results
from SLAM.test_slam import run_all_tests
from SLAM.visualize_slam import (setup_rcparams, animate_trajectory_comparison,
    animate_map_building, visualize_comprehensive_comparison)


ROOT = pathlib.Path(__file__).parent
FIGS_DIR = ROOT.parent / 'figs'
RESULTS_DIR = ROOT.parent / 'results'


# === Phase 1: Configuration and simulation data generation ===

def generate_simulation_data(course_type='figure8'):
    """
    Generate complete simulation dataset including trajectory, landmarks, and sensor readings.

    :param course_type: (str) Trajectory type: 'figure8', 'circle', 'straight', or 'mixed'
    :return: (tuple) (ref_traj, landmarks, lidar_scans, imu_scans, wheel_scans)
    """
    ref_traj = generate_reference_trajectory(course_type, UKF_DT)
    landmarks = generate_landmarks(50, 12.0)
    n_steps = len(ref_traj)
    n_imu_sub = max(1, int(UKF_DT / IMU_DT))
    lidar_scans = np.zeros((n_steps, LIDAR_N_BEAMS))
    imu_scans = np.zeros((n_steps, n_imu_sub, 6))
    wheel_scans = np.zeros((n_steps, 2))
    v_profile = np.ones(n_steps) * 1.5
    delta_profile = np.zeros(n_steps)
    for i in range(n_steps):
        x, y, theta = ref_traj[i]
        lidar_scans[i] = generate_lidar_scan(x, y, theta, landmarks)
        v_inst = v_profile[min(i, len(v_profile) - 1)]
        for k in range(n_imu_sub):
            a_m, w_m = generate_imu(0.0, 0.0, np.array([1.0, 0.0, 0.0, 0.0]),
                                     np.zeros(3), np.zeros(3), IMU_DT)
            imu_scans[i, k, 0:3] = a_m
            imu_scans[i, k, 3:6] = w_m
        wheel_scans[i] = generate_wheel_odom(v_inst, 0.0, UKF_DT)
    print(f'[Phase1] generated {n_steps} steps, {n_imu_sub} IMU substeps, '
          f'{len(landmarks)} landmarks')
    return ref_traj, landmarks, lidar_scans, imu_scans, wheel_scans


# === Phase 2: Offline mapping (Cartographer mode) ===

def run_offline_mapping(ref_traj, lidar_scans, landmarks):
    """
    Build occupancy map offline using Cartographer-style submap accumulation and pose graph optimization.

    :param ref_traj: (np.ndarray) Reference trajectory, shape (N, 3)
    :param lidar_scans: (np.ndarray) LiDAR range measurements, shape (N, LIDAR_N_BEAMS)
    :param landmarks: (np.ndarray) Landmark positions, shape (M, 2)
    :return: (tuple) (combined_grid, origin, resolution, nodes, grid_layers_history)
    """
    n_steps = len(ref_traj)
    submap_collection = SubmapCollection(width=300, height=300, resolution=VOXEL_SIZE)
    angles = np.linspace(-np.pi, np.pi, LIDAR_N_BEAMS)
    odom_constraints = []
    nodes = []
    grid_layers_history = []
    keyframe_indices = []
    for i in range(0, n_steps, max(1, int(0.1 / UKF_DT))):
        pose = ref_traj[i]
        nodes.append(pose.copy())
        scan_ranges = lidar_scans[i]
        scan_points = _scan_to_points(scan_ranges, angles)
        scan_points = voxel_filter(scan_points)
        submap_collection.insert_scan(scan_points, pose)
        if len(nodes) > 1:
            z = nodes[-1] - nodes[-2]
            z[2] = angle_mod(z[2])
            Omega = np.eye(3) * 10.0
            odom_constraints.append((len(nodes) - 2, len(nodes) - 1, z, Omega,
                                     0.1, False))
        keyframe_indices.append(i)
        if i % 200 == 0:
            combined = submap_collection.get_combined_grid()
            grid_layers_history.append(combined)
    print(f'[Phase2] mapping: {len(nodes)} keyframes, {len(odom_constraints)} odom edges')
    loop_constraints = []
    for i in range(len(nodes)):
        if i < 10 or i % 10 != 0:
            continue
        current_scan = _scan_to_points(lidar_scans[keyframe_indices[i]], angles)
        prob_grids = submap_collection.get_all_prob_grids()
        if len(prob_grids) == 0:
            continue
        origin = np.array([submap_collection.map_origin_x,
                           submap_collection.map_origin_y])
        constraint = detect_loop_closure_bb(current_scan, nodes[i],
                                          prob_grids, prob_grids,
                                          origin, submap_collection.resolution)
        if constraint is not None:
            submap_idx, rel_pose, info_matrix, score = constraint
            if submap_idx < i:
                loop_constraints.append((submap_idx, i, rel_pose, info_matrix,
                                         0.3, True))
    print(f'[Phase2] loop closures detected: {len(loop_constraints)}')
    if len(nodes) >= 2:
        graph = build_pose_graph(nodes, odom_constraints, loop_constraints)
        opt_nodes = optimize_pose_graph_spa(graph, n_iter=20)
        print(f'[Phase2] pose graph optimized: {len(opt_nodes)} nodes')
        nodes = opt_nodes
    combined_grid = submap_collection.get_combined_grid()
    origin = np.array([submap_collection.map_origin_x,
                       submap_collection.map_origin_y])
    print(f'[Phase2] final map: {combined_grid.shape}, origin=({origin[0]:.3f},{origin[1]:.3f})')
    return combined_grid, origin, submap_collection.resolution, nodes, grid_layers_history


# === Phase 3: Online localization (Cartographer + UKF) ===

def run_online_localization(ref_traj, lidar_scans, imu_scans, wheel_scans,
                             prob_grid, origin, resolution):
    n_steps = len(ref_traj)
    angles = np.linspace(-np.pi, np.pi, LIDAR_N_BEAMS)
    carto_loc = CartoPureLoc(prob_grid, origin, resolution)
    x_ukf, P_ukf, W_m, W_c = ukf_init()
    sigma = ukf_generate_sigma(x_ukf, P_ukf, W_m, W_c)
    deg = DegradationManager()
    fused_traj = np.zeros((n_steps, 3))
    ukf_states = []
    ukf_covs = []
    health_log = []
    status_log = []
    for i in range(n_steps):
        scan_ranges = lidar_scans[i]
        carto_pose = carto_loc.update(scan_ranges, angles, x_ukf[:3], UKF_DT)
        carto_health = carto_loc.check_health()
        sigma = ukf_generate_sigma(x_ukf, P_ukf, W_m, W_c)
        x_pred, P_pred, sigma_pred = ukf_predict(sigma, W_m, W_c, UKF_DT)
        x_ukf, P_ukf = ukf_update_carto(x_pred, P_pred, sigma_pred, W_m, W_c,
                                          carto_pose[:3],
                                          ukf_adaptive_R_carto(
                                              carto_health['score'],
                                              carto_health['time_since_match']))
        status = deg.decide(carto_health)
        fused = deg.get_fused_pose(x_ukf[:3], carto_pose)
        if fused is not None:
            fused_traj[i] = fused
        else:
            fused_traj[i] = x_ukf[:3]
        ukf_states.append(x_ukf.copy())
        ukf_covs.append(P_ukf.copy())
        health_log.append(carto_health)
        status_log.append(status)
    print(f'[Phase3] localization complete: {n_steps} steps, {len(np.unique(status_log))} status transitions')
    return (fused_traj, np.array(ukf_states), np.array(ukf_covs),
            health_log, status_log, deg.status_log)


# === Phase 4-6: Comparison experiments, testing, visualization and result saving ===

def save_run_results(results):
    """
    Save metrics dictionary to JSON file.

    :param results: (dict) Algorithm metrics keyed by algorithm name
    :return: None
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(RESULTS_DIR / 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[save] metrics -> {RESULTS_DIR / "metrics.json"}')


def main():
    """
    Execute full SLAM pipeline: simulation -> mapping -> localization -> comparison -> test -> visualization.
    """
    t_start = time.perf_counter()
    setup_rcparams()
    print('=' * 70)
    print('SLAM System: Cartographer + UKF')
    print('=' * 70)

    # === Phase 1 ===
    t0 = time.perf_counter()
    ref_traj, landmarks, lidar_scans, imu_scans, wheel_scans = generate_simulation_data('figure8')
    print(f'[Phase1] done in {time.perf_counter() - t0:.2f}s')

    # === Phase 2 ===
    t0 = time.perf_counter()
    prob_grid, origin, resolution, opt_nodes, grid_history = run_offline_mapping(
        ref_traj, lidar_scans, landmarks)
    print(f'[Phase2] done in {time.perf_counter() - t0:.2f}s')

    # === Phase 3 ===
    t0 = time.perf_counter()
    (fused_traj, ukf_states, ukf_covs,
     health_log, status_log, deg_log) = run_online_localization(
        ref_traj, lidar_scans, imu_scans, wheel_scans,
        prob_grid, origin, resolution)
    print(f'[Phase3] done in {time.perf_counter() - t0:.2f}s')

    # === Phase 4: Comparison experiments ===
    t0 = time.perf_counter()
    compare_results = run_all_algorithms('figure8')
    compare_results['Cartographer-UKF'] = {
        'trajectory': fused_traj.tolist(),
        'metrics': _compute_metrics_simple(fused_traj, ref_traj)}
    print_comparison(compare_results)
    print(f'[Phase4] done in {time.perf_counter() - t0:.2f}s')

    # === Phase 5: Testing ===
    t0 = time.perf_counter()
    n_passed, n_failed, test_results = run_all_tests()
    all_pass = (n_failed == 0)
    print(f'[Phase5] tests: {"ALL PASS" if all_pass else "SOME FAILURES"} in {time.perf_counter() - t0:.2f}s')

    # === Phase 6: Visualization and result saving ===
    t0 = time.perf_counter()
    gt_traj = np.array(compare_results.get('ground_truth', {}).get('trajectory', ref_traj))
    ekf_traj = np.array(compare_results.get('EKF_SLAM', {}).get('trajectory', []))
    fs_traj = np.array(compare_results.get('FastSLAM', {}).get('trajectory', []))
    graph_traj = np.array(compare_results.get('GraphSLAM', {}).get('trajectory', []))
    algo_trajs = {}
    if len(ekf_traj) > 0:
        algo_trajs['EKF_SLAM'] = ekf_traj
    if len(fs_traj) > 0:
        algo_trajs['FastSLAM'] = fs_traj
    if len(graph_traj) > 0:
        algo_trajs['GraphSLAM'] = graph_traj
    if len(gt_traj) > 0:
        animate_trajectory_comparison(
            gt_traj,
            algo_trajs=algo_trajs if algo_trajs else None,
            save_path=str(FIGS_DIR / 'trajectory_comparison.gif'))
    if len(grid_history) > 0:
        est_landmarks_history = [landmarks + np.random.randn(*landmarks.shape) * 0.1 for _ in range(len(ref_traj))]
    animate_map_building(ref_traj, landmarks, est_landmarks_history,
                         save_path=str(FIGS_DIR / 'slam_map_building.gif'))
    visualize_comprehensive_comparison(
        compare_results,
        save_dir=str(FIGS_DIR))
    csv_path = RESULTS_DIR / 'trajectory_output.csv'
    np.savetxt(str(csv_path), fused_traj, delimiter=',',
               header='x,y,theta', comments='')
    print(f'[save] trajectory -> {csv_path}')
    save_run_results({k: v.get('metrics', {}) if isinstance(v, dict) and 'metrics' in v else {}
                       for k, v in compare_results.items()})
    print(f'[Phase6] done in {time.perf_counter() - t0:.2f}s')

    elapsed = time.perf_counter() - t_start
    print(f'=' * 70)
    print(f'SLAM pipeline complete in {elapsed:.2f}s')
    print(f'Output: {FIGS_DIR}/  {RESULTS_DIR}/')
    print(f'=' * 70)


def _compute_metrics_simple(traj_est, traj_true):
    """
    Compute position and heading error metrics between estimated and true trajectories.

    :param traj_est: (np.ndarray) Estimated trajectory, shape (N, 3)
    :param traj_true: (np.ndarray) Ground truth trajectory, shape (N, 3)
    :return: (dict) Metrics: rmse_pos, max_pos, rmse_theta, mean_pos
    """
    n = min(len(traj_est), len(traj_true))
    pos_err = np.sqrt((traj_est[:n, 0] - traj_true[:n, 0])**2 +
                       (traj_est[:n, 1] - traj_true[:n, 1])**2)
    theta_err = np.abs(angle_mod(traj_est[:n, 2] - traj_true[:n, 2]))
    return {
        'pos_rmse': float(np.sqrt(np.mean(pos_err**2))),
        'max_pos': float(np.max(pos_err)),
        'heading_rmse': float(np.sqrt(np.mean(theta_err**2))),
        'mean_pos': float(np.mean(pos_err)),
        'ate': float(np.sqrt(np.mean(pos_err**2))),
        'step_time_ms': 0.0}


if __name__ == '__main__':
    main()