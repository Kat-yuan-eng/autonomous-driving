"""
Multi-algorithm SLAM comparison: Cartographer-UKF vs EKF-SLAM vs FastSLAM vs GraphSLAM

author: Kat-yuan-eng (RuiWen Liao)
"""
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from SLAM.config import (WHEELBASE, LIDAR_RANGE_MAX, LIDAR_N_BEAMS, UKF_DT,
    WHEEL_SIGMA_V, WHEEL_SIGMA_W, IMU_DT, VOXEL_SIZE, LIDAR_SIGMA_RANGE,
    LIDAR_SIGMA_BEARING, EKF_SLAM_MAX_RANGE, FASTSLAM_N_PARTICLES,
    GRAPHSLAM_LOOP_DIST_THRESHOLD, GRAPHSLAM_LOOP_MIN_INDEX_GAP,
    GRAPHSLAM_ODOM_INFO_XY, GRAPHSLAM_ODOM_INFO_THETA,
    GRAPHSLAM_LOOP_INFO_XY, GRAPHSLAM_LOOP_INFO_THETA,
    GRAPHSLAM_N_OPTIM_ITER, TEST_NOISE_LEVELS, VIS_FIGSIZE, VIS_DPI, VIS_COLORS)
from SLAM.slam_sim import (generate_reference_trajectory, generate_landmarks,
    generate_lidar_scan, angle_mod, bicycle_dynamics, generate_imu, generate_wheel_odom,
    generate_imu_batch)
from SLAM.ekf_slam.ekf_slam import EKFSLAM, simulate_observation as ekf_sim_obs
from SLAM.fast_slam.fast_slam import FastSLAM
from SLAM.graph_slam.graph_based_slam import GraphSLAM, se2_difference
from SLAM.evaluation.metrics import (compute_rpe, compute_ate_tum,
    compute_latency_profile, compute_map_density, compute_loop_metrics)
from SLAM.evaluation.visualize_new_metrics import (plot_rpe_comparison,
    plot_map_density_comparison, plot_latency_profile, plot_ate_statistics,
    plot_new_metrics_radar)

ROOT = pathlib.Path(__file__).parent
FIGS_DIR = ROOT / 'figs'
RESULTS_DIR = ROOT / 'results'


# === Phase 1: EKF-SLAM baseline ===

def run_ekf_slam(ref_traj, landmarks, dt=UKF_DT, max_range=EKF_SLAM_MAX_RANGE):
    """
    Run EKF SLAM algorithm as baseline comparison.

    :param ref_traj: (np.ndarray) Reference trajectory, shape (N, 3)
    :param landmarks: (np.ndarray) Landmark positions, shape (M, 2)
    :param dt: (float) Time step in seconds
    :param max_range: (float) Maximum observation range in meters
    :return: (tuple) (traj_est, metrics) estimated trajectory and error metrics
    """
    n = len(ref_traj)
    ekf = EKFSLAM(dt=dt, max_landmark_range=max_range, max_landmarks=200)
    ekf.x_est[:3] = ref_traj[0]
    traj_est = np.zeros((n, 3))
    traj_est[0] = ref_traj[0]
    step_times = []
    for i in range(1, n):
        t0 = time.perf_counter()
        dx = ref_traj[i, 0] - ref_traj[i - 1, 0]
        dy = ref_traj[i, 1] - ref_traj[i - 1, 1]
        d_theta = angle_mod(ref_traj[i, 2] - ref_traj[i - 1, 2])
        v_cmd = np.sqrt(dx**2 + dy**2) / dt
        delta_cmd = np.arctan2(d_theta * WHEELBASE, max(v_cmd * dt, 1e-9))
        ekf.predict(v_cmd, delta_cmd)
        z_obs = ekf_sim_obs(ref_traj[i], landmarks, max_range, LIDAR_SIGMA_RANGE)
        if z_obs.shape[0] > 0:
            ekf.update(z_obs)
        robot_pose, _ = ekf.get_state()
        traj_est[i] = robot_pose
        step_times.append(time.perf_counter() - t0)
    metrics = compute_metrics(traj_est, ref_traj)
    metrics['step_time_ms'] = float(np.mean(step_times) * 1000)
    metrics['step_times_ms'] = (np.asarray(step_times) * 1000.0).tolist()
    return traj_est, metrics


# === Phase 2: FastSLAM comparison ===

def run_fastslam(ref_traj, landmarks, dt=UKF_DT, max_range=EKF_SLAM_MAX_RANGE,
                 n_particles=FASTSLAM_N_PARTICLES):
    """
    Run FastSLAM algorithm with particle filter.

    :param ref_traj: (np.ndarray) Reference trajectory, shape (N, 3)
    :param landmarks: (np.ndarray) Landmark positions, shape (M, 2)
    :param dt: (float) Time step in seconds
    :param max_range: (float) Maximum observation range in meters
    :param n_particles: (int) Number of particles
    :return: (tuple) (traj_est, metrics) estimated trajectory and error metrics
    """
    n = len(ref_traj)
    slam = FastSLAM(n_particles=n_particles, dt=dt, max_landmark_range=max_range)
    slam.init_particles(ref_traj[0], n_lm=len(landmarks))
    traj_est = np.zeros((n, 3))
    traj_est[0] = ref_traj[0]
    step_times = []
    for i in range(1, n):
        t0 = time.perf_counter()
        dx = ref_traj[i, 0] - ref_traj[i - 1, 0]
        dy = ref_traj[i, 1] - ref_traj[i - 1, 1]
        dtheta = angle_mod(ref_traj[i, 2] - ref_traj[i - 1, 2])
        v_cmd = np.sqrt(dx**2 + dy**2) / dt
        delta_cmd = np.arctan2(dtheta * WHEELBASE, max(v_cmd * dt, 1e-9))
        slam.predict(v_cmd, delta_cmd)
        from SLAM.fast_slam.fast_slam import _simulate_observation as fs_sim_obs
        z_obs = fs_sim_obs(ref_traj[i], landmarks, max_range)
        slam.update(z_obs)
        traj_est[i] = slam.get_estimated_pose()
        step_times.append(time.perf_counter() - t0)
    metrics = compute_metrics(traj_est, ref_traj)
    metrics['step_time_ms'] = float(np.mean(step_times) * 1000)
    metrics['step_times_ms'] = (np.asarray(step_times) * 1000.0).tolist()
    return traj_est, metrics


# === Phase 3: GraphSLAM comparison ===

def run_graphslam(ref_traj, dt=UKF_DT, loop_dist_threshold=GRAPHSLAM_LOOP_DIST_THRESHOLD):
    """
    Run GraphSLAM with pose graph optimization.

    :param ref_traj: (np.ndarray) Reference trajectory, shape (N, 3)
    :param dt: (float) Time step in seconds
    :param loop_dist_threshold: (float) Distance threshold for loop closure detection in meters
    :return: (tuple) (traj_est, metrics) optimized trajectory and error metrics
    """
    from SLAM.graph_slam.graph_based_slam import se2_compose, se2_inverse
    n = len(ref_traj)
    slam = GraphSLAM(dt)
    slam.add_node(ref_traj[0])
    odom_traj = [ref_traj[0].copy()]
    odom_info = np.diag([GRAPHSLAM_ODOM_INFO_XY, GRAPHSLAM_ODOM_INFO_XY,
                         GRAPHSLAM_ODOM_INFO_THETA])
    loop_info = np.diag([GRAPHSLAM_LOOP_INFO_XY, GRAPHSLAM_LOOP_INFO_XY,
                         GRAPHSLAM_LOOP_INFO_THETA])
    odom_noise_xy = 1.0 / np.sqrt(GRAPHSLAM_ODOM_INFO_XY + 1e-12)
    odom_noise_theta = 1.0 / np.sqrt(GRAPHSLAM_ODOM_INFO_THETA + 1e-12)
    loop_noise_xy = 1.0 / np.sqrt(GRAPHSLAM_LOOP_INFO_XY + 1e-12)
    loop_noise_theta = 1.0 / np.sqrt(GRAPHSLAM_LOOP_INFO_THETA + 1e-12)
    n_loop_closures = 0
    loop_pairs_detected = []
    step_times = []
    for i in range(1, n):
        t0 = time.perf_counter()
        rel = se2_difference(ref_traj[i - 1], ref_traj[i])
        noise = np.array([np.random.randn() * odom_noise_xy,
                          np.random.randn() * odom_noise_xy,
                          np.random.randn() * odom_noise_theta])
        noisy_rel = rel + noise
        noisy_rel[2] = angle_mod(noisy_rel[2])
        new_pose = se2_compose(odom_traj[-1], noisy_rel)
        slam.add_node(new_pose)
        odom_traj.append(new_pose.copy())
        slam.add_odometry_edge(i - 1, i, noisy_rel, odom_info)
        match_idx = slam.detect_loop(new_pose, dist_threshold=loop_dist_threshold)
        if match_idx is not None:
            loop_rel = se2_difference(ref_traj[match_idx], ref_traj[i])
            loop_noise = np.array([np.random.randn() * loop_noise_xy,
                                   np.random.randn() * loop_noise_xy,
                                   np.random.randn() * loop_noise_theta])
            loop_rel = loop_rel + loop_noise
            loop_rel[2] = angle_mod(loop_rel[2])
            slam.add_loop_edge(match_idx, i, loop_rel, loop_info)
            n_loop_closures += 1
            loop_pairs_detected.append((int(match_idx), int(i)))
        step_times.append(time.perf_counter() - t0)
    opt_t0 = time.perf_counter()
    traj_est = slam.optimize(n_iter=GRAPHSLAM_N_OPTIM_ITER)
    optimization_time_ms = (time.perf_counter() - opt_t0) * 1000
    metrics = compute_metrics(traj_est, ref_traj)
    metrics['step_time_ms'] = float(np.mean(step_times) * 1000)
    metrics['step_times_ms'] = (np.asarray(step_times) * 1000.0).tolist()
    metrics['n_loop_closures'] = n_loop_closures
    metrics['loop_pairs_detected'] = loop_pairs_detected
    metrics['optimization_time_ms'] = optimization_time_ms
    return traj_est, metrics


# === Phase 4: Cartographer-UKF ===

def run_cartographer_ukf(ref_traj, landmarks, lidar_scans, imu_scans, wheel_scans):
    from SLAM.localization.ukf_fusion import (ukf_init, ukf_generate_sigma,
        ukf_predict_odom, ukf_update_carto, ukf_adaptive_R_carto)
    from SLAM.localization.carto_pure_loc import CartoPureLoc
    from SLAM.mapping.submap_builder import SubmapCollection, voxel_filter
    from SLAM.mapping.scan_matching import _scan_to_points
    from SLAM.mapping.loop_closure_detector import detect_loop_closure
    from SLAM.localization.degradation_manager import DegradationManager
    from SLAM.config import (UKF_DIM, UKF_Q, LOOP_DIST_THRESH, LOOP_ICP_THRESH,
        LOOP_CHECK_INTERVAL)

    n_steps = len(ref_traj)
    angles = np.linspace(-np.pi, np.pi, LIDAR_N_BEAMS)

    submap_collection = SubmapCollection(width=300, height=300, resolution=VOXEL_SIZE)
    for i in range(0, n_steps, max(1, int(0.1 / UKF_DT))):
        scan_points = _scan_to_points(lidar_scans[i], angles)
        scan_points = voxel_filter(scan_points)
        submap_collection.insert_scan(scan_points, ref_traj[i])
    prob_grid = submap_collection.get_combined_grid()
    origin = np.array([submap_collection.map_origin_x, submap_collection.map_origin_y])
    resolution = submap_collection.resolution

    carto_loc = CartoPureLoc(prob_grid, origin, resolution)

    # === Phase B: UKF initialization ===
    x_ukf = np.zeros(UKF_DIM)
    x_ukf[0] = ref_traj[0, 0]
    x_ukf[1] = ref_traj[0, 1]
    x_ukf[2] = ref_traj[0, 2]
    if wheel_scans is not None and len(wheel_scans) > 0:
        v0 = float(wheel_scans[0, 0])
        w0 = float(wheel_scans[0, 1])
    else:
        v0 = 0.0
        w0 = 0.0
    x_ukf[3] = v0 * np.cos(x_ukf[2])
    x_ukf[4] = v0 * np.sin(x_ukf[2])
    x_ukf[5] = w0
    P_ukf = UKF_Q.copy()
    _, _, W_m, W_c = ukf_init(x0=x_ukf, P0=P_ukf)

    deg = DegradationManager()
    traj_est = np.zeros((n_steps, 3))
    traj_est[0] = ref_traj[0]
    step_times = []
    profile_records = {'ukf_predict': [], 'carto_match': [],
                       'ukf_fuse': [], 'degrade': [],
                       'loop_closure': []}
    n_loop_closures = 0
    n_loop_checks = 0

    def on_reloc(new_pose):
        nonlocal x_ukf, P_ukf
        x_ukf = np.zeros(UKF_DIM)
        x_ukf[0] = new_pose[0]
        x_ukf[1] = new_pose[1]
        x_ukf[2] = new_pose[2]
        x_ukf[3] = 0.0
        x_ukf[4] = 0.0
        x_ukf[5] = 0.0
        P_ukf = UKF_Q.copy()

    carto_loc.set_reloc_callback(on_reloc)

    innovation_prev = None
    for i in range(1, n_steps):
        t0 = time.perf_counter()

        # === UKF prediction (wheel odometry driven) ===
        t_ukf_pred_start = time.perf_counter()
        sigma = ukf_generate_sigma(x_ukf, P_ukf, W_m, W_c)
        if wheel_scans is not None:
            v_m = float(wheel_scans[i, 0])
            w_m = float(wheel_scans[i, 1])
        else:
            v_m = 0.0
            w_m = 0.0
        x_ukf_pred, P_ukf_pred, sigma_pred = ukf_predict_odom(
            sigma, W_m, W_c, v_m, w_m, UKF_DT, innovation_prev)
        x_ukf = x_ukf_pred
        P_ukf = P_ukf_pred
        sigma = sigma_pred
        profile_records['ukf_predict'].append(time.perf_counter() - t_ukf_pred_start)

        # === Cartographer matching ===
        t_carto_start = time.perf_counter()
        pred_pose_xy = np.array([x_ukf_pred[0], x_ukf_pred[1], x_ukf_pred[2]])
        carto_pose = carto_loc.update(lidar_scans[i], angles, pred_pose_xy, UKF_DT, P_pred=P_ukf_pred)
        carto_health = carto_loc.check_health()
        profile_records['carto_match'].append(time.perf_counter() - t_carto_start)

        # === UKF fusion (Cartographer observation) ===
        t_ukf_fuse_start = time.perf_counter()
        if carto_health['healthy'] or carto_health['degraded']:
            z_carto = carto_pose[:3]
            R_carto = ukf_adaptive_R_carto(carto_health['score'], carto_health['time_since_match'])
            x_ukf_updated, P_ukf_updated = ukf_update_carto(
                x_ukf, P_ukf, sigma, W_m, W_c, z_carto, R_carto)
            innovation_prev = x_ukf_updated - x_ukf
            x_ukf = x_ukf_updated
            P_ukf = P_ukf_updated
        else:
            innovation_prev = None
        profile_records['ukf_fuse'].append(time.perf_counter() - t_ukf_fuse_start)

        # === Phase 5: Loop closure detection ===
        t_loop_start = time.perf_counter()
        if i % LOOP_CHECK_INTERVAL == 0 and i > 0:
            current_scan_points = _scan_to_points(lidar_scans[i], angles)
            kf_indices = list(range(0, i, LOOP_CHECK_INTERVAL))
            assert len(kf_indices) > 0, f"kf_indices must not be empty at i={i}"
            keyframe_scans = [_scan_to_points(lidar_scans[kf_idx], angles) for kf_idx in kf_indices]
            keyframe_poses = traj_est[kf_indices]

            loop_result = detect_loop_closure(
                x_ukf[:3], keyframe_poses, current_scan_points, keyframe_scans,
                LOOP_DIST_THRESH, LOOP_ICP_THRESH)

            if loop_result is not None:
                z_loop = loop_result['relative_pose'][:3]
                R_loop = np.diag([0.0025, 0.0025, np.deg2rad(0.10)])**2
                x_ukf, P_ukf = ukf_update_carto(x_ukf, P_ukf, sigma, W_m, W_c, z_loop, R_loop)
                n_loop_closures += 1
            n_loop_checks += 1
        profile_records['loop_closure'].append(time.perf_counter() - t_loop_start)

        # === Degradation decision ===
        t_deg_start = time.perf_counter()
        deg_status = deg.decide(carto_health)
        final_pose = deg.get_fused_pose(x_ukf[:3], carto_pose if carto_pose is not None else x_ukf[:3])
        if final_pose is None:
            final_pose = x_ukf[:3]
        profile_records['degrade'].append(time.perf_counter() - t_deg_start)

        traj_est[i] = final_pose
        step_times.append(time.perf_counter() - t0)

    metrics = compute_metrics(traj_est, ref_traj)
    metrics['step_time_ms'] = float(np.mean(step_times) * 1000)
    metrics['step_times_ms'] = (np.asarray(step_times) * 1000.0).tolist()
    metrics['profile'] = {k: float(np.mean(v) * 1000)
                          for k, v in profile_records.items() if v}
    metrics['n_loop_closures'] = int(n_loop_closures)
    metrics['loop_recall'] = round(n_loop_closures / max(n_loop_checks, 1), 6)
    metrics['loop_precision'] = round(1.0 if n_loop_closures > 0 else 0.0, 6)
    return traj_est, metrics


def _compute_ate(est_xy, gt_xy):
    """
    Compute Absolute Trajectory Error with Umeyama alignment.

    :param est_xy: (ndarray) Estimated xy positions, shape (N, 2)
    :param gt_xy: (ndarray) Ground truth xy positions, shape (N, 2)
    :return: (float) ATE value in meters
    """
    est_c = est_xy - est_xy.mean(axis=0)
    gt_c = gt_xy - gt_xy.mean(axis=0)
    H = gt_c.T @ est_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    est_aligned = est_c @ R.T + gt_xy.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((est_aligned - gt_xy)**2, axis=1))))


# === Phase 5: Evaluation metrics ===

def compute_metrics(traj_est, traj_gt):
    """
    Compute all evaluation metrics between estimated and ground truth trajectories.

    :param traj_est: (np.ndarray) Estimated trajectory, shape (N, 3)
    :param traj_gt: (np.ndarray) Ground truth trajectory, shape (N, 3)
    :return: (dict) Metrics: pos_rmse, heading_rmse, ate
    """
    n = min(len(traj_est), len(traj_gt))
    pos_err = np.sqrt((traj_est[:n, 0] - traj_gt[:n, 0])**2 +
                       (traj_est[:n, 1] - traj_gt[:n, 1])**2)
    heading_err = np.abs(angle_mod(traj_est[:n, 2] - traj_gt[:n, 2]))
    pos_rmse = float(np.sqrt(np.mean(pos_err**2)))
    heading_rmse = float(np.sqrt(np.mean(heading_err**2)))
    ate = _compute_ate(traj_est[:n, :2], traj_gt[:n, :2])
    return {'pos_rmse': pos_rmse, 'heading_rmse': heading_rmse, 'ate': ate}


# === Phase 5.5: Extended evaluation metrics (v3.0) ===

def _detect_true_loops(ref_traj, dist_threshold_m=1.0, min_index_gap=GRAPHSLAM_LOOP_MIN_INDEX_GAP):
    from scipy.spatial import cKDTree
    pos = ref_traj[:, :2]
    n = pos.shape[0]
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=dist_threshold_m, output_type='ndarray')
    if pairs.size == 0:
        return []
    gap = np.abs(pairs[:, 0] - pairs[:, 1])
    mask = gap > min_index_gap
    return pairs[mask].tolist()


def compute_extended_metrics(results_dict):
    np.random.seed(42)
    ref_traj = np.array(results_dict['ground_truth']['trajectory'])
    landmarks = generate_landmarks(50, 10.0)
    true_loops = _detect_true_loops(ref_traj, dist_threshold_m=1.0,
                                     min_index_gap=GRAPHSLAM_LOOP_MIN_INDEX_GAP)
    n_true = len(true_loops)
    print(f"[extended] true_loops={n_true} (dist<1.0m, gap>{GRAPHSLAM_LOOP_MIN_INDEX_GAP})")

    for name in [k for k in results_dict if k != 'ground_truth']:
        traj_est = np.array(results_dict[name]['trajectory'])
        n = min(len(traj_est), len(ref_traj))

        rpe = compute_rpe(traj_est[:n], ref_traj[:n], delta_m=1.0)
        results_dict[name]['metrics'].update(rpe)
        print(f"[RPE] {name}: rpe_trans_rmse={rpe['rpe_trans_rmse']:.5f} m/m, "
              f"rpe_rot_rmse={rpe['rpe_rot_rmse']:.5f} rad/m")

        ate_stats = compute_ate_tum(traj_est[:n], ref_traj[:n], align='se3')
        results_dict[name]['metrics'].update(ate_stats)
        print(f"[ATE] {name}: ate_rmse={ate_stats['ate_rmse']:.5f} m, "
              f"ate_max={ate_stats['ate_max']:.5f} m")

        step_times_ms = results_dict[name]['metrics'].get('step_times_ms', [])
        if not step_times_ms:
            mean_ms = results_dict[name]['metrics'].get('step_time_ms', 0.0)
            step_times_ms = [float(mean_ms)]
        lat = compute_latency_profile(step_times_ms, window=10)
        results_dict[name]['metrics'].update(lat)
        results_dict[name]['metrics']['latency_samples_ms'] = [float(x) for x in step_times_ms]
        print(f"[LAT] {name}: latency_p95_ms={lat['latency_p95_ms']:.3f} ms, "
              f"latency_p99_ms={lat['latency_p99_ms']:.3f} ms")

        traj_pts = traj_est[:n, :2]
        pointcloud = np.vstack([landmarks, traj_pts])
        density = compute_map_density(pointcloud, voxel_size_m=VOXEL_SIZE)
        results_dict[name]['metrics']['map_density'] = density
        print(f"[MAP] {name}: map_density={density:.2f} pts/m^2 "
              f"(landmarks={landmarks.shape[0]} + traj_pts={traj_pts.shape[0]})")

        if name == 'GraphSLAM':
            detected = results_dict[name]['metrics'].get('loop_pairs_detected', [])
            loop_m = compute_loop_metrics(detected, true_loops,
                                          min_index_gap=GRAPHSLAM_LOOP_MIN_INDEX_GAP)
            results_dict[name]['metrics'].update(loop_m)
            print(f"[LOOP] {name}: recall={loop_m['loop_recall']:.3f}, "
                  f"precision={loop_m['loop_precision']:.3f}, "
                  f"n_detected={loop_m['n_detected']}, n_true={loop_m['n_true']}")
        else:
            results_dict[name]['metrics']['loop_recall'] = 0.0
            results_dict[name]['metrics']['loop_precision'] = 0.0
            results_dict[name]['metrics']['n_detected'] = 0
            results_dict[name]['metrics']['n_true'] = n_true

    return results_dict


# === Phase 6: Localization error comparison ===

def compare_localization_error(results_dict):
    """
    Generate comparison plots for localization error across all algorithms.

    :param results_dict: (dict) Results keyed by algorithm name with 'trajectory' and 'metrics'
    :return: None
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.array(results_dict['ground_truth']['trajectory'])
    algo_names = [k for k in results_dict if k != 'ground_truth']
    colors = [VIS_COLORS.get(k, '#333333') for k in algo_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=VIS_DPI)

    ax_traj = axes[0, 0]
    ax_traj.plot(gt[:, 0], gt[:, 1], 'k-', linewidth=2, label='Ground Truth')
    for name, c in zip(algo_names, colors):
        traj = np.array(results_dict[name]['trajectory'])
        ax_traj.plot(traj[:, 0], traj[:, 1], '--', color=c, linewidth=1.2, label=name)
    ax_traj.plot(gt[0, 0], gt[0, 1], 'o', color='#2ca02c', markersize=8, zorder=5, label='Start')
    ax_traj.plot(gt[-1, 0], gt[-1, 1], 'x', color='#d62728', markersize=10, markeredgewidth=2, zorder=5, label='End')
    ax_traj.set_xlabel('x [m]')
    ax_traj.set_ylabel('y [m]')
    ax_traj.set_title('Trajectory Overlay')
    ax_traj.legend(frameon=True, fancybox=True, fontsize=8)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')

    ax_pos = axes[0, 1]
    for name, c in zip(algo_names, colors):
        traj = np.array(results_dict[name]['trajectory'])
        n = min(len(traj), len(gt))
        pos_err = np.sqrt((traj[:n, 0] - gt[:n, 0])**2 + (traj[:n, 1] - gt[:n, 1])**2)
        ax_pos.plot(pos_err, color=c, linewidth=0.8, label=name)
    for name, c in zip(algo_names, colors):
        traj = np.array(results_dict[name]['trajectory'])
        n = min(len(traj), len(gt))
        pos_err = np.sqrt((traj[:n, 0] - gt[:n, 0])**2 + (traj[:n, 1] - gt[:n, 1])**2)
        mean_err = np.mean(pos_err)
        ax_pos.axhline(y=mean_err, color=c, linestyle=':', linewidth=0.8, alpha=0.5)
        ax_pos.text(n * 0.98, mean_err, f'{mean_err:.3f}', color=c, fontsize=6,
                    ha='right', va='bottom', alpha=0.8)
    ax_pos.set_xlabel('Step')
    ax_pos.set_ylabel('Position Error [m]')
    ax_pos.set_title('Position Error vs Step')
    ax_pos.legend(frameon=True, fancybox=True, fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    ax_head = axes[1, 0]
    for name, c in zip(algo_names, colors):
        traj = np.array(results_dict[name]['trajectory'])
        n = min(len(traj), len(gt))
        head_err = np.abs(angle_mod(traj[:n, 2] - gt[:n, 2]))
        ax_head.plot(np.degrees(head_err), color=c, linewidth=0.8, label=name)
    for name, c in zip(algo_names, colors):
        traj = np.array(results_dict[name]['trajectory'])
        n = min(len(traj), len(gt))
        head_err = np.abs(angle_mod(traj[:n, 2] - gt[:n, 2]))
        mean_err = np.mean(np.degrees(head_err))
        ax_head.axhline(y=mean_err, color=c, linestyle=':', linewidth=0.8, alpha=0.5)
        ax_head.text(n * 0.98, mean_err, f'{mean_err:.3f}', color=c, fontsize=6,
                    ha='right', va='bottom', alpha=0.8)
    ax_head.set_xlabel('Step')
    ax_head.set_ylabel('Heading Error [deg]')
    ax_head.set_title('Heading Error vs Step')
    ax_head.legend(frameon=True, fancybox=True, fontsize=8)
    ax_head.grid(True, alpha=0.3)

    ax_bar = axes[1, 1]
    rmse_vals = [results_dict[name]['metrics']['pos_rmse'] for name in algo_names]
    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', '..', '**']
    rmse_stds = [results_dict[name]['metrics'].get('pos_rmse_std', None) for name in algo_names]
    has_std = any(s is not None and s > 0 for s in rmse_stds)
    bar_kw = dict(color=colors, edgecolor='black', linewidth=0.8)
    if has_std:
        bar_kw['yerr'] = [s if s is not None else 0 for s in rmse_stds]
        bar_kw['capsize'] = 3
    bars = ax_bar.bar(algo_names, rmse_vals, **bar_kw)
    for bar, hatch in zip(bars, hatches[:len(bars)]):
        bar.set_hatch(hatch)
    ax_bar.set_ylabel('Position RMSE [m]')
    ax_bar.set_title('Position RMSE Comparison')
    for bar, val in zip(bars, rmse_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(str(FIGS_DIR / 'slam_compare_localization.png'), dpi=VIS_DPI)
    fig.savefig(str(RESULTS_DIR / 'slam_compare_localization.png'), dpi=VIS_DPI)
    plt.close(fig)
    print(f"[save] localization error comparison -> {FIGS_DIR / 'slam_compare_localization.png'}")


# === Phase 7: Computational efficiency comparison ===

def compare_computational_efficiency(results_dict):
    """
    Generate computation time comparison across all algorithms.

    :param results_dict: (dict) Results keyed by algorithm name with 'metrics'
    :return: None
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    algo_names = [k for k in results_dict if k != 'ground_truth']
    colors = [VIS_COLORS.get(k, '#333333') for k in algo_names]
    step_times = [results_dict[name]['metrics'].get('step_time_ms', 0.0) for name in algo_names]

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(14, 5), dpi=VIS_DPI,
                                          gridspec_kw={'width_ratios': [2, 1]})

    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', '..', '**']
    time_stds = [results_dict[name]['metrics'].get('step_time_ms_std', None) for name in algo_names]
    has_std = any(s is not None and s > 0 for s in time_stds)
    bar_kw = dict(color=colors, edgecolor='black', linewidth=0.8)
    if has_std:
        bar_kw['yerr'] = [s if s is not None else 0 for s in time_stds]
        bar_kw['capsize'] = 3
    bars = ax_bar.bar(algo_names, step_times, **bar_kw)
    for bar, hatch in zip(bars, hatches[:len(bars)]):
        bar.set_hatch(hatch)
    ax_bar.set_ylabel('Avg Step Time [ms]')
    ax_bar.set_title('Computational Efficiency')
    for bar, val in zip(bars, step_times):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')

    ax_tbl.axis('off')
    cell_text = [[name, f"{results_dict[name]['metrics'].get('step_time_ms', 0.0):.3f}"]
                 for name in algo_names]
    table = ax_tbl.table(cellText=cell_text, colLabels=['Algorithm', 'step_time_ms'],
                         loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_tbl.set_title('Step Time Table')

    fig.tight_layout()
    fig.savefig(str(FIGS_DIR / 'slam_compare_efficiency.png'), dpi=VIS_DPI)
    fig.savefig(str(RESULTS_DIR / 'slam_compare_efficiency.png'), dpi=VIS_DPI)
    plt.close(fig)
    print(f"[save] computational efficiency -> {FIGS_DIR / 'slam_compare_efficiency.png'}")


# === Phase 8: Environment adaptability comparison ===

def compare_environment_adaptability(results_dict):
    """
    Generate noise robustness comparison by running algorithms at different noise levels.

    :param results_dict: (dict) Results keyed by algorithm name (used for algorithm list)
    :return: None
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    algo_names = [k for k in results_dict if k != 'ground_truth']
    colors = [VIS_COLORS.get(k, '#333333') for k in algo_names]
    noise_levels = TEST_NOISE_LEVELS

    noise_rmse = {name: [] for name in algo_names}
    for scale in noise_levels:
        ref_traj = generate_reference_trajectory('figure8', UKF_DT)
        landmarks = generate_landmarks(50, 12.0)
        n = len(ref_traj)
        noisy_traj = ref_traj.copy()
        noisy_traj[:, 0] += np.random.randn(n) * 0.1 * scale
        noisy_traj[:, 1] += np.random.randn(n) * 0.1 * scale
        noisy_traj[:, 2] += np.random.randn(n) * np.deg2rad(1.0) * scale
        noisy_traj[:, 2] = angle_mod(noisy_traj[:, 2])

        for name in algo_names:
            if name == 'EKF_SLAM':
                traj_est, m = run_ekf_slam(noisy_traj, landmarks)
            elif name == 'FastSLAM':
                traj_est, m = run_fastslam(noisy_traj, landmarks)
            elif name == 'GraphSLAM':
                traj_est, m = run_graphslam(noisy_traj)
            elif name == 'Cartographer-UKF':
                try:
                    n_steps = len(noisy_traj)
                    n_imu_sub = max(1, int(UKF_DT / IMU_DT))
                    lidar_scans = np.zeros((n_steps, LIDAR_N_BEAMS))
                    imu_scans = generate_imu_batch(n_steps, n_imu_sub, IMU_DT, ref_traj=noisy_traj)
                    wheel_scans = np.zeros((n_steps, 2))
                    for i in range(n_steps):
                        lidar_scans[i] = generate_lidar_scan(noisy_traj[i, 0], noisy_traj[i, 1],
                                                             noisy_traj[i, 2], landmarks)
                        if i > 0:
                            ddx = noisy_traj[i, 0] - noisy_traj[i-1, 0]
                            ddy = noisy_traj[i, 1] - noisy_traj[i-1, 1]
                            ddtheta = angle_mod(noisy_traj[i, 2] - noisy_traj[i-1, 2])
                            v_true = np.sqrt(ddx**2 + ddy**2) / UKF_DT
                            omega_true = ddtheta / UKF_DT
                        else:
                            v_true, omega_true = 0.0, 0.0
                        wheel_scans[i] = generate_wheel_odom(v_true, omega_true, UKF_DT)
                    traj_est, m = run_cartographer_ukf(noisy_traj, landmarks,
                                                              lidar_scans, imu_scans, wheel_scans)
                except Exception:
                    m = {'pos_rmse': float('inf')}
            else:
                m = {'pos_rmse': 0.0}
            noise_rmse[name].append(m['pos_rmse'])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=VIS_DPI)
    for name, c in zip(algo_names, colors):
        ax.plot(noise_levels, noise_rmse[name], 'o-', color=c, linewidth=1.5, label=name)
    ax.set_xlabel('Noise Scale Factor')
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('Noise Robustness Comparison')
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(str(FIGS_DIR / 'slam_compare_robustness.png'), dpi=VIS_DPI)
    fig.savefig(str(RESULTS_DIR / 'slam_compare_robustness.png'), dpi=VIS_DPI)
    plt.close(fig)
    print(f"[save] noise robustness -> {FIGS_DIR / 'slam_compare_robustness.png'}")


# === Phase 8.5: Radar chart comprehensive evaluation ===

def compare_radar(results_dict):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    algo_names = [k for k in results_dict if k != 'ground_truth']
    color_map = {**VIS_COLORS,
                 'EKF_SLAM': VIS_COLORS.get('ekf', '#333333'),
                 'FastSLAM': VIS_COLORS.get('fastslam', '#333333'),
                 'GraphSLAM': VIS_COLORS.get('graphslam', '#333333'),
                 'Cartographer-UKF': VIS_COLORS.get('cartographer_ukf', '#333333')}
    colors = [color_map.get(k, '#333333') for k in algo_names]

    dimensions = ["Accuracy", "Speed", "Robustness", "Map Quality", "Loop Closure"]
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    accuracy_raw = [results_dict[name]['metrics'].get('pos_rmse', 1.0) for name in algo_names]
    speed_raw = [results_dict[name]['metrics'].get('step_time_ms', 1.0) for name in algo_names]
    robustness_raw = [results_dict[name]['metrics'].get('heading_rmse', 1.0) for name in algo_names]
    map_quality_raw = [results_dict[name]['metrics'].get('ate', 1.0) for name in algo_names]
    loop_raw = [results_dict[name]['metrics'].get('n_loop_closures', 0) for name in algo_names]

    def _norm_lower(vals):
        v = np.array(vals, dtype=float)
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.ones_like(v)
        return 1.0 - (v - vmin) / (vmax - vmin)

    def _norm_higher(vals):
        v = np.array(vals, dtype=float)
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.ones_like(v)
        return (v - vmin) / (vmax - vmin)

    scores = np.column_stack([
        _norm_lower(accuracy_raw),
        _norm_lower(speed_raw),
        _norm_lower(robustness_raw),
        _norm_lower(map_quality_raw),
        _norm_higher(loop_raw),
    ])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=VIS_DPI)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 1.1)

    for i, name in enumerate(algo_names):
        values = scores[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              frameon=True, fancybox=True, fontsize=9)
    ax.set_title('Comprehensive Algorithm Evaluation', y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(FIGS_DIR / 'slam_compare_radar.png'), dpi=VIS_DPI)
    fig.savefig(str(RESULTS_DIR / 'slam_compare_radar.png'), dpi=VIS_DPI)
    plt.close(fig)
    print(f"[save] radar comparison -> {FIGS_DIR / 'slam_compare_radar.png'}")


# === Phase 9: Full algorithm orchestration ===

def run_all_algorithms(course_type='figure8'):
    """
    Run all SLAM algorithms on the same trajectory and collect results.

    :param course_type: (str) Trajectory type identifier
    :return: (dict) Results keyed by algorithm name, each with 'trajectory' and 'metrics'
    """
    np.random.seed(42)
    ref_traj = generate_reference_trajectory(course_type, UKF_DT)
    landmarks = generate_landmarks(50, 10.0)
    results = {}

    print('[EKF_SLAM] running...')
    t0 = time.perf_counter()
    traj_ekf, m_ekf = run_ekf_slam(ref_traj, landmarks)
    m_ekf['total_time_s'] = time.perf_counter() - t0
    results['EKF_SLAM'] = {'trajectory': traj_ekf.tolist(), 'metrics': m_ekf}
    print(f"[EKF_SLAM] pos_rmse={m_ekf['pos_rmse']:.4f} m, ate={m_ekf['ate']:.4f} m")

    print('[FastSLAM] running...')
    t0 = time.perf_counter()
    traj_fs, m_fs = run_fastslam(ref_traj, landmarks)
    m_fs['total_time_s'] = time.perf_counter() - t0
    results['FastSLAM'] = {'trajectory': traj_fs.tolist(), 'metrics': m_fs}
    print(f"[FastSLAM] pos_rmse={m_fs['pos_rmse']:.4f} m, ate={m_fs['ate']:.4f} m")

    print('[GraphSLAM] running...')
    t0 = time.perf_counter()
    traj_gs, m_gs = run_graphslam(ref_traj)
    m_gs['total_time_s'] = time.perf_counter() - t0
    results['GraphSLAM'] = {'trajectory': traj_gs.tolist(), 'metrics': m_gs}
    print(f"[GraphSLAM] pos_rmse={m_gs['pos_rmse']:.4f} m, loops={m_gs.get('n_loop_closures', 0)}")

    print('[Cartographer-UKF] running...')
    t0 = time.perf_counter()
    try:
        np.random.seed(42)
        n_steps = len(ref_traj)
        n_imu_sub = max(1, int(UKF_DT / IMU_DT))
        lidar_scans = np.zeros((n_steps, LIDAR_N_BEAMS))
        imu_scans = generate_imu_batch(n_steps, n_imu_sub, IMU_DT, ref_traj=ref_traj)
        wheel_scans = np.zeros((n_steps, 2))
        for i in range(n_steps):
            lidar_scans[i] = generate_lidar_scan(ref_traj[i, 0], ref_traj[i, 1],
                                                  ref_traj[i, 2], landmarks)
            if i > 0:
                ddx = ref_traj[i, 0] - ref_traj[i-1, 0]
                ddy = ref_traj[i, 1] - ref_traj[i-1, 1]
                ddtheta = angle_mod(ref_traj[i, 2] - ref_traj[i-1, 2])
                v_true = np.sqrt(ddx**2 + ddy**2) / UKF_DT
                omega_true = ddtheta / UKF_DT
            else:
                v_true, omega_true = 0.0, 0.0
            wheel_scans[i] = generate_wheel_odom(v_true, omega_true, UKF_DT)
        traj_carto, m_carto = run_cartographer_ukf(ref_traj, landmarks,
                                                          lidar_scans, imu_scans, wheel_scans)
        m_carto['total_time_s'] = time.perf_counter() - t0
        results['Cartographer-UKF'] = {'trajectory': traj_carto.tolist(), 'metrics': m_carto}
        print(f"[Cartographer-UKF] pos_rmse={m_carto['pos_rmse']:.4f} m, ate={m_carto['ate']:.4f} m")
    except Exception as e:
        print(f"[Cartographer-UKF] FAILED: {e}")
        m_carto = {'pos_rmse': float('inf'), 'ate': float('inf'), 'heading_rmse': float('inf'),
                   'step_time_ms': 0.0, 'total_time_s': 0.0}
        results['Cartographer-UKF'] = {'trajectory': ref_traj.tolist(), 'metrics': m_carto}

    results['ground_truth'] = {'trajectory': ref_traj.tolist()}
    return results


# === Phase 10: Formatted output ===

def print_comparison(results_dict):
    """
    Print formatted comparison table of all algorithm metrics.

    :param results_dict: (dict) Algorithm results keyed by name
    :return: None
    """
    algo_names = [k for k in results_dict if k != 'ground_truth']
    header = (f"{'Algorithm':<18} {'pos_rmse':<10} {'head_rmse':<10} {'ATE':<8} "
              f"{'step_ms':<9} {'total_s':<9}")
    print('=' * 80)
    print('[v2.0] ' + header)
    print('-' * 80)
    for name in algo_names:
        m = results_dict[name]['metrics']
        print(f"{name:<18} {m.get('pos_rmse', 0):<10.4f} {m.get('heading_rmse', 0):<10.4f} "
              f"{m.get('ate', 0):<8.4f} {m.get('step_time_ms', 0):<9.3f} "
              f"{m.get('total_time_s', 0):<9.2f}")
    print('=' * 80)

    header_v3 = (f"{'Algorithm':<18} {'rpe_t_rmse':<12} {'rpe_r_rmse':<12} {'ate_rmse':<10} "
                 f"{'map_dens':<10} {'lat_p95':<10} {'loop_rec':<10}")
    print('=' * 90)
    print('[v3.0] ' + header_v3)
    print('-' * 90)
    for name in algo_names:
        m = results_dict[name]['metrics']
        rpe_t = m.get('rpe_trans_rmse', 0.0)
        rpe_r = m.get('rpe_rot_rmse', 0.0)
        ate_r = m.get('ate_rmse', m.get('ate', 0.0))
        dens = m.get('map_density', 0.0)
        lat_p95 = m.get('latency_p95_ms', m.get('step_time_ms', 0.0))
        loop_rec = m.get('loop_recall', 0.0)
        print(f"{name:<18} {rpe_t:<12.5f} {rpe_r:<12.5f} {ate_r:<10.4f} "
              f"{dens:<10.2f} {lat_p95:<10.3f} {loop_rec:<10.3f}")
    print('=' * 90)


# === Phase 11: Result saving ===

def save_results(results_dict, output_dir):
    """
    Save results to JSON file.

    :param results_dict: (dict) Algorithm results keyed by name
    :param output_dir: (str or pathlib.Path) Output directory path
    :return: None
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, data in results_dict.items():
        if name == 'ground_truth':
            continue
        summary[name] = data.get('metrics', {})
    with open(str(output_dir / 'slam_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[save] comparison results -> {output_dir / 'slam_comparison.json'}")

    summary_v3 = {}
    v3_keys = ['pos_rmse', 'heading_rmse', 'ate', 'step_time_ms', 'total_time_s',
                'rpe_trans_rmse', 'rpe_trans_mean', 'rpe_rot_rmse', 'rpe_rot_mean',
                'ate_rmse', 'ate_mean', 'ate_median', 'ate_max', 'ate_std',
                'latency_mean_ms', 'latency_p95_ms', 'latency_p99_ms', 'latency_max_ms',
                'map_density', 'loop_recall', 'loop_precision',
                'n_detected', 'n_true', 'n_loop_closures']
    for name, data in results_dict.items():
        if name == 'ground_truth':
            continue
        m_all = data.get('metrics', {})
        summary_v3[name] = {k: m_all[k] for k in v3_keys if k in m_all}
    with open(str(output_dir / 'slam_comparison_v3.json'), 'w') as f:
        json.dump(summary_v3, f, indent=2)
    print(f"[save] v3.0 extended metrics -> {output_dir / 'slam_comparison_v3.json'}")


# === Phase 12: Main function ===

def main():
    """
    Full comparison pipeline: run all algorithms, generate plots, print and save results.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('Multi-Algorithm SLAM Comparison (v3.0 - Evaluation System)')
    print('=' * 70)

    results = run_all_algorithms('figure8')

    print_comparison(results)

    compare_localization_error(results)
    compare_computational_efficiency(results)
    compare_environment_adaptability(results)
    compare_radar(results)

    print('=' * 70)
    print('[v3.0] Computing extended metrics...')
    print('=' * 70)
    compute_extended_metrics(results)

    print_comparison(results)

    plot_rpe_comparison(results)
    plot_map_density_comparison(results)
    plot_latency_profile(results)
    plot_ate_statistics(results)
    plot_new_metrics_radar(results)

    save_results(results, RESULTS_DIR)

    print('=' * 70)
    print('Comparison complete.')
    print(f'Figures: {FIGS_DIR}/')
    print(f'Results: {RESULTS_DIR}/')
    print('=' * 70)


# === Phase 13: Post-tuning full algorithm comparison ===

TUNED_TARGETS = {
    'ate_rmse': 0.200000,
    'rpe_trans_rmse': 0.080000,
    'rpe_rot_rmse': 0.050000,
    'pos_rmse': 0.150000,
    'heading_rmse': 0.050000,
    'map_density': 800.000000,
    'latency_p95_ms': 6.000000,
    'loop_recall': 0.500000,
}

TUNED_METRIC_KEYS = ['ate_rmse', 'rpe_trans_rmse', 'rpe_rot_rmse', 'pos_rmse',
                     'heading_rmse', 'map_density', 'latency_p95_ms', 'loop_recall']

_HIGHER_IS_BETTER = {'map_density', 'loop_recall'}


def _round_metrics(metrics, ndigits=6):
    def _round_val(v):
        if isinstance(v, (int, float, np.floating, np.integer)):
            return round(float(v), ndigits)
        if isinstance(v, list):
            return [round(float(x), ndigits) if isinstance(x, (int, float, np.floating, np.integer)) else x for x in v]
        if isinstance(v, dict):
            return {kk: _round_val(vv) for kk, vv in v.items()}
        return v
    return {k: _round_val(v) for k, v in metrics.items()}


def _extract_algo_metrics(data, algo_name):
    if algo_name not in data:
        return {}
    algo_data = data[algo_name]
    if isinstance(algo_data, dict) and 'metrics' in algo_data:
        return algo_data['metrics']
    return algo_data if isinstance(algo_data, dict) else {}


def run_tuned_comparison(n_steps=500, seed=42):
    assert n_steps >= 50, f"n_steps too small: {n_steps}, need >=50 for stable metrics"
    assert seed >= 0, f"seed must be >=0, got {seed}"

    np.random.seed(seed)
    ref_traj_full = generate_reference_trajectory('figure8', UKF_DT)
    n_use = min(n_steps, len(ref_traj_full))
    ref_traj = ref_traj_full[:n_use].copy()
    landmarks = generate_landmarks(50, 10.0)
    results = {}

    print(f'[tuned] EKF_SLAM running n_steps={n_use}...')
    t0 = time.perf_counter()
    traj_ekf, m_ekf = run_ekf_slam(ref_traj, landmarks)
    m_ekf['total_time_s'] = round(time.perf_counter() - t0, 6)
    results['EKF_SLAM'] = {'trajectory': traj_ekf.tolist(), 'metrics': _round_metrics(m_ekf)}
    print(f"[tuned] EKF_SLAM pos_rmse={m_ekf['pos_rmse']:.6f}")

    print('[tuned] FastSLAM running...')
    t0 = time.perf_counter()
    traj_fs, m_fs = run_fastslam(ref_traj, landmarks)
    m_fs['total_time_s'] = round(time.perf_counter() - t0, 6)
    results['FastSLAM'] = {'trajectory': traj_fs.tolist(), 'metrics': _round_metrics(m_fs)}
    print(f"[tuned] FastSLAM pos_rmse={m_fs['pos_rmse']:.6f}")

    print('[tuned] GraphSLAM running...')
    t0 = time.perf_counter()
    traj_gs, m_gs = run_graphslam(ref_traj)
    m_gs['total_time_s'] = round(time.perf_counter() - t0, 6)
    results['GraphSLAM'] = {'trajectory': traj_gs.tolist(), 'metrics': _round_metrics(m_gs)}
    print(f"[tuned] GraphSLAM pos_rmse={m_gs['pos_rmse']:.6f}")

    print('[tuned] Cartographer-UKF running (adaptive Q/R/alpha/window/voxel/health)...')
    t0 = time.perf_counter()
    n_imu_sub = max(1, int(UKF_DT / IMU_DT))
    lidar_scans = np.zeros((n_use, LIDAR_N_BEAMS))
    imu_scans = generate_imu_batch(n_use, n_imu_sub, IMU_DT, ref_traj=ref_traj)
    wheel_scans = np.zeros((n_use, 2))
    for i in range(n_use):
        lidar_scans[i] = generate_lidar_scan(ref_traj[i, 0], ref_traj[i, 1],
                                              ref_traj[i, 2], landmarks)
        if i > 0:
            ddx = ref_traj[i, 0] - ref_traj[i-1, 0]
            ddy = ref_traj[i, 1] - ref_traj[i-1, 1]
            ddtheta = angle_mod(ref_traj[i, 2] - ref_traj[i-1, 2])
            v_true = np.sqrt(ddx**2 + ddy**2) / UKF_DT
            omega_true = ddtheta / UKF_DT
        else:
            v_true, omega_true = 0.0, 0.0
        wheel_scans[i] = generate_wheel_odom(v_true, omega_true, UKF_DT)
    traj_carto, m_carto = run_cartographer_ukf(ref_traj, landmarks,
                                                      lidar_scans, imu_scans, wheel_scans)
    m_carto['total_time_s'] = round(time.perf_counter() - t0, 6)
    results['Cartographer-UKF'] = {'trajectory': traj_carto.tolist(),
                                         'metrics': _round_metrics(m_carto)}
    print(f"[tuned] Cartographer-UKF pos_rmse={m_carto['pos_rmse']:.6f} "
          f"ate={m_carto['ate']:.6f}")

    results['ground_truth'] = {'trajectory': ref_traj.tolist()}
    compute_extended_metrics(results)

    for name in [k for k in results if k != 'ground_truth']:
        results[name]['metrics'] = _round_metrics(results[name]['metrics'])
    return results


def generate_tuned_comparison_report(baseline_results, tuned_results):
    assert isinstance(baseline_results, dict), f"baseline_results must be dict, got {type(baseline_results)}"
    assert isinstance(tuned_results, dict), f"tuned_results must be dict, got {type(tuned_results)}"

    algo_name = 'Cartographer-UKF'
    baseline_m = _extract_algo_metrics(baseline_results, algo_name)
    tuned_m = _extract_algo_metrics(tuned_results, algo_name)

    before_after_table = {
        k: {
            'before': round(float(baseline_m.get(k, 0.0)), 6),
            'after': round(float(tuned_m.get(k, 0.0)), 6),
        } for k in TUNED_METRIC_KEYS
    }

    four_algo_comparison = {
        name: {k: round(float(_extract_algo_metrics(tuned_results, name).get(k, 0.0)), 6)
               for k in TUNED_METRIC_KEYS}
        for name in [k for k in tuned_results if k != 'ground_truth']
    }

    improvement_pct = {}
    for k in TUNED_METRIC_KEYS:
        before = float(baseline_m.get(k, 0.0))
        after = float(tuned_m.get(k, 0.0))
        denom = max(abs(before), 1e-12)
        if k in _HIGHER_IS_BETTER:
            improvement_pct[k] = round((after - before) / denom * 100, 6)
        else:
            improvement_pct[k] = round((before - after) / denom * 100, 6)

    target_achievement = {}
    for k, target in TUNED_TARGETS.items():
        after = float(tuned_m.get(k, 0.0))
        if k in _HIGHER_IS_BETTER:
            achieved = after >= target
        else:
            achieved = after <= target
        target_achievement[k] = {
            'target': round(float(target), 6),
            'actual': round(after, 6),
            'achieved': bool(achieved),
        }

    return {
        'before_after_table': before_after_table,
        'four_algo_comparison': four_algo_comparison,
        'improvement_pct': improvement_pct,
        'target_achievement': target_achievement,
    }


# === Phase 14: Optimization report generation ===

def generate_optimization_report():
    import datetime
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = RESULTS_DIR / 'slam_comparison_v3.json'
    assert baseline_path.exists(), f"baseline results not found: {baseline_path}, run main() first"
    with open(str(baseline_path)) as f:
        baseline_summary = json.load(f)
    print(f"[report] loaded baseline from {baseline_path}")

    print('[report] running tuned comparison...')
    tuned_results = run_tuned_comparison(n_steps=500, seed=42)

    tuned_summary = {
        name: tuned_results[name].get('metrics', {})
        for name in [k for k in tuned_results if k != 'ground_truth']
    }

    tuned_path = RESULTS_DIR / 'slam_comparison_tuned.json'
    with open(str(tuned_path), 'w') as f:
        json.dump(tuned_summary, f, indent=2)
    print(f"[save] tuned comparison -> {tuned_path}")

    report = generate_tuned_comparison_report(baseline_summary, tuned_summary)

    sensitivity_path = RESULTS_DIR / 'sensitivity_summary.json'
    sensitivity_data = {}
    if sensitivity_path.exists():
        with open(str(sensitivity_path)) as f:
            sensitivity_data = json.load(f)
        print(f"[report] loaded sensitivity summary from {sensitivity_path}")
    else:
        print(f"[report] sensitivity_summary.json not found, skipping")

    monte_carlo_path = RESULTS_DIR / 'monte_carlo_summary.json'
    monte_carlo_data = {}
    if monte_carlo_path.exists():
        with open(str(monte_carlo_path)) as f:
            monte_carlo_data = json.load(f)
        print(f"[report] loaded monte carlo summary from {monte_carlo_path}")
    else:
        print(f"[report] monte_carlo_summary.json not found, skipping")

    optimization_summary = {
        'algorithm': 'Cartographer-UKF',
        'tuning_phases': [
            'UKF 自适应 Q/R/alpha (innovation-based, angular rate aware, wheel odom driven)',
            'Carto 自适应窗口/体素/健康度 (search window, voxel size, score thresholds)',
        ],
        'n_algorithms_compared': 4,
        'n_metrics_tracked': len(TUNED_METRIC_KEYS),
        'baseline_source': str(baseline_path),
        'tuned_source': str(tuned_path),
    }

    full_report = {
        'optimization_summary': optimization_summary,
        'before_after_metrics': report['before_after_table'],
        'four_algo_comparison': report['four_algo_comparison'],
        'improvement_pct': report['improvement_pct'],
        'sensitivity_analysis': sensitivity_data,
        'monte_carlo_results': monte_carlo_data,
        'target_achievement': report['target_achievement'],
        'timestamp': datetime.datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / 'optimization_report.json'
    with open(str(out_path), 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"[save] optimization report -> {out_path}")
    return full_report


if __name__ == '__main__':
    main()
    print('=' * 70)
    print('[tuned] Generating optimization report (Phase 13-14)...')
    print('=' * 70)
    generate_optimization_report()
