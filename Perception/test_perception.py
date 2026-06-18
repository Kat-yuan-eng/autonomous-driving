"""
Perception module test suite: multi-scenario evaluation and visualization

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import time
import pathlib
import sys

from scipy.optimize import linear_sum_assignment

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(BASE / "baselines"))

from config import *
from perception_pipeline import init_pipeline, run_cycle
from baselines.fixed_threshold_filter import init_fixed_pipeline, run_fixed_cycle
from baselines.dbscan_nn_tracker import init_gridnn_pipeline, run_gridnn_cycle
from sdf_filter import compute_sdf, sdf_adaptive_filter, fixed_threshold_filter
from voxel_filter import voxel_filter, transform_to_global
from costmap import inflate_static_layer, gaussian_dynamic_layer, fuse_costmap

FIGS_DIR = BASE / "figs"
RESULTS_DIR = BASE / "results"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "figure.figsize": (10, 6),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

COLORS = {"proposed": "#3A86FF", "fixed": "#FF9E00", "grid_nn": "#6C757D"}
ALG_LABELS = {"proposed": "Proposed (SDF)", "fixed": "Fixed Threshold", "grid_nn": "Grid-NN"}

# === Phase 0: Simulation tools ===

def create_corridor_map(n_row=100, n_col=100):
    """
    Create a corridor-style occupancy map with internal walls.

    :param n_row: (int) Number of grid rows
    :param n_col: (int) Number of grid columns
    :return: (np.ndarray) 2-D binary occupancy grid
    """
    grid = np.zeros((n_row, n_col), dtype=np.float64)
    grid[0, :] = 1.0; grid[-1, :] = 1.0; grid[:, 0] = 1.0; grid[:, -1] = 1.0
    grid[45:55, 30:70] = 1.0
    grid[20:25, 50:80] = 1.0
    grid[70:75, 20:60] = 1.0
    return grid

def simulate_lidar_scan(pose, grid, grid_origin, resolution, obstacles_gt,
                        n_beams=360, max_range=5.0, sigma_rho=0.02):
    """
    Simulate a 2D LiDAR scan with ray-casting against walls and obstacles.

    :param pose: (np.ndarray) Length-3 robot pose [x, y, theta]
    :param grid: (np.ndarray) 2-D binary occupancy grid
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :param resolution: (float) Grid resolution in m/cell
    :param obstacles_gt: (list) List of (x, y, radius) ground-truth obstacles
    :param n_beams: (int) Number of LiDAR beams
    :param max_range: (float) Maximum sensing range in meters
    :param sigma_rho: (float) Range noise standard deviation in meters
    :return: (np.ndarray) Nx2 simulated point cloud in local frame
    """
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    n_steps = 50
    distances = np.linspace(0.1, max_range, n_steps)

    ray_x = distances[None, :] * np.cos(angles[:, None])
    ray_y = distances[None, :] * np.sin(angles[:, None])

    c, s = np.cos(pose[2]), np.sin(pose[2])
    gx = pose[0] + ray_x * c - ray_y * s
    gy = pose[1] + ray_x * s + ray_y * c

    ix = ((gx - grid_origin[0]) / resolution).astype(np.int32)
    iy = ((gy - grid_origin[1]) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
    wall_hit = np.zeros_like(valid)
    wall_hit[valid] = grid[iy[valid], ix[valid]] > 0.5

    obs_hit = np.zeros((n_beams, n_steps), dtype=bool)
    for obs in obstacles_gt:
        d_obs = np.sqrt((gx - obs[0])**2 + (gy - obs[1])**2)
        obs_hit |= (d_obs < obs[2])

    any_hit = wall_hit | obs_hit
    hit_dist_idx = np.argmax(any_hit, axis=1)
    has_hit = np.any(any_hit, axis=1)

    hit_d = distances[hit_dist_idx]
    hit_d_noisy = np.maximum(0.05, hit_d + np.random.normal(0, sigma_rho, n_beams))
    hit_d_noisy[~has_hit] = max_range

    pts_local = np.column_stack([hit_d_noisy * np.cos(angles), hit_d_noisy * np.sin(angles)])

    mask = has_hit & (hit_d_noisy < max_range - 0.1)
    wall_pts = pts_local[mask] if mask.any() else np.zeros((0, 2))

    obs_pts_list = []
    for obs in obstacles_gt:
        dx = obs[0] - pose[0]
        dy = obs[1] - pose[1]
        c_inv, s_inv = np.cos(-pose[2]), np.sin(-pose[2])
        lx = c_inv * dx - s_inv * dy
        ly = s_inv * dx + c_inv * dy
        d = np.sqrt(lx**2 + ly**2)
        if d < max_range and d > 0.1:
            n_obs_pts = 30
            angle_to_obs = np.arctan2(ly, lx)
            spread = np.random.uniform(-0.15, 0.15, n_obs_pts)
            a = angle_to_obs + spread
            noise_d = np.random.normal(0, sigma_rho, n_obs_pts)
            r = d + noise_d
            obs_pts_list.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    if obs_pts_list:
        obs_pts = np.vstack(obs_pts_list)
        return np.vstack([wall_pts, obs_pts]) if len(wall_pts) > 0 else obs_pts
    return wall_pts

def simulate_obstacle_trajectory(start, velocity, n_steps, dt=0.1):
    """
    Generate a linear obstacle trajectory.

    :param start: (np.ndarray) Length-2 start position [x, y]
    :param velocity: (np.ndarray) Length-2 velocity [vx, vy] in m/s
    :param n_steps: (int) Number of time steps
    :param dt: (float) Time step in seconds
    :return: (np.ndarray) n_steps x 2 trajectory positions
    """
    t = np.arange(n_steps) * dt
    return start + velocity * t[:, None]

def simulate_curved_trajectory(center, radius, omega, n_steps, dt=0.1, start_angle=0.0):
    """
    Generate a circular obstacle trajectory.

    :param center: (np.ndarray) Length-2 circle center [x, y]
    :param radius: (float) Circle radius in meters
    :param omega: (float) Angular velocity in rad/s
    :param n_steps: (int) Number of time steps
    :param dt: (float) Time step in seconds
    :param start_angle: (float) Initial angle in radians
    :return: (np.ndarray) n_steps x 2 trajectory positions
    """
    angles = start_angle + omega * np.arange(n_steps) * dt
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

# === Phase 1: Evaluation metrics ===

def compute_metrics(obstacles_list, gt_positions_list, n_frames):
    """
    Compute tracking metrics: recall, FPR, MOTA, IDSW, RMSE.

    :param obstacles_list: (list) Per-frame list of obstacle dicts
    :param gt_positions_list: (list) Per-frame list of (x, y) ground-truth tuples
    :param n_frames: (int) Total number of frames
    :return: (dict) Metrics dict with keys 'recall', 'fpr', 'mota', 'idsw', 'idsw_rate', 'rmse'
    """
    tp_total, fp_total, fn_total, idsw_total = 0, 0, 0, 0
    rmse_list = []
    prev_id_map = {}
    for frame in range(n_frames):
        obs = obstacles_list[frame]
        gt = gt_positions_list[frame]
        if len(gt) == 0:
            fp_total += len(obs)
            continue
        if len(obs) == 0:
            fn_total += len(gt)
            continue
        obs_centers = np.array([o["center"] for o in obs])
        gt_centers = np.array(gt)
        dist_mat = np.linalg.norm(obs_centers[:, None, :] - gt_centers[None, :, :], axis=2)
        n_det, n_gt = dist_mat.shape
        cost_pad = np.full((n_det, n_gt), 1e6)
        cost_pad[:n_det, :n_gt] = dist_mat
        row_ind, col_ind = linear_sum_assignment(cost_pad)
        matched_obs, matched_gt = set(), set()
        for r, c in zip(row_ind, col_ind):
            if r < n_det and c < n_gt and dist_mat[r, c] <= 1.0:
                matched_obs.add(r)
                matched_gt.add(c)
                rmse_list.append(dist_mat[r, c])
                tid = obs[r].get("track_id", -1)
                if c in prev_id_map and prev_id_map[c] != tid and tid != -1:
                    idsw_total += 1
                prev_id_map[c] = tid
        tp_total += len(matched_gt)
        fp_total += len(obs) - len(matched_obs)
        fn_total += len(gt) - len(matched_gt)
    recall = tp_total / max(tp_total + fn_total, 1)
    fpr = fp_total / max(tp_total + fp_total, 1)
    mota = 1.0 - (fn_total + fp_total + idsw_total) / max(tp_total + fn_total, 1)
    rmse = np.sqrt(np.mean(np.array(rmse_list)**2)) if rmse_list else np.inf
    idsw_rate = idsw_total / max(tp_total, 1)
    return {"recall": recall, "fpr": fpr, "mota": mota, "idsw": idsw_total, "idsw_rate": idsw_rate, "rmse": rmse}

# === Phase 2: Test scenarios ===

def run_scenario(name, grid, grid_origin, resolution, gt_obstacles_traj, n_frames, pose, dt=0.1):
    """
    Run all three algorithms on a single test scenario and collect metrics.

    :param name: (str) Scenario name
    :param grid: (np.ndarray) 2-D binary occupancy grid
    :param grid_origin: (np.ndarray) Length-2 grid origin
    :param resolution: (float) Grid resolution in m/cell
    :param gt_obstacles_traj: (list) List of n_steps x 2 trajectory arrays
    :param n_frames: (int) Number of simulation frames
    :param pose: (np.ndarray) Length-3 robot pose
    :param dt: (float) Time step in seconds
    :return: (tuple) (metrics, results, gt_positions, timings)
    """
    s1 = init_pipeline(grid, grid_origin)
    s2 = init_fixed_pipeline(grid, grid_origin)
    s3 = init_gridnn_pipeline(grid, grid_origin)

    warmup_pts = np.random.randn(20, 2) * 0.1 + 2.5
    run_cycle(s1, warmup_pts, pose, dt)
    run_fixed_cycle(s2, warmup_pts, pose, dt)
    run_gridnn_cycle(s3, warmup_pts, pose, dt)

    results = {"proposed": [], "fixed": [], "grid_nn": []}
    timings = {"proposed": [], "fixed": [], "grid_nn": []}

    for frame in range(n_frames):
        obs_gt = [(o[frame, 0], o[frame, 1], 0.2) for o in gt_obstacles_traj]
        points = simulate_lidar_scan(pose, grid, grid_origin, resolution, obs_gt)
        if len(points) == 0:
            points = np.zeros((1, 2))

        r1 = run_cycle(s1, points, pose, dt)
        s1 = r1['state']
        r2 = run_fixed_cycle(s2, points, pose, dt)
        s2 = r2['state']
        r3 = run_gridnn_cycle(s3, points, pose, dt)
        s3 = r3['state']

        results["proposed"].append(r1["obstacles"])
        results["fixed"].append(r2["obstacles"])
        results["grid_nn"].append(r3["obstacles"])
        timings["proposed"].append(r1["timing"]["total_ms"])
        timings["fixed"].append(r2["timing"]["total_ms"])
        timings["grid_nn"].append(r3["timing"]["total_ms"])

    gt_positions = [[(o[f, 0], o[f, 1]) for o in gt_obstacles_traj] for f in range(n_frames)]
    metrics = {}
    for alg in ["proposed", "fixed", "grid_nn"]:
        m = compute_metrics(results[alg], gt_positions, n_frames)
        m["avg_time_ms"] = np.mean(timings[alg])
        metrics[alg] = m
    return metrics, results, gt_positions, timings

def test_t1_static():
    """
    Test scenario T1: static environment with no moving obstacles.

    :return: (tuple) (metrics, results, gt_pos, timings)
    """
    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    metrics, results, gt_pos, timings = run_scenario("static", grid, origin, res, [], 10, pose)
    return metrics, results, gt_pos, timings

def test_t2_near_wall_single():
    """
    Test scenario T2: single obstacle moving near a wall.

    :return: (tuple) (metrics, results, gt_pos, timings)
    """
    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    traj = simulate_obstacle_trajectory(np.array([1.5, 2.2]), np.array([0.2, 0.0]), 40)
    metrics, results, gt_pos, timings = run_scenario("near_wall_single", grid, origin, res, [traj], 40, pose)
    return metrics, results, gt_pos, timings

def test_t3_multi_cross():
    """
    Test scenario T3: multiple obstacles crossing paths in open space.

    :return: (tuple) (metrics, results, gt_pos, timings)
    """
    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    t1 = simulate_obstacle_trajectory(np.array([1.0, 1.0]), np.array([0.3, 0.15]), 40)
    t2 = simulate_obstacle_trajectory(np.array([3.5, 3.0]), np.array([-0.25, -0.1]), 40)
    t3 = simulate_obstacle_trajectory(np.array([2.0, 3.5]), np.array([0.1, -0.2]), 40)
    metrics, results, gt_pos, timings = run_scenario("multi_cross", grid, origin, res, [t1, t2, t3], 40, pose)
    return metrics, results, gt_pos, timings

def test_t4_near_wall_cross():
    """
    Test scenario T4: two obstacles crossing near a wall.

    :return: (tuple) (metrics, results, gt_pos, timings)
    """
    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    t1 = simulate_obstacle_trajectory(np.array([1.2, 2.3]), np.array([0.25, 0.0]), 40)
    t2 = simulate_obstacle_trajectory(np.array([3.5, 2.3]), np.array([-0.2, 0.05]), 40)
    metrics, results, gt_pos, timings = run_scenario("near_wall_cross", grid, origin, res, [t1, t2], 40, pose)
    return metrics, results, gt_pos, timings

def test_t5_high_speed_curve():
    """
    Test scenario T5: high-speed obstacle on a curved trajectory.

    :return: (tuple) (metrics, results, gt_pos, timings)
    """
    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    traj = simulate_curved_trajectory(np.array([2.0, 2.0]), 1.0, 0.8, 50, dt=0.1, start_angle=0.0)
    metrics, results, gt_pos, timings = run_scenario("high_speed_curve", grid, origin, res, [traj], 50, pose)
    return metrics, results, gt_pos, timings

# === Phase 3: Visualization ===

def plot_comprehensive_comparison(all_metrics, scenarios):
    """
    Plot bar charts comparing all algorithms across metrics and a scatter plot of scene difficulty vs recall.

    :param all_metrics: (dict) Nested dict [scenario][alg] → metrics dict
    :param scenarios: (list) List of scenario name strings
    :return: None
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    algs = ["proposed", "fixed", "grid_nn"]
    active_scenarios = [s for s in scenarios if s != "static"]
    x = np.arange(len(active_scenarios))
    w = 0.25

    metric_configs = [
        ("recall", "Recall", (0, 1.1)),
        ("mota", "MOTA", (-0.5, 1.1)),
        ("rmse", "Position RMSE (m)", None),
        ("idsw_rate", "IDSW Rate", None),
        ("avg_time_ms", "Avg Time (ms)", None),
    ]

    for col, (key, ylabel, ylim) in enumerate(metric_configs):
        ax = axes[0] if col < 3 else axes[1]
        c = col if col < 3 else col - 3
        for i, alg in enumerate(algs):
            vals = []
            for s in active_scenarios:
                v = all_metrics[s][alg][key]
                if key == "rmse" and (v is None or np.isinf(v)):
                    vals.append(0.0)
                elif key == "rmse":
                    vals.append(min(v, 2.0))
                else:
                    vals.append(v)
            bars = ax[c].bar(x + i * w, vals, w, label=ALG_LABELS[alg],
                             color=COLORS[alg], alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if abs(v) > 0.001:
                    ax[c].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                               f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=COLORS[alg])
        ax[c].set_ylabel(ylabel)
        ax[c].set_xticks(x + w)
        ax[c].set_xticklabels([s.replace("_", "\n") for s in active_scenarios], fontsize=7)
        if ylim:
            ax[c].set_ylim(ylim)
        ax[c].legend(fontsize=7)
        ax[c].set_title(f"({chr(97+col)}) {ylabel}", fontweight="bold", fontsize=10)
        ax[c].grid(axis="y", alpha=0.2)

    ax_scatter = axes[1, 2]
    sdf = compute_sdf(create_corridor_map(100, 100), 0.05)
    scenario_obs_init = {
        "near_wall_single": np.array([[1.5, 2.2]]),
        "multi_cross": np.array([[1.0, 1.0], [3.5, 3.0], [2.0, 3.5]]),
        "near_wall_cross": np.array([[1.2, 2.3], [3.5, 2.3]]),
        "high_speed_curve": np.array([[2.0, 2.0]]),
    }
    scene_difficulty = {}
    for s, obs_pos in scenario_obs_init.items():
        ix = np.clip((obs_pos[:, 0] / 0.05).astype(int), 0, sdf.shape[1] - 1)
        iy = np.clip((obs_pos[:, 1] / 0.05).astype(int), 0, sdf.shape[0] - 1)
        scene_difficulty[s] = np.mean(sdf[iy, ix])
    scatter_scenarios = [s for s in active_scenarios if s in scene_difficulty]
    for alg in ["proposed", "fixed", "grid_nn"]:
        xs = [scene_difficulty[s] for s in scatter_scenarios]
        ys = [all_metrics[s][alg]["recall"] for s in scatter_scenarios]
        ax_scatter.scatter(xs, ys, c=COLORS[alg], label=ALG_LABELS[alg],
                           s=60, alpha=0.7, zorder=3, edgecolors="white", linewidth=0.5)
    for s in scatter_scenarios:
        ax_scatter.annotate(s.replace("_", "\n"),
                           (scene_difficulty[s], all_metrics[s]["proposed"]["recall"]),
                           fontsize=6, alpha=0.7, xytext=(5, 5), textcoords="offset points")
    ax_scatter.set_xlabel("Avg SDF at Obstacles (m)")
    ax_scatter.set_ylabel("Recall")
    ax_scatter.set_title("(f) Scene Difficulty vs Recall", fontweight="bold", fontsize=10)
    ax_scatter.legend(fontsize=7)
    ax_scatter.grid(alpha=0.2)

    fig.suptitle("Comprehensive Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sdf_filter_comparison(grid, grid_origin, resolution, pose):
    """
    Visualize raw, fixed-threshold, and SDF-adaptive filtered point clouds side by side.

    :param grid: (np.ndarray) 2-D binary occupancy grid
    :param grid_origin: (np.ndarray) Length-2 grid origin
    :param resolution: (float) Grid resolution in m/cell
    :param pose: (np.ndarray) Length-3 robot pose
    :return: None
    """
    sdf = compute_sdf(grid, resolution)
    obs_near_wall = [(1.5, 2.2, 0.2)]
    obs_open_space = [(2.5, 2.5, 0.2)]
    obs_gt = obs_near_wall + obs_open_space
    points = simulate_lidar_scan(pose, grid, grid_origin, resolution, obs_gt, n_beams=360)
    if len(points) < 5:
        return
    pts_voxel = voxel_filter(points, R_VOXEL)
    pts_global = transform_to_global(pts_voxel, pose)
    pts_adaptive, mask_a = sdf_adaptive_filter(pts_global, sdf, grid_origin, resolution,
                                                TAU_SDF, BETA_SDF, D_NEAR)
    pts_fixed, mask_f = fixed_threshold_filter(pts_global, sdf, grid_origin, resolution, TAU_SDF)

    n_obs_near_raw = np.sum(np.linalg.norm(pts_global - np.array([[1.5, 2.2]]), axis=1) < 0.3)
    n_obs_near_fixed = np.sum(np.linalg.norm(pts_fixed - np.array([[1.5, 2.2]]), axis=1) < 0.3) if len(pts_fixed) > 0 else 0
    n_obs_near_adaptive = np.sum(np.linalg.norm(pts_adaptive - np.array([[1.5, 2.2]]), axis=1) < 0.3) if len(pts_adaptive) > 0 else 0
    retention_near_fixed = n_obs_near_fixed / max(n_obs_near_raw, 1) * 100
    retention_near_adaptive = n_obs_near_adaptive / max(n_obs_near_raw, 1) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    extent = [0, grid.shape[1]*resolution, 0, grid.shape[0]*resolution]
    configs = [
        (pts_global, "Raw (after voxel)", len(pts_global), None),
        (pts_fixed, "Fixed Threshold", len(pts_fixed), retention_near_fixed),
        (pts_adaptive, "SDF Adaptive", len(pts_adaptive), retention_near_adaptive),
    ]
    panel_labels = ["(a)", "(b)", "(c)"]
    for idx, (ax, (pts, title, n_pts, retention)) in enumerate(zip(axes, configs)):
        ax.imshow(grid, origin="lower", cmap="Greys", extent=extent, alpha=0.6)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#3A86FF", alpha=0.6,
                       zorder=3, edgecolors="none")
        for ox, oy, _ in obs_gt:
            ax.plot(ox, oy, "*", color="#FF9E00", markersize=12, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.5)
        ann = f"N={n_pts}"
        if retention is not None:
            ann += f"\nNear-wall retention: {retention:.0f}%"
        ax.set_title(f"{panel_labels[idx]} {title}\n{ann}", fontweight="bold", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.suptitle("SDF Filter Effect Comparison (star = obstacle)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "sdf_filter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_costmap_visualization(grid, grid_origin, resolution, pose):
    """
    Visualize static, dynamic, and fused costmap layers with tracker annotations.

    :param grid: (np.ndarray) 2-D binary occupancy grid
    :param grid_origin: (np.ndarray) Length-2 grid origin
    :param resolution: (float) Grid resolution in m/cell
    :param pose: (np.ndarray) Length-3 robot pose
    :return: None
    """
    c_static = inflate_static_layer(grid, R_INFLATE, resolution)

    s1 = init_pipeline(grid, grid_origin)
    n_warmup = 15
    for frame in range(n_warmup):
        obs_x = 1.5 + 0.15 * frame * DT
        obs_y = 2.2
        obs_gt = [(obs_x, obs_y, 0.2), (3.0, 2.5, 0.2)]
        points = simulate_lidar_scan(pose, grid, grid_origin, resolution, obs_gt)
        if len(points) == 0:
            points = np.zeros((1, 2))
        r1 = run_cycle(s1, points, pose, DT)
        s1 = r1['state']

    confirmed = [t for t in s1['trackers'] if t['confirmed']]
    c_dyn = gaussian_dynamic_layer(confirmed, grid.shape, grid_origin, resolution,
                                    SIGMA_DYN, DT_PRED, W_PRED)
    c_fused = fuse_costmap(c_static, c_dyn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    extent = [0, grid.shape[1]*resolution, 0, grid.shape[0]*resolution]
    panel_labels = ["(a)", "(b)", "(c)"]
    layers = [(axes[0], c_static, "Static Layer"),
              (axes[1], c_dyn, "Dynamic Layer"),
              (axes[2], c_fused, "Fused Costmap")]
    for idx, (ax, cm, title) in enumerate(layers):
        im = ax.imshow(cm, origin="lower", cmap="viridis", extent=extent, vmin=0, vmax=1)
        ax.set_title(f"{panel_labels[idx]} {title}", fontweight="bold", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cost", fontsize=8)

    for trk in confirmed:
        pos = trk['x_trk'][:2]
        vel = trk['x_trk'][2:4]
        for ax_idx, ax in enumerate([axes[1], axes[2]]):
            color = "#FF9E00" if ax_idx == 0 else "#38B000"
            ax.annotate(f"ID:{trk['track_id']}", xy=(pos[0], pos[1]),
                        fontsize=7, color=color, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
            speed = np.linalg.norm(vel)
            if speed > 0.02:
                scale = min(1.0, 0.5 / max(speed, 0.01))
                ax.arrow(pos[0], pos[1], vel[0]*scale, vel[1]*scale,
                         head_width=0.04, head_length=0.02, fc=color, ec=color,
                         linewidth=1.5, zorder=5)

    fig.suptitle(f"Costmap Visualization ({len(confirmed)} confirmed trackers)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "costmap_visualization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tracking_trajectory_timeline(all_scenario_data, scenarios):
    """
    Plot tracking timeline (X/Y position vs frame) for the near_wall_cross scenario.

    :param all_scenario_data: (dict) Nested dict [scenario] → results/gt/timings
    :param scenarios: (list) List of scenario name strings
    :return: None
    """
    name = "near_wall_cross"
    if name not in all_scenario_data:
        return
    data = all_scenario_data[name]
    results = data["results"]
    gt_pos = data["gt_positions"]
    n_frames = len(gt_pos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panel_labels = ["(a)", "(b)"]

    for comp_idx, (ax, ylabel) in enumerate(zip(axes, ["X position (m)", "Y position (m)"])):
        for gt_i in range(len(gt_pos[0])):
            gt_traj = np.array([gt_pos[f][gt_i] for f in range(n_frames)])
            ax.plot(range(n_frames), gt_traj[:, comp_idx], 'k-', linewidth=2.0,
                    label="Ground Truth" if gt_i == 0 else None, zorder=4)

        for alg, ls, marker in [("proposed", "--", "o"), ("fixed", ":", "s"), ("grid_nn", "-.", "D")]:
            obs_list = results[alg]
            trk_positions = {}
            prev_id_map = {}
            idsw_events = []
            for f in range(n_frames):
                for obs in obs_list[f]:
                    tid = obs.get("track_id", -1)
                    pos = obs["center"]
                    if tid not in trk_positions:
                        trk_positions[tid] = []
                    trk_positions[tid].append((f, pos[comp_idx]))
                    for gt_j in range(len(gt_pos[f])):
                        dist = np.linalg.norm(np.array(obs["center"]) - np.array(gt_pos[f][gt_j]))
                        if dist < 1.0:
                            if gt_j in prev_id_map and prev_id_map[gt_j] != tid and tid != -1:
                                idsw_events.append((f, pos[comp_idx]))
                            prev_id_map[gt_j] = tid

            for tid, positions in trk_positions.items():
                frames, vals = zip(*positions)
                ax.plot(frames, vals, ls, color=COLORS[alg], linewidth=1.2, alpha=0.8,
                        marker=marker, markersize=2, markevery=5,
                        label=ALG_LABELS[alg] if tid == list(trk_positions.keys())[0] else None)

            if idsw_events:
                frames_idsw, vals_idsw = zip(*idsw_events)
                ax.scatter(frames_idsw, vals_idsw, marker="x", c="#E63946", s=60, zorder=5,
                           linewidths=2, label=f"IDSW ({ALG_LABELS[alg]})" if alg == "proposed" else None)

        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_labels[comp_idx]} Tracking Timeline — {name} ({ylabel.split()[0]})",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle("Tracking Trajectory Timeline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "tracking_trajectory_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_improvement_heatmap(all_metrics, scenarios):
    """
    Plot heatmap of proposed-vs-fixed improvement percentage across metrics and scenarios.

    :param all_metrics: (dict) Nested dict [scenario][alg] → metrics dict
    :param scenarios: (list) List of scenario name strings
    :return: None
    """
    metric_keys = ["recall", "mota", "rmse", "avg_time_ms"]
    metric_labels = ["Recall", "MOTA", "1/RMSE", "Speed"]

    active_scenarios = [s for s in scenarios if s != "static"]
    n_scenarios = len(active_scenarios)
    n_metrics = len(metric_keys)

    improvement = np.zeros((n_scenarios, n_metrics))
    for i, s in enumerate(active_scenarios):
        for j, key in enumerate(metric_keys):
            proposed_val = all_metrics[s]["proposed"][key]
            fixed_val = all_metrics[s]["fixed"][key]
            if key == "rmse":
                proposed_val = 1.0 / (1.0 + proposed_val)
                fixed_val = 1.0 / (1.0 + fixed_val)
            elif key == "avg_time_ms":
                proposed_val = 1.0 / (1.0 + proposed_val / 10.0)
                fixed_val = 1.0 / (1.0 + fixed_val / 10.0)
            denom = max(abs(fixed_val), 1e-9)
            improvement[i, j] = (proposed_val - fixed_val) / denom * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(improvement.min()), abs(improvement.max()), 1.0)
    im = ax.imshow(improvement, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels([s.replace("_", " ") for s in active_scenarios], fontsize=9)

    for i in range(n_scenarios):
        for j in range(n_metrics):
            val = improvement[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Improvement over Fixed (%)", fontsize=9)
    ax.set_title("Algorithm Improvement Heatmap (Proposed vs Fixed)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "improvement_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# === Phase 4: Main entry ===

def run_all_tests():
    """
    Run all test scenarios, collect metrics, generate visualizations, and save results.

    :return: None
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = ["static", "near_wall_single", "multi_cross", "near_wall_cross", "high_speed_curve"]
    test_fns = [test_t1_static, test_t2_near_wall_single, test_t3_multi_cross,
                test_t4_near_wall_cross, test_t5_high_speed_curve]
    all_metrics = {}
    all_scenario_data = {}

    for name, fn in zip(scenarios, test_fns):
        print(f"[{name}] running...")
        t0 = time.perf_counter()
        metrics, results, gt_pos, timings = fn()
        t1 = time.perf_counter()
        all_metrics[name] = metrics
        all_scenario_data[name] = {"results": results, "gt_positions": gt_pos, "timings": timings}
        print(f"[{name}] done in {t1-t0:.2f}s")
        for alg in ["proposed", "fixed", "grid_nn"]:
            m = metrics[alg]
            print(f"  {alg}: recall={m['recall']:.3f} mota={m['mota']:.3f} "
                  f"rmse={m['rmse']:.3f} idsw={m['idsw']} time={m['avg_time_ms']:.1f}ms")

    print("\n=== Generating Visualizations ===")
    plot_comprehensive_comparison(all_metrics, scenarios)

    grid = create_corridor_map(100, 100)
    res = 0.05; origin = np.array([0.0, 0.0]); pose = np.array([2.5, 2.5, 0.0])
    plot_sdf_filter_comparison(grid, origin, res, pose)
    plot_costmap_visualization(grid, origin, res, pose)
    plot_tracking_trajectory_timeline(all_scenario_data, scenarios)
    plot_improvement_heatmap(all_metrics, scenarios)

    report = {}
    for s in scenarios:
        report[s] = {alg: {k: (float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v)
                               else int(v) if isinstance(v, (np.integer, int))
                               else None)
                           for k, v in all_metrics[s][alg].items()}
                     for alg in ["proposed", "fixed", "grid_nn"]}
    with open(str(RESULTS_DIR / "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'test_report.json'}")
    print(f"Figures saved to {FIGS_DIR}/")

if __name__ == "__main__":
    run_all_tests()
