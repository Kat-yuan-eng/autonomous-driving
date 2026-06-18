"""Comparison of global and local path planning algorithms

author: Kat-yuan-eng (RuiWen Liao)
"""

import math
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.grid_map import generate_random_map, save_metrics_json

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from GlobalPlanning.AStar.a_star import AStarPlanner, calc_path_length
from GlobalPlanning.Dijkstra.dijkstra import DijkstraPlanner
from GlobalPlanning.AdaptiveAStar.adaptive_a_star import AdaptiveAStarPlanner, adaptive_heuristic
from LocalPlanning.RRT.rrt import RRT
from LocalPlanning.DWA.dynamic_window_approach import Config as DWAConfig, dwa_control, motion as dwa_motion
from LocalPlanning.TEB.timed_elastic_band import TEBConfig, optimize_teb

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

COLORS = {
    'astar': '#1f77b4',
    'dijkstra': '#ff7f0e',
    'adaptive_astar': '#2ca02c',
    'dwa': '#d62728',
    'rrt': '#9467bd',
    'teb': '#8c564b',
    'grid': '#cccccc',
}

LINE_STYLES = {
    'astar': '-',
    'dijkstra': '--',
    'adaptive_astar': '-.',
    'dwa': ':',
    'rrt': '-',
    'teb': '--',
}

LINE_MARKERS = {
    'astar': None,
    'dijkstra': None,
    'adaptive_astar': None,
    'dwa': None,
    'rrt': 'o',
    'teb': 's',
}


# === Phase 1: Global Planner Wrappers ===

def dijkstra(grid, start, goal):
    """Run Dijkstra planning via DijkstraPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = DijkstraPlanner(grid)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def astar(grid, start, goal, heuristic='euclidean'):
    """Run A* planning via AStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :param heuristic: (str) 'euclidean', 'manhattan', or 'chebyshev'
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = AStarPlanner(grid, heuristic=heuristic)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def adaptive_astar(grid, start, goal):
    """Run Adaptive A* planning via AdaptiveAStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = AdaptiveAStarPlanner(grid)
    rx, ry, expanded, elapsed_ms = planner.planning(sx, sy, gx, gy, heuristic_type="adaptive")
    return rx, ry, expanded, elapsed_ms


# === Phase 2: Global Planner Comparison ===

def compare_global_planners(grid, start, goal):
    """Compare Dijkstra, A* (3 heuristics), and Adaptive A* on a single grid.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (dict) Per-algorithm metrics
    """
    planners = [
        ('Dijkstra', 'dijkstra', {}),
        ('A*(euclidean)', 'astar', {'heuristic': 'euclidean'}),
        ('A*(manhattan)', 'astar', {'heuristic': 'manhattan'}),
        ('A*(chebyshev)', 'astar', {'heuristic': 'chebyshev'}),
        ('AdaptiveA*', 'adaptive_astar', {}),
    ]

    results = {}
    for name, fn_name, kwargs in planners:
        fn = {'dijkstra': dijkstra, 'astar': astar, 'adaptive_astar': adaptive_astar}[fn_name]
        px, py, exp, t_ms = fn(grid, start, goal, **kwargs)
        p_len = calc_path_length(px, py) if len(px) > 0 else float('inf')
        diag_violation = False
        if len(px) > 1:
            px_arr, py_arr = np.asarray(px), np.asarray(py)
            dpx = np.diff(px_arr)
            dpy = np.diff(py_arr)
            diag_mask = (np.abs(dpx) == 1) & (np.abs(dpy) == 1)
            if np.any(diag_mask):
                diag_idx = np.where(diag_mask)[0]
                for i in diag_idx:
                    if grid[py_arr[i], px_arr[i + 1]] == 1 and grid[py_arr[i + 1], px_arr[i]] == 1:
                        diag_violation = True
                        break
        results[name] = {
            'path_length': round(p_len, 2),
            'expanded_nodes': exp,
            'planning_time_ms': round(t_ms, 2),
            'path_found': len(px) > 0,
            'diagonal_violation': diag_violation,
        }

    return results


def compare_global_on_maps(n_row=100, n_col=100, obs_ratios=None, n_trials=5, seed=42):
    """Run global planner comparison across multiple obstacle densities and trials.

    :param n_row: (int) Grid rows
    :param n_col: (int) Grid columns
    :param obs_ratios: (list) Obstacle density ratios
    :param n_trials: (int) Number of trials per ratio
    :param seed: (int) Random seed base
    :return: (dict) Averaged results per obstacle density
    """
    if obs_ratios is None:
        obs_ratios = [0.1, 0.2, 0.3]

    results = {}
    for ratio in obs_ratios:
        ratio_key = f"obs_{int(ratio * 100)}%"
        accum = {}

        for trial in range(n_trials):
            trial_seed = seed + trial
            grid = generate_random_map(n_row, n_col, ratio, seed=trial_seed)
            start, goal = (0, 0), (n_row - 1, n_col - 1)
            trial_res = compare_global_planners(grid, start, goal)

            for algo_name, metrics in trial_res.items():
                if algo_name not in accum:
                    accum[algo_name] = {'path_length': [], 'expanded_nodes': [], 'planning_time_ms': []}
                if metrics['path_found']:
                    accum[algo_name]['path_length'].append(metrics['path_length'])
                    accum[algo_name]['expanded_nodes'].append(metrics['expanded_nodes'])
                    accum[algo_name]['planning_time_ms'].append(metrics['planning_time_ms'])

        averaged = {}
        for algo_name, vals in accum.items():
            n_valid = len(vals['path_length'])
            averaged[algo_name] = {
                'path_length': round(np.mean(vals['path_length']), 2) if n_valid > 0 else float('inf'),
                'expanded_nodes': round(np.mean(vals['expanded_nodes']), 1) if n_valid > 0 else 0,
                'planning_time_ms': round(np.mean(vals['planning_time_ms']), 2) if n_valid > 0 else 0,
                'n_valid_trials': n_valid,
            }

        results[ratio_key] = averaged

    return results


# === Phase 3: Global Comparison Visualization ===

def plot_global_comparison(results, save_dir='figs'):
    """Plot bar charts comparing global planners across obstacle densities.

    :param results: (dict) Output of compare_global_on_maps
    :param save_dir: (str) Directory to save figure
    """
    algo_names = ['Dijkstra', 'A*(euclidean)', 'A*(manhattan)', 'A*(chebyshev)', 'AdaptiveA*']
    metric_keys = ['path_length', 'expanded_nodes', 'planning_time_ms']
    metric_labels = ['Path Length [m]', 'Expanded Nodes', 'Computation Time [ms]']
    metric_titles = ['Path Length Comparison', 'Node Expansion Comparison', 'Computation Time Comparison']
    ratio_keys = list(results.keys())
    n_algos = len(algo_names)
    n_ratios = len(ratio_keys)

    x = np.arange(n_ratios)
    width = 0.8 / n_algos
    colors = [COLORS['dijkstra'], COLORS['astar'], COLORS['astar'], COLORS['astar'], COLORS['adaptive_astar']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    for ax, metric_key, metric_label, metric_title in zip(axes, metric_keys, metric_labels, metric_titles):
        for i, algo in enumerate(algo_names):
            vals = [results[rk].get(algo, {}).get(metric_key, 0) for rk in ratio_keys]
            bars = ax.bar(x + i * width, vals, width, label=algo, color=colors[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{v:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_title)
        ax.set_xticks(x + width * (n_algos - 1) / 2)
        ax.set_xticklabels(ratio_keys)
        ax.legend(fontsize=7, frameon=True, fancybox=True, loc='best')

    fig.suptitle('Global Path Planning Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = pathlib.Path(save_dir) / 'global_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.show()
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def plot_search_process(grid, start, goal, save_dir='figs'):
    """Plot search process comparison between Dijkstra and Adaptive A*.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :param save_dir: (str) Directory to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    sy, sx = start
    gy, gx = goal

    planners = [
        ('Dijkstra', 'dijkstra', DijkstraPlanner(grid), {}),
        ('AdaptiveA*', 'adaptive_astar', AdaptiveAStarPlanner(grid), {'heuristic_type': 'adaptive'}),
    ]

    for ax, (name, style_key, planner, kwargs) in zip(axes, planners):
        t0 = time.perf_counter()
        if name == 'Dijkstra':
            rx, ry, expanded = planner.planning(sx, sy, gx, gy)
        else:
            rx, ry, expanded, _ = planner.planning(sx, sy, gx, gy, **kwargs)
        t_ms = (time.perf_counter() - t0) * 1000.0

        ax.imshow(grid, cmap='gray_r', origin='upper')

        if len(rx) > 0:
            ax.plot(rx, ry, color=COLORS[style_key], linestyle=LINE_STYLES[style_key],
                    linewidth=2, label='path')

        ax.scatter(sx, sy, c='green', s=80, marker='s', label='start', zorder=5)
        ax.scatter(gx, gy, c='blue', s=80, marker='s', label='goal', zorder=5)
        ax.set_title(f'{name} Search\nexpanded={expanded}, time={t_ms:.2f}ms')
        ax.legend(fontsize=8, frameon=True, fancybox=True, loc='best')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    fig.suptitle('Search Process Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = pathlib.Path(save_dir) / 'search_process_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.show()
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# === Phase 4: Local Planner Comparison ===

def _min_obstacle_dist(path_x, path_y, obstacle_list):
    """Calculate minimum distance from path to obstacle surfaces.

    :param path_x: (list) X coordinates of the path
    :param path_y: (list) Y coordinates of the path
    :param obstacle_list: (list) List of (ox, oy, radius) tuples
    :return: (float) Minimum surface distance (>=0)
    """
    px_arr = np.asarray(path_x)
    py_arr = np.asarray(path_y)
    obs_arr = np.array([(ox, oy, r) for ox, oy, r in obstacle_list])
    dx = px_arr[:, np.newaxis] - obs_arr[:, 0][np.newaxis, :]
    dy = py_arr[:, np.newaxis] - obs_arr[:, 1][np.newaxis, :]
    d = np.hypot(dx, dy) - obs_arr[:, 2][np.newaxis, :]
    return max(0.0, float(np.min(d)))


def _check_collision(path_x, path_y, obstacle_list, robot_radius=1.5):
    """Check collision points between path and obstacles.

    :param path_x: (list) X coordinates of the path
    :param path_y: (list) Y coordinates of the path
    :param obstacle_list: (list) List of (ox, oy, radius) tuples
    :param robot_radius: (float) Robot radius
    :return: (list) Collision records (path_idx, px, py, ox, oy, r, dist)
    """
    px_arr = np.asarray(path_x)
    py_arr = np.asarray(path_y)
    obs_arr = np.array([(ox, oy, r) for ox, oy, r in obstacle_list])
    dx = px_arr[:, np.newaxis] - obs_arr[:, 0][np.newaxis, :]
    dy = py_arr[:, np.newaxis] - obs_arr[:, 1][np.newaxis, :]
    d = np.hypot(dx, dy) - obs_arr[:, 2][np.newaxis, :]
    mask = d < robot_radius
    idx_obs = np.argwhere(mask)
    collisions = [(int(idx), float(px_arr[idx]), float(py_arr[idx]),
                   float(obs_arr[j, 0]), float(obs_arr[j, 1]), float(obs_arr[j, 2]),
                   float(d[idx, j])) for idx, j in idx_obs]
    return collisions


def _calc_smoothness(path_x, path_y):
    """Calculate path smoothness as normalized cumulative curvature change.

    :param path_x: (list) X coordinates of the path
    :param path_y: (list) Y coordinates of the path
    :return: (float) Smoothness metric (lower is smoother)
    """
    if len(path_x) < 4:
        return 0.0
    px, py = np.asarray(path_x), np.asarray(path_y)
    dx = np.diff(px)
    dy = np.diff(py)
    ds = np.hypot(dx, dy) + 1e-9
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    kappa = dtheta / ds[:-1]
    S = float(np.sum(np.abs(np.diff(kappa))))
    path_len = float(np.sum(ds))
    return S / max(path_len, 1e-9)


def _circles_to_point_cloud(obstacle_list, spacing=2.0):
    """Convert circular obstacles to point cloud representation.

    :param obstacle_list: (list) List of (ox, oy, radius) tuples
    :param spacing: (float) Point spacing
    :return: (numpy.ndarray) Point cloud of shape (N, 2)
    """
    pts_list = []
    for ox, oy, r in obstacle_list:
        n = max(2, int(r / spacing))
        ix = np.arange(-n, n + 1)
        iy = np.arange(-n, n + 1)
        gx, gy = np.meshgrid(ox + ix * spacing, oy + iy * spacing)
        mask = (gx - ox) ** 2 + (gy - oy) ** 2 <= r ** 2 + 1e-9
        pts_list.append(np.column_stack([gx[mask], gy[mask]]))
        n_bnd = max(8, int(2 * math.pi * r / spacing))
        angles = np.linspace(0, 2 * math.pi, n_bnd, endpoint=False)
        bnd_pts = np.column_stack([ox + r * np.cos(angles), oy + r * np.sin(angles)])
        pts_list.append(bnd_pts)
    return np.vstack(pts_list) if pts_list else np.empty((0, 2))


def compare_local_planners():
    """Compare RRT, DWA, and TEB local planners on a standard obstacle course.

    :return: (dict) Per-algorithm metrics including composite score J
    """
    obstacle_list = [
        (20, 15, 3), (35, 40, 4), (55, 25, 3), (45, 65, 3), (70, 55, 4),
        (30, 80, 3), (75, 20, 3), (60, 80, 3),
    ]
    start = (5, 5)
    goal = (95, 90)
    results = {}

    # RRT
    t0 = time.perf_counter()
    rrt = RRT(start=start, goal=goal,
              obstacle_list=obstacle_list,
              rand_area=(0, 100),
              expand_dis=5.0,
              path_resolution=1.0,
              goal_sample_rate=15,
              max_iter=3000,
              robot_radius=1.5,
              safety_margin=1.0,
              use_rrt_star=True,
              connect_circle_dist=50.0)
    rrt_path, rrt_nodes, rrt_time = rrt.planning()
    rrt_elapsed = (time.perf_counter() - t0) * 1000.0

    if rrt_path is not None and len(rrt_path) > 1:
        rrt_path = rrt.smooth_path(rrt_path, max_iter=300)
        rrt_px = [p[0] for p in rrt_path]
        rrt_py = [p[1] for p in rrt_path]
        rrt_px.reverse()
        rrt_py.reverse()
        rrt_length = calc_path_length(rrt_px, rrt_py)
        rrt_min_dist = _min_obstacle_dist(rrt_px, rrt_py, obstacle_list)
        rrt_smoothness = _calc_smoothness(rrt_px, rrt_py)
        rrt_pts = np.array(list(zip(rrt_px, rrt_py)))
        rrt_dp = np.diff(rrt_pts, axis=0)
        rrt_seg_len = np.linalg.norm(rrt_dp, axis=1)
        rrt_cv = np.std(rrt_seg_len) / max(np.mean(rrt_seg_len), 1e-9)
        rrt_adapt = round(max(0, min(1.0, 1.0 / (1.0 + rrt_cv))), 2)
        results['RRT'] = {
            'smoothness': round(rrt_smoothness, 6),
            'obstacle_distance': round(min(1.0, rrt_min_dist / 5.0), 2),
            'computation_time': round(max(0, 1.0 - rrt_elapsed / 1000.0), 2),
            'dynamic_adaptability': rrt_adapt,
            'path_length': round(rrt_length, 2),
            'min_obs_dist': round(rrt_min_dist, 2),
            'plan_time_ms': round(rrt_elapsed, 2),
            'exec_time': round(rrt_length / 2.5, 2),
        }
    else:
        results['RRT'] = {'smoothness': 0.0, 'obstacle_distance': 0.0, 'computation_time': 0.0, 'dynamic_adaptability': 0.0, 'path_length': 0, 'min_obs_dist': 0, 'plan_time_ms': 0}

    # DWA
    dwa_config = DWAConfig()
    dwa_config.robot_radius = 1.5
    dwa_config.safety_margin = 0.3
    dwa_config.max_speed = 2.5
    dwa_config.max_accel = 2.0
    dwa_config.max_steer = math.radians(30)
    dwa_config.max_steer_rate = math.radians(100)
    dwa_config.wheelbase = 1.5
    dwa_config.v_resolution = 0.1
    dwa_config.steer_resolution = math.radians(2)
    dwa_config.predict_time = 3.0
    dwa_config.to_goal_cost_gain = 10.0
    dwa_config.speed_cost_gain = 1.0
    dwa_config.obstacle_cost_gain = 5.0
    dwa_config.robot_stuck_flag_cons = 0.01
    ob = _circles_to_point_cloud(obstacle_list)
    dwa_x = np.array([start[0], start[1], math.atan2(goal[1]-start[1], goal[0]-start[0]), 0.0, 0.0])
    dwa_traj = [dwa_x.copy()]
    dwa_waypoints = [(10, 12), (18, 25), (25, 35), (22, 50), (40, 55), (55, 52), (65, 62), (78, 78), (88, 86), tuple(goal)]
    wp_idx = 0
    dwa_goal = np.array(dwa_waypoints[wp_idx])
    t0 = time.perf_counter()
    stuck_count = 0
    prev_pos = dwa_x[:2].copy()
    for step in range(5000):
        u, _ = dwa_control(dwa_x, dwa_config, dwa_goal, ob)
        dwa_x = dwa_motion(dwa_x, u, dwa_config.dt, dwa_config.wheelbase)
        dwa_traj.append(dwa_x.copy())
        dist_to_wp = math.hypot(dwa_x[0] - dwa_goal[0], dwa_x[1] - dwa_goal[1])
        if wp_idx < len(dwa_waypoints) - 1:
            reach_thresh = dwa_config.robot_radius + 3.0
        else:
            reach_thresh = dwa_config.robot_radius + 1.0
        if dist_to_wp < reach_thresh:
            wp_idx += 1
            if wp_idx >= len(dwa_waypoints):
                break
            dwa_goal = np.array(dwa_waypoints[wp_idx])
            stuck_count = 0
            prev_pos = dwa_x[:2].copy()
            continue
        if np.linalg.norm(dwa_x[:2] - prev_pos) < 0.1:
            stuck_count += 1
            if stuck_count > 50:
                nearest_obs_d = min(math.hypot(dwa_x[0]-ox, dwa_x[1]-oy)-r for ox,oy,r in obstacle_list)
                print(f"[DWA] stuck at step {step}, pos=({dwa_x[0]:.1f},{dwa_x[1]:.1f}), nearest_obs_d={nearest_obs_d:.2f}")
                break
        else:
            stuck_count = 0
            prev_pos = dwa_x[:2].copy()
    dwa_elapsed = (time.perf_counter() - t0) * 1000.0
    dwa_per_iter_ms = dwa_elapsed / max(step + 1, 1)
    dwa_traj_arr = np.array(dwa_traj)
    dwa_px, dwa_py = dwa_traj_arr[:, 0], dwa_traj_arr[:, 1]
    dwa_length = calc_path_length(dwa_px, dwa_py)
    dwa_min_dist = _min_obstacle_dist(dwa_px.tolist(), dwa_py.tolist(), obstacle_list)
    dwa_smoothness = _calc_smoothness(dwa_px.tolist(), dwa_py.tolist())
    print(f"[DWA] path_length={dwa_length:.2f}, min_obs_dist={dwa_min_dist:.2f}, steps={step+1}, per_iter={dwa_per_iter_ms:.2f}ms")
    dwa_vel_arr = dwa_traj_arr[:, 3] if dwa_traj_arr.shape[1] > 3 else np.zeros(len(dwa_traj_arr))
    dwa_v_diff = np.diff(dwa_vel_arr) if len(dwa_vel_arr) > 1 else np.array([0.0])
    dwa_adapt = round(max(0, min(1.0, 1.0 - np.std(dwa_v_diff) / max(dwa_config.max_speed, 1e-9))), 2)
    dwa_exec_time = (step + 1) * dwa_config.dt
    results['DWA'] = {
        'smoothness': round(dwa_smoothness, 6),
        'obstacle_distance': round(min(1.0, dwa_min_dist / 5.0), 2),
        'computation_time': round(max(0, 1.0 - dwa_elapsed / 2000.0), 2),
        'dynamic_adaptability': dwa_adapt,
        'path_length': round(dwa_length, 2),
        'min_obs_dist': round(dwa_min_dist, 2),
        'plan_time_ms': round(dwa_elapsed, 2),
        'exec_time': round(dwa_exec_time, 2),
    }

    # TEB
    teb_config = TEBConfig()
    teb_config.n_poses = 30
    teb_config.min_obstacle_dist = 2.5
    teb_config.robot_radius = 1.5
    teb_config.max_vel = 2.5
    teb_config.weight_obstacle = 50.0
    teb_config.weight_path = 0.5
    teb_config.n_opt_iter = 30
    teb_config.weight_kin = 5.0
    n_pts = teb_config.n_poses
    waypoints = np.array([
        [5, 5], [15, 18], [25, 28], [40, 35], [50, 45], [80, 78], [95, 90]
    ])
    t_wp = np.linspace(0, 1, len(waypoints))
    t_fine = np.linspace(0, 1, n_pts)
    ref_x = np.interp(t_fine, t_wp, waypoints[:, 0])
    ref_y = np.interp(t_fine, t_wp, waypoints[:, 1])
    ref_beta = np.arctan2(np.gradient(ref_y), np.gradient(ref_x))
    ref_path = np.column_stack([ref_x, ref_y, ref_beta])
    obstacles_arr = _circles_to_point_cloud(obstacle_list)
    start_state = ref_path[0].copy()
    t0 = time.perf_counter()
    teb_poses, teb_dt, teb_plan_ms, teb_history, teb_nit = optimize_teb(ref_path, start_state, obstacles_arr, teb_config, goal=np.array(goal))
    teb_elapsed = (time.perf_counter() - t0) * 1000.0
    teb_px = teb_poses[:, 0].tolist()
    teb_py = teb_poses[:, 1].tolist()
    teb_length = calc_path_length(teb_px, teb_py)
    teb_min_dist = _min_obstacle_dist(teb_px, teb_py, obstacle_list)
    teb_smoothness = _calc_smoothness(teb_px, teb_py)
    teb_dp = np.diff(teb_poses[:, :2], axis=0)
    teb_v = np.linalg.norm(teb_dp, axis=1) / np.maximum(teb_dt, 1e-9)
    teb_adapt = round(max(0, min(1.0, 1.0 - np.std(np.diff(teb_v)) / max(teb_config.max_vel, 1e-9))), 2)
    teb_exec_time = float(np.sum(teb_dt))
    teb_per_iter = teb_elapsed / max(teb_nit, 1)
    teb_speed = max(0, 1.0 - teb_elapsed / 1000.0)
    print(f"[TEB] path_length={teb_length:.2f}, min_obs_dist={teb_min_dist:.2f}, exec_time={teb_exec_time:.2f}s, nit={teb_nit}, per_iter={teb_per_iter:.2f}ms")
    results['TEB'] = {
        'smoothness': round(teb_smoothness, 6),
        'obstacle_distance': round(min(1.0, teb_min_dist / 5.0), 2),
        'computation_time': round(teb_speed, 2),
        'dynamic_adaptability': teb_adapt,
        'path_length': round(teb_length, 2),
        'min_obs_dist': round(teb_min_dist, 2),
        'plan_time_ms': round(teb_elapsed, 2),
        'exec_time': round(teb_exec_time, 2),
    }

    L_ref = min(m['path_length'] for m in results.values() if m['path_length'] > 0)
    T_plan_ref = min(m['plan_time_ms'] for m in results.values() if m['plan_time_ms'] > 0)
    S_ref = min(m['smoothness'] for m in results.values() if m['smoothness'] > 0)
    T_exec_ref = min(m['exec_time'] for m in results.values() if m['exec_time'] > 0)

    for algo_name, m in results.items():
        J = (0.3 * m['path_length'] / max(L_ref, 1e-9)
             + 0.2 * m['plan_time_ms'] / max(T_plan_ref, 1e-9)
             + 0.3 * m['smoothness'] / max(S_ref, 1e-9)
             + 0.2 * m['exec_time'] / max(T_exec_ref, 1e-9))
        m['J'] = round(J, 4)
        print(f"[{algo_name}] J={J:.4f} (L={m['path_length']/max(L_ref,1e-9):.3f}, T_plan={m['plan_time_ms']/max(T_plan_ref,1e-9):.3f}, S={m['smoothness']/max(S_ref,1e-9):.3f}, T_exec={m['exec_time']/max(T_exec_ref,1e-9):.3f})")

    return results


def plot_local_comparison(results, save_dir='figs'):
    """Plot multi-metric bar charts for local planner comparison.

    :param results: (dict) Output of compare_local_planners
    :param save_dir: (str) Directory to save figure
    """
    algo_names = list(results.keys())
    colors = {"RRT": COLORS['rrt'], "DWA": COLORS['dwa'], "TEB": COLORS['teb']}

    raw_metrics = {
        "Smoothness": {a: results[a]["smoothness"] for a in algo_names},
        "Safety": {a: results[a]["obstacle_distance"] for a in algo_names},
        "Speed": {a: results[a]["computation_time"] for a in algo_names},
        "Adaptability": {a: results[a]["dynamic_adaptability"] for a in algo_names},
        "Path Length [m]": {a: results[a]["path_length"] for a in algo_names},
        "Composite J": {a: results[a]["J"] for a in algo_names},
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, (metric_name, vals) in enumerate(raw_metrics.items()):
        ax = axes[idx // 3][idx % 3]
        x = np.arange(len(algo_names))
        bar_vals = [vals[a] for a in algo_names]
        bar_colors = [colors[a] for a in algo_names]
        bars = ax.bar(x, bar_vals, width=0.5, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar_i, (bar, v) in enumerate(zip(bars, bar_vals)):
            fmt = f"{v:.4f}" if v < 0.01 else f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=bar_colors[bar_i])
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_title(f"{panel_labels[idx]} {metric_name}", fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Local Planner Multi-Metric Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = pathlib.Path(save_dir) / 'local_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def plot_trajectory_comparison(save_dir='figs'):
    """Plot trajectory comparison of RRT, DWA, and TEB on the same obstacle map.

    :param save_dir: (str) Directory to save figure
    """
    obstacle_list = [
        (20, 15, 3), (35, 40, 4), (55, 25, 3), (45, 65, 3), (70, 55, 4),
        (30, 80, 3), (75, 20, 3), (60, 80, 3),
    ]
    start = (5, 5)
    goal = (95, 90)
    robot_radius = 1.5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    for ox, oy, r in obstacle_list:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5, zorder=2)
        ax.add_patch(circle)
        inflated_r = r + robot_radius
        safety = plt.Circle((ox, oy), inflated_r, color='orange', alpha=0.12,
                             linestyle='--', fill=False, linewidth=1.0, zorder=2)
        ax.add_patch(safety)

    algo_results = {}

    # RRT
    rrt = RRT(start=start, goal=goal,
              obstacle_list=obstacle_list,
              rand_area=(0, 100),
              expand_dis=5.0, path_resolution=1.0,
              goal_sample_rate=15, max_iter=3000,
              robot_radius=robot_radius, safety_margin=1.0)
    rrt_path, _, _ = rrt.planning()
    if rrt_path is not None and len(rrt_path) > 1:
        rrt_path = rrt.smooth_path(rrt_path, max_iter=300)
        rrt_px = [p[0] for p in rrt_path]
        rrt_py = [p[1] for p in rrt_path]
        rrt_px.reverse()
        rrt_py.reverse()
        rrt_min_d = _min_obstacle_dist(rrt_px, rrt_py, obstacle_list)
        rrt_collisions = _check_collision(rrt_px, rrt_py, obstacle_list, robot_radius)
        rrt_length = calc_path_length(rrt_px, rrt_py)
        algo_results['RRT'] = {'px': rrt_px, 'py': rrt_py, 'min_d': rrt_min_d, 'collisions': rrt_collisions, 'length': rrt_length}
        ax.plot(rrt_px, rrt_py, linestyle=LINE_STYLES['rrt'], marker=LINE_MARKERS['rrt'],
                markersize=3, linewidth=2, color=COLORS['rrt'],
                label=f'RRT (d_min={rrt_min_d:.2f})', alpha=0.85, zorder=3)
        step = max(1, len(rrt_px) // 8)
        for i in range(0, len(rrt_px), step):
            ax.add_patch(plt.Circle((rrt_px[i], rrt_py[i]), robot_radius,
                                     color=COLORS['rrt'], alpha=0.15, zorder=2))

    # DWA
    dwa_config = DWAConfig()
    dwa_config.robot_radius = robot_radius
    dwa_config.safety_margin = 0.3
    dwa_config.max_speed = 2.5
    dwa_config.max_accel = 2.0
    dwa_config.max_steer = math.radians(30)
    dwa_config.max_steer_rate = math.radians(100)
    dwa_config.wheelbase = 1.5
    dwa_config.v_resolution = 0.1
    dwa_config.steer_resolution = math.radians(2)
    dwa_config.predict_time = 3.0
    dwa_config.to_goal_cost_gain = 10.0
    dwa_config.speed_cost_gain = 1.0
    dwa_config.obstacle_cost_gain = 5.0
    dwa_config.robot_stuck_flag_cons = 0.01
    ob = _circles_to_point_cloud(obstacle_list)
    dwa_x = np.array([start[0], start[1], math.atan2(goal[1]-start[1], goal[0]-start[0]), 0.0, 0.0])
    dwa_traj = [dwa_x.copy()]
    dwa_waypoints = [(10, 12), (18, 25), (25, 35), (22, 50), (40, 55), (55, 52), (65, 62), (78, 78), (88, 86), goal]
    wp_idx = 0
    dwa_goal = np.array(dwa_waypoints[wp_idx])
    stuck_count = 0
    prev_pos = dwa_x[:2].copy()
    for step in range(5000):
        u, _ = dwa_control(dwa_x, dwa_config, dwa_goal, ob)
        dwa_x = dwa_motion(dwa_x, u, dwa_config.dt, dwa_config.wheelbase)
        dwa_traj.append(dwa_x.copy())
        dist_to_wp = math.hypot(dwa_x[0] - dwa_goal[0], dwa_x[1] - dwa_goal[1])
        if wp_idx < len(dwa_waypoints) - 1:
            reach_thresh = dwa_config.robot_radius + 3.0
        else:
            reach_thresh = dwa_config.robot_radius + 1.0
        if dist_to_wp < reach_thresh:
            wp_idx += 1
            if wp_idx >= len(dwa_waypoints):
                break
            dwa_goal = np.array(dwa_waypoints[wp_idx])
            stuck_count = 0
            prev_pos = dwa_x[:2].copy()
            continue
        if np.linalg.norm(dwa_x[:2] - prev_pos) < 0.1:
            stuck_count += 1
            if stuck_count > 50:
                nearest_obs_d = min(math.hypot(dwa_x[0]-ox, dwa_x[1]-oy)-r for ox,oy,r in obstacle_list)
                print(f"[DWA] stuck at step {step}, pos=({dwa_x[0]:.1f},{dwa_x[1]:.1f}), nearest_obs_d={nearest_obs_d:.2f}")
                break
        else:
            stuck_count = 0
            prev_pos = dwa_x[:2].copy()
    dwa_traj_arr = np.array(dwa_traj)
    dwa_px = dwa_traj_arr[:, 0].tolist()
    dwa_py = dwa_traj_arr[:, 1].tolist()
    dist_final = math.hypot(dwa_px[-1] - goal[0], dwa_py[-1] - goal[1])
    if dist_final > robot_radius and dist_final < robot_radius + 5.0:
        n_interp = max(2, int(dist_final / 0.5))
        interp_x = np.linspace(dwa_px[-1], goal[0], n_interp + 1)[1:].tolist()
        interp_y = np.linspace(dwa_py[-1], goal[1], n_interp + 1)[1:].tolist()
        collision_on_interp = False
        for ix, iy in zip(interp_x, interp_y):
            for ox, oy, r in obstacle_list:
                if math.hypot(ix - ox, iy - oy) < r + robot_radius:
                    collision_on_interp = True
                    break
            if collision_on_interp:
                break
        if not collision_on_interp:
            dwa_px.extend(interp_x)
            dwa_py.extend(interp_y)
    dwa_min_d = _min_obstacle_dist(dwa_px, dwa_py, obstacle_list)
    dwa_collisions = _check_collision(dwa_px, dwa_py, obstacle_list, robot_radius)
    dwa_length = calc_path_length(dwa_px, dwa_py)
    algo_results['DWA'] = {'px': dwa_px, 'py': dwa_py, 'min_d': dwa_min_d, 'collisions': dwa_collisions, 'length': dwa_length}
    ax.plot(dwa_px, dwa_py, linestyle=LINE_STYLES['dwa'], linewidth=2,
            color=COLORS['dwa'], label=f'DWA (d_min={dwa_min_d:.2f})', alpha=0.85, zorder=3)
    step = max(1, len(dwa_px) // 8)
    for i in range(0, len(dwa_px), step):
        ax.add_patch(plt.Circle((dwa_px[i], dwa_py[i]), robot_radius,
                                 color=COLORS['dwa'], alpha=0.15, zorder=2))

    # TEB
    teb_config = TEBConfig()
    teb_config.n_poses = 30
    teb_config.min_obstacle_dist = 2.5
    teb_config.robot_radius = robot_radius
    teb_config.max_vel = 2.5
    teb_config.weight_obstacle = 50.0
    teb_config.weight_path = 0.5
    teb_config.n_opt_iter = 30
    teb_config.weight_kin = 5.0
    n_pts = teb_config.n_poses
    waypoints = np.array([
        [5, 5], [15, 18], [25, 28], [40, 35], [50, 45], [80, 78], [95, 90]
    ])
    t_wp = np.linspace(0, 1, len(waypoints))
    t_fine = np.linspace(0, 1, n_pts)
    ref_x = np.interp(t_fine, t_wp, waypoints[:, 0])
    ref_y = np.interp(t_fine, t_wp, waypoints[:, 1])
    ref_beta = np.arctan2(np.gradient(ref_y), np.gradient(ref_x))
    ref_path = np.column_stack([ref_x, ref_y, ref_beta])
    obstacles_arr = _circles_to_point_cloud(obstacle_list)
    start_state = ref_path[0].copy()
    teb_poses, _, _, _, _ = optimize_teb(ref_path, start_state, obstacles_arr, teb_config, goal=np.array(goal))
    teb_px = teb_poses[:, 0].tolist()
    teb_py = teb_poses[:, 1].tolist()
    teb_min_d = _min_obstacle_dist(teb_px, teb_py, obstacle_list)
    teb_collisions = _check_collision(teb_px, teb_py, obstacle_list, robot_radius)
    teb_length = calc_path_length(teb_px, teb_py)
    algo_results['TEB'] = {'px': teb_px, 'py': teb_py, 'min_d': teb_min_d, 'collisions': teb_collisions, 'length': teb_length}
    ax.plot(teb_px, teb_py, linestyle=LINE_STYLES['teb'], marker=LINE_MARKERS['teb'],
            markersize=3, linewidth=2, color=COLORS['teb'],
            label=f'TEB (d_min={teb_min_d:.2f})', alpha=0.85, zorder=3)
    step = max(1, len(teb_px) // 8)
    for i in range(0, len(teb_px), step):
        ax.add_patch(plt.Circle((teb_px[i], teb_py[i]), robot_radius,
                                 color=COLORS['teb'], alpha=0.15, zorder=2))

    for name, res in algo_results.items():
        if res['collisions']:
            for _, cx, cy, ox, oy, r, d in res['collisions'][:5]:
                ax.plot(cx, cy, 'x', color='red', markersize=10, markeredgewidth=2, zorder=6)
            print(f"[COLLISION] {name}: {len(res['collisions'])} points inside obstacles!")

    ax.plot(start[0], start[1], 'gs', markersize=12, label='Start', zorder=5)
    ax.plot(goal[0], goal[1], 'b^', markersize=12, label='Goal', zorder=5)

    info_lines = []
    for name, res in algo_results.items():
        status = "PASS" if not res['collisions'] else "FAIL"
        info_lines.append(f"{name}: d_min={res['min_d']:.2f}, len={res['length']:.1f} [{status}]")
    info_text = "\n".join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), zorder=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Local Planner Trajectories', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, fancybox=True, loc='best')
    ax.set_aspect('equal')

    fig.tight_layout()
    save_path = pathlib.Path(save_dir) / 'trajectory_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# === Phase 5: Main ===

def main():
    """Run full comparison pipeline: global planners, local planners, and generate reports."""
    print("=== Global Planner Comparison ===")
    global_results = compare_global_on_maps(
        n_row=100, n_col=100,
        obs_ratios=[0.1, 0.2, 0.3],
        n_trials=3, seed=42,
    )

    for ratio_key, algo_data in global_results.items():
        print(f"\n[{ratio_key}]")
        for algo_name, metrics in algo_data.items():
            print(f"  {algo_name:<20} path_len={metrics['path_length']:<8} "
                  f"expanded={metrics['expanded_nodes']:<8} "
                  f"time={metrics['planning_time_ms']:<8}ms "
                  f"valid_trials={metrics['n_valid_trials']}")

    print("\n=== Generating Charts ===")
    figs_dir = str(pathlib.Path(__file__).parent / 'figs')
    plot_global_comparison(global_results, save_dir=figs_dir)

    demo_grid = generate_random_map(100, 100, 0.2, seed=42)
    start, goal = (0, 0), (99, 99)
    plot_search_process(demo_grid, start, goal, save_dir=figs_dir)

    local_results = compare_local_planners()
    plot_local_comparison(local_results, save_dir=figs_dir)

    plot_trajectory_comparison(save_dir=figs_dir)

    all_metrics = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'global_comparison': global_results,
        'local_comparison': local_results,
    }

    results_dir = pathlib.Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'comparison_metrics.json'
    save_metrics_json(all_metrics, metrics_path)
    print(f"\n[output] metrics saved to {metrics_path}")

    print("\n=== Summary ===")
    print(f"{'Algorithm':<20} {'Ratio':<10} {'PathLen':<10} {'Expanded':<10} {'Time(ms)':<10}")
    print("-" * 60)
    for ratio_key, algo_data in global_results.items():
        for algo_name, metrics in algo_data.items():
            print(f"{algo_name:<20} {ratio_key:<10} {metrics['path_length']:<10} "
                  f"{metrics['expanded_nodes']:<10} {metrics['planning_time_ms']:<10}")

    print("\n=== Local Planner Summary ===")
    print(f"{'Algorithm':<15} {'Smooth':<8} {'Safety':<8} {'Speed':<8} {'Adapt':<8} {'PathLen':<10} {'MinDist':<10} {'Time(ms)':<10}")
    print("-" * 80)
    for algo_name, metrics in local_results.items():
        print(f"{algo_name:<15} {metrics['smoothness']:<8.2f} "
              f"{metrics['obstacle_distance']:<8.2f} "
              f"{metrics['computation_time']:<8.2f} "
              f"{metrics['dynamic_adaptability']:<8.2f} "
              f"{metrics.get('path_length', 0):<10.2f} "
              f"{metrics.get('min_obs_dist', 0):<10.2f} "
              f"{metrics.get('plan_time_ms', 0):<10.2f}")


if __name__ == '__main__':
    main()
