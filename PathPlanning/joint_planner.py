"""Joint global-local planning framework with replanning

author: Kat-yuan-eng (RuiWen Liao)
"""

import math
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.grid_map import generate_random_map, inflate_obstacles, save_path_csv, save_metrics_json

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from GlobalPlanning.Dijkstra.dijkstra import DijkstraPlanner
from GlobalPlanning.AStar.a_star import AStarPlanner
from GlobalPlanning.AdaptiveAStar.adaptive_a_star import AdaptiveAStarPlanner
from LocalPlanning.DWA.dynamic_window_approach import Config as DWAConfig, dwa_control, motion as dwa_motion
from LocalPlanning.TEB.timed_elastic_band import TEBConfig, optimize_teb, smooth_acceleration

DEFAULT_CONFIG = {
    'local_planner': 'TEB',
    'n_ref': 20,
    'd_deviate': 3.0,  # [m]
    'd_obs_trigger': 3.0,  # [m]
    'teb_n_poses': 20,
    'teb_max_vel': 2.5,  # [m/s]
    'teb_max_acc': 2.0,  # [m/s^2]
    'teb_min_obs_dist': 1.5,  # [m]
    'teb_weight_path': 2.0,
    'teb_weight_obs': 50.0,
    'teb_weight_vel': 1.0,
    'teb_weight_kin': 10.0,
    'teb_weight_time': 1.0,
    'teb_weight_acc': 5.0,
    'teb_weight_curv': 2.0,
    'teb_n_opt_iter': 50,
    'teb_wheelbase': 0.3,  # [m]
    'heuristic': 'euclidean',
    'k_sigmoid': 0.5,
    'inflate_radius': 1,  # [cell]
    'cell_size': 1.0,  # [m]
}


# === Phase 1: Global Planner Wrappers ===

def dijkstra_plan(grid, start, goal):
    """Run Dijkstra planning via DijkstraPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sx, sy = start
    gx, gy = goal
    planner = DijkstraPlanner(grid)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def astar_plan(grid, start, goal, heuristic='euclidean'):
    """Run A* planning via AStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param heuristic: (str) 'euclidean', 'manhattan', or 'chebyshev'
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sx, sy = start
    gx, gy = goal
    planner = AStarPlanner(grid, heuristic=heuristic)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def adaptive_astar_plan(grid, start, goal, k_sigmoid=0.5):
    """Run Adaptive A* planning via AdaptiveAStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param k_sigmoid: (float) Sigmoid sharpness for adaptive heuristic
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sx, sy = start
    gx, gy = goal
    planner = AdaptiveAStarPlanner(grid)
    rx, ry, expanded, elapsed_ms = planner.planning(sx, sy, gx, gy, heuristic_type="adaptive", k_sigmoid=k_sigmoid)
    return rx, ry, expanded, elapsed_ms


# === Phase 2: Reference Extraction & Replanning ===

def extract_reference(path_x, path_y, current_pos, n_ref=20):
    """Extract a local reference path segment from the global path.

    :param path_x: (list) X coordinates of the global path
    :param path_y: (list) Y coordinates of the global path
    :param current_pos: (tuple) (cx, cy) current robot position
    :param n_ref: (int) Number of reference points to extract
    :return: (tuple) (ref_x, ref_y) local reference path coordinates
    """
    px = np.asarray(path_x, dtype=float)
    py = np.asarray(path_y, dtype=float)
    cx, cy = current_pos
    dists = (px - cx) ** 2 + (py - cy) ** 2
    idx_closest = int(np.argmin(dists))
    idx_end = min(idx_closest + n_ref, len(px))
    return px[idx_closest:idx_end].tolist(), py[idx_closest:idx_end].tolist()


def need_replan(current_pos, global_path, obstacles_dyn, d_deviate=3.0, d_obs_trigger=3.0):
    """Check whether replanning is needed based on deviation or dynamic obstacles.

    :param current_pos: (tuple) (cx, cy) current robot position
    :param global_path: (tuple) (path_x_list, path_y_list) global path
    :param obstacles_dyn: (list or None) Dynamic obstacle positions as (x, y) pairs
    :param d_deviate: (float) Deviation threshold for replanning
    :param d_obs_trigger: (float) Obstacle proximity threshold for replanning
    :return: (bool) True if replanning is needed
    """
    gx = np.array(global_path[0])
    gy = np.array(global_path[1])
    cx, cy = current_pos

    dists = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    closest_idx = np.argmin(dists)

    forward_dists = dists[closest_idx:]
    if len(forward_dists) > 0 and np.min(forward_dists) > d_deviate:
        return True

    if obstacles_dyn is not None and len(obstacles_dyn) > 0:
        obs = np.asarray(obstacles_dyn)
        d_obs_all = np.sqrt(
            (gx[:, np.newaxis] - obs[:, 0][np.newaxis, :]) ** 2
            + (gy[:, np.newaxis] - obs[:, 1][np.newaxis, :]) ** 2)
        if np.min(d_obs_all) < d_obs_trigger:
            return True

    return False


# === Phase 3: Local Planning Helpers ===

def _build_teb_config(cfg):
    """Build a TEBConfig object from a configuration dictionary.

    :param cfg: (dict) Configuration dictionary with TEB parameters
    :return: (TEBConfig) Configured TEB configuration object
    """
    teb_cfg = TEBConfig()
    teb_cfg.n_poses = cfg['teb_n_poses']
    teb_cfg.max_vel = cfg['teb_max_vel']
    teb_cfg.max_acc = cfg['teb_max_acc']
    teb_cfg.min_obstacle_dist = cfg['teb_min_obs_dist']
    teb_cfg.weight_path = cfg['teb_weight_path']
    teb_cfg.weight_obstacle = cfg['teb_weight_obs']
    teb_cfg.weight_vel = cfg['teb_weight_vel']
    teb_cfg.weight_kin = cfg['teb_weight_kin']
    teb_cfg.weight_time = cfg['teb_weight_time']
    teb_cfg.weight_acc = cfg['teb_weight_acc']
    teb_cfg.weight_curv = cfg['teb_weight_curv']
    teb_cfg.n_opt_iter = cfg['teb_n_opt_iter']
    teb_cfg.wheelbase = cfg['teb_wheelbase']
    return teb_cfg


def _build_ref_path(ref_x, ref_y, n_poses, start_state):
    """Build a reference path array with interpolated headings for TEB optimization.

    :param ref_x: (list) X coordinates of the reference path
    :param ref_y: (list) Y coordinates of the reference path
    :param n_poses: (int) Number of TEB poses
    :param start_state: (numpy.ndarray) Start state [x, y, yaw]
    :return: (numpy.ndarray) Reference path of shape (n_poses, 3)
    """
    ref_x_arr = np.asarray(ref_x, dtype=float)
    ref_y_arr = np.asarray(ref_y, dtype=float)
    n_pts = len(ref_x_arr)
    indices = np.linspace(0, n_pts - 1, n_poses).astype(int)
    rx_s, ry_s = ref_x_arr[indices], ref_y_arr[indices]
    dx_ref = np.gradient(rx_s)
    dy_ref = np.gradient(ry_s)
    rtheta_s = np.arctan2(dy_ref, dx_ref)
    ref_path = np.column_stack([rx_s, ry_s, rtheta_s])
    ref_path[0] = start_state
    return ref_path


def teb_plan(ref_x, ref_y, start_state, obstacles, cfg, goal=None):
    """Run TEB local planning via optimize_teb from the independent module.

    :param ref_x: (list) X coordinates of the reference path
    :param ref_y: (list) Y coordinates of the reference path
    :param start_state: (numpy.ndarray) Start state [x, y, yaw]
    :param obstacles: (numpy.ndarray) Obstacle positions of shape (M, 2)
    :param cfg: (dict) Configuration dictionary with TEB parameters
    :param goal: (numpy.ndarray or None) Goal position [gx, gy]
    :return: (tuple) (poses_final, dt_final, plan_ms, ref_path)
    """
    N = cfg['teb_n_poses']
    t0 = time.perf_counter()

    ref_x_arr = np.asarray(ref_x, dtype=float)
    ref_y_arr = np.asarray(ref_y, dtype=float)
    n_pts = len(ref_x_arr)
    if n_pts < 2:
        return (np.array([]).reshape(0, 3), np.array([]),
                0.0, (time.perf_counter() - t0) * 1000.0)

    teb_cfg = _build_teb_config(cfg)
    ref_path = _build_ref_path(ref_x, ref_y, N, start_state)

    poses, dt, plan_ms, _, _ = optimize_teb(ref_path, start_state, obstacles, teb_cfg, goal=goal)

    if len(obstacles) > 0 and len(poses) > 0:
        dists_final = np.linalg.norm(
            poses[:, np.newaxis, :2] - obstacles[np.newaxis, :, :], axis=2)
        if np.min(dists_final) < teb_cfg.min_obstacle_dist * 0.5:
            poses = ref_path.copy()
            dp_ref = np.linalg.norm(np.diff(ref_path[:, :2], axis=0), axis=1)
            v_nom = teb_cfg.max_vel * 0.5
            dt = dp_ref / max(v_nom, 1e-9)
            dt = np.clip(dt, 0.05, 0.5)

    total_ms = (time.perf_counter() - t0) * 1000.0
    return poses, dt, total_ms, ref_path


# === Phase 4: Joint Planning Cycle ===

GLOBAL_METHODS = {
    'dijkstra': dijkstra_plan,
    'astar': astar_plan,
    'adaptive_astar': adaptive_astar_plan,
}


def _grid_obstacles_xy(grid):
    """Extract obstacle (x, y) positions from a grid map.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :return: (numpy.ndarray) Obstacle positions of shape (N, 2)
    """
    obs_y, obs_x = np.where(grid == 1)
    return (np.column_stack([obs_x, obs_y]).astype(float)
            if len(obs_x) > 0 else np.empty((0, 2)))


def run_planning_cycle(grid, start, goal, obstacles_dyn=None,
                       global_method='adaptive_astar', local_method='teb',
                       config=None):
    """Run a full global-local planning cycle with optional replanning.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param obstacles_dyn: (list or None) Dynamic obstacle positions
    :param global_method: (str) 'dijkstra', 'astar', or 'adaptive_astar'
    :param local_method: (str) 'teb' or 'dwa'
    :param config: (dict or None) Override configuration parameters
    :return: (dict) Planning results with global_path, local_trajectory, metrics
    """
    assert global_method in GLOBAL_METHODS, \
        f"global_method must be one of {list(GLOBAL_METHODS)}, got '{global_method}'"
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    local_planner = cfg['local_planner'].upper()
    assert local_planner in ('TEB', 'DWA'), \
        f"local_planner must be 'TEB' or 'DWA', got '{cfg['local_planner']}'"
    t_total_start = time.perf_counter()

    grid_inflated = inflate_obstacles(grid, cfg['inflate_radius'],
                                      protect_positions=[start, goal])

    global_fn = GLOBAL_METHODS[global_method]
    if global_method == 'astar':
        path_x, path_y, n_expanded, global_ms = global_fn(
            grid_inflated, start, goal, cfg['heuristic'])
    elif global_method == 'adaptive_astar':
        path_x, path_y, n_expanded, global_ms = global_fn(
            grid_inflated, start, goal, cfg['k_sigmoid'])
    else:
        path_x, path_y, n_expanded, global_ms = global_fn(grid_inflated, start, goal)
    if len(path_x) == 0:
        if global_method == 'astar':
            path_x, path_y, n_expanded, global_ms = global_fn(
                grid, start, goal, cfg['heuristic'])
        elif global_method == 'adaptive_astar':
            path_x, path_y, n_expanded, global_ms = global_fn(
                grid, start, goal, cfg['k_sigmoid'])
        else:
            path_x, path_y, n_expanded, global_ms = global_fn(grid, start, goal)
        grid_inflated = grid.copy()
    assert len(path_x) > 0, "Global planning failed: no path found"

    path_arr = np.column_stack([path_x, path_y]).astype(float)
    obs_check = _grid_obstacles_xy(grid)
    if len(obs_check) > 0:
        dists_g = np.linalg.norm(
            path_arr[:, np.newaxis, :] - obs_check[np.newaxis, :, :], axis=2)
        d_min_global = float(np.min(dists_g))
        if d_min_global < 0.5:
            print(f"[Warning] Global path d_min={d_min_global:.2f} cells, close to obstacles")

    replan = need_replan(start, (path_x, path_y), obstacles_dyn,
                         cfg['d_deviate'], cfg['d_obs_trigger'])
    if replan:
        if global_method == 'astar':
            path_x, path_y, n_expanded, global_ms = global_fn(
                grid_inflated, start, goal, cfg['heuristic'])
        elif global_method == 'adaptive_astar':
            path_x, path_y, n_expanded, global_ms = global_fn(
                grid_inflated, start, goal, cfg['k_sigmoid'])
        else:
            path_x, path_y, n_expanded, global_ms = global_fn(grid_inflated, start, goal)
        assert len(path_x) > 0, "Global planning failed on replan: no path found"

    gx_arr = np.asarray(path_x, dtype=float)
    gy_arr = np.asarray(path_y, dtype=float)
    global_path_len = float(np.sum(np.hypot(np.diff(gx_arr), np.diff(gy_arr))))
    n_ref_adaptive = max(20, min(50, int(global_path_len * 0.3)))
    ref_x, ref_y = extract_reference(path_x, path_y, start, n_ref_adaptive)

    start_state = np.array([float(start[0]), float(start[1]), 0.0])
    if len(ref_x) >= 2:
        start_state[2] = math.atan2(ref_y[1] - ref_y[0], ref_x[1] - ref_x[0])

    obs_static = _grid_obstacles_xy(grid_inflated)
    obs_all = (obs_static if obstacles_dyn is None
               else np.vstack([obs_static, np.asarray(obstacles_dyn, dtype=float)]))

    if local_planner == 'TEB':
        local_traj, dt_traj, local_ms, _ = teb_plan(
            ref_x, ref_y, start_state, obs_all, cfg, goal=np.array([float(goal[0]), float(goal[1])]))
    elif local_planner == 'DWA':
        dwa_cfg = DWAConfig()
        dwa_cfg.max_speed = cfg['teb_max_vel']
        dwa_cfg.max_accel = cfg['teb_max_acc']
        dwa_cfg.max_steer = math.radians(30)
        dwa_cfg.wheelbase = cfg['teb_wheelbase']
        dwa_cfg.robot_radius = cfg.get('dwa_robot_radius', 0.3)
        dwa_cfg.safety_margin = cfg.get('dwa_safety_margin', 0.2)
        dwa_cfg.dt = cfg.get('dwa_dt', 0.1)
        dwa_cfg.predict_time = cfg.get('dwa_predict_time', 2.0)
        dwa_cfg.to_goal_cost_gain = cfg.get('dwa_to_goal_cost_gain', 5.0)
        dwa_cfg.speed_cost_gain = cfg.get('dwa_speed_cost_gain', 2.0)
        dwa_cfg.obstacle_cost_gain = cfg.get('dwa_obstacle_cost_gain', 5.0)

        gx_arr = np.asarray(path_x, dtype=float)
        gy_arr = np.asarray(path_y, dtype=float)
        waypoint_step = max(1, len(gx_arr) // 20)
        wp_x = gx_arr[::waypoint_step].tolist()
        wp_y = gy_arr[::waypoint_step].tolist()
        if (wp_x[-1], wp_y[-1]) != (gx_arr[-1], gy_arr[-1]):
            wp_x.append(float(gx_arr[-1]))
            wp_y.append(float(gy_arr[-1]))

        x_dwa = np.array([float(start[0]), float(start[1]),
                          start_state[2], 0.0, 0.0])
        dwa_traj_list = [x_dwa.copy()]
        dwa_dt_list = []
        obs_for_dwa = obs_all if len(obs_all) > 0 else np.empty((0, 2))

        t_dwa_start = time.perf_counter()
        for wp_idx in range(len(wp_x)):
            goal_wp = np.array([wp_x[wp_idx], wp_y[wp_idx]])
            dist_to_wp = math.hypot(x_dwa[0] - goal_wp[0], x_dwa[1] - goal_wp[1])
            max_steps = int(dist_to_wp / (dwa_cfg.dt * dwa_cfg.max_speed) * 3) + 50
            for _ in range(max_steps):
                u, _ = dwa_control(x_dwa, dwa_cfg, goal_wp, obs_for_dwa)
                x_dwa = dwa_motion(x_dwa, u, dwa_cfg.dt, dwa_cfg.wheelbase)
                dwa_traj_list.append(x_dwa.copy())
                dwa_dt_list.append(dwa_cfg.dt)
                dist_to_wp = math.hypot(x_dwa[0] - goal_wp[0], x_dwa[1] - goal_wp[1])
                if dist_to_wp <= dwa_cfg.robot_radius + 2.0:
                    break
        local_ms = (time.perf_counter() - t_dwa_start) * 1000.0

        dwa_traj_arr = np.array(dwa_traj_list)
        local_traj = np.column_stack([
            dwa_traj_arr[:, 0], dwa_traj_arr[:, 1], dwa_traj_arr[:, 2]])
        dt_traj = np.array(dwa_dt_list)
    else:
        rx_arr, ry_arr = np.asarray(ref_x), np.asarray(ref_y)
        dp = np.diff(np.column_stack([rx_arr, ry_arr]), axis=0)
        theta_arr = np.zeros(len(rx_arr))
        if len(dp) > 0:
            theta_arr[1:] = np.arctan2(dp[:, 1], dp[:, 0])
        local_traj = np.column_stack([rx_arr, ry_arr, theta_arr])
        dt_traj = np.full(max(len(rx_arr) - 1, 1), 0.1)
        local_ms = 0.0

    if local_planner != 'TEB' and len(local_traj) > 1:
        teb_cfg_for_smooth = _build_teb_config(cfg)
        local_traj, dt_traj = smooth_acceleration(local_traj, dt_traj, teb_cfg_for_smooth)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    gx_arr = np.asarray(path_x, dtype=float)
    gy_arr = np.asarray(path_y, dtype=float)
    global_path_len = float(np.sum(np.hypot(np.diff(gx_arr), np.diff(gy_arr))))
    local_exec_time = float(np.sum(dt_traj)) if len(dt_traj) > 0 else 0.0

    metrics = {
        'global_path_length': global_path_len,
        'global_expanded': n_expanded,
        'global_time_ms': global_ms,
        'local_time_ms': local_ms,
        'local_exec_time_s': local_exec_time,
        'total_time_ms': total_ms,
        'replan_needed': replan,
    }

    return {
        'global_path': (path_x, path_y),
        'local_trajectory': local_traj,
        'metrics': metrics,
        'replan_needed': replan,
        'local_planner': local_planner,
    }


# === Phase 5: Visualization & Main ===

def main():
    """Run joint planning demo on a 50x50 random map and save results."""
    N_ROW, N_COL = 50, 50
    OBS_RATIO = 0.2

    grid = generate_random_map(N_ROW, N_COL, OBS_RATIO, seed=42)
    grid_inflated = inflate_obstacles(grid, DEFAULT_CONFIG['inflate_radius'],
                                      protect_positions=[(0, 0), (N_COL - 1, N_ROW - 1)])

    start = (0, 0)
    goal = (N_COL - 1, N_ROW - 1)

    result = run_planning_cycle(
        grid, start, goal,
        global_method='adaptive_astar', local_method='teb',
        config={'local_planner': 'TEB'})

    m = result['metrics']
    print("=== Joint Planning Results ===")
    print(f"[Global] path_length={m['global_path_length']:.2f} cells")
    print(f"[Global] expanded={m['global_expanded']}")
    print(f"[Global] time={m['global_time_ms']:.2f} ms")
    print(f"[Local]  plan_time={m['local_time_ms']:.2f} ms")
    print(f"[Local]  exec_time={m['local_exec_time_s']:.3f} s")
    print(f"[Total]  time={m['total_time_ms']:.2f} ms")
    print(f"[Replan] needed={m['replan_needed']}")

    gx, gy = result['global_path']
    traj = result['local_trajectory']

    obs_check = _grid_obstacles_xy(grid)
    if len(traj) > 0 and len(obs_check) > 0:
        dists_check = np.linalg.norm(
            traj[:, np.newaxis, :2] - obs_check[np.newaxis, :, :], axis=2)
        d_min_val = float(np.min(dists_check))
        n_collision = int(np.sum(np.min(dists_check, axis=1) < 0.5))
        print(f"[Safety] d_min={d_min_val:.2f} cells (threshold={DEFAULT_CONFIG['teb_min_obs_dist']})")
        print(f"[Safety] collision_points={n_collision}")
    elif len(traj) > 0:
        print("[Safety] no obstacles in grid")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(grid_inflated, cmap='Oranges', alpha=0.3, origin='lower')
    ax.imshow(grid, cmap='binary', alpha=0.7, origin='lower')
    obs_y, obs_x = np.where(grid == 1)
    ax.scatter(obs_x, obs_y, c='black', s=4, marker='s', label='obstacles')
    inf_y, inf_x = np.where((grid_inflated == 1) & (grid == 0))
    ax.scatter(inf_x, inf_y, c='orange', s=2, alpha=0.3, marker='s', label='inflated zone')
    ax.plot(gx, gy, '-b', linewidth=2, label='global path')
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], '-r', linewidth=2, label='local trajectory')
        obs_vis = _grid_obstacles_xy(grid)
        if len(obs_vis) > 0:
            dists = np.linalg.norm(
                traj[:, np.newaxis, :2] - obs_vis[np.newaxis, :, :], axis=2)
            min_d = float(np.min(dists))
            collision_pts = traj[np.min(dists, axis=1) < 0.5]
            if len(collision_pts) > 0:
                ax.scatter(collision_pts[:, 0], collision_pts[:, 1],
                           c='red', s=30, marker='x', zorder=6, label='collision')
            info = f'd_min={min_d:.2f} cells'
        else:
            info = 'no obstacles'
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.plot(start[0], start[1], 'og', markersize=10, label='start')
    ax.plot(goal[0], goal[1], 'xb', markersize=10, label='goal')
    ax.legend(frameon=True, fancybox=True, fontsize=8)
    ax.set_xlabel('x [cell]')
    ax.set_ylabel('y [cell]')
    local_planner_name = result.get('local_planner', 'TEB')
    ax.set_title(f'Joint Global-Local Planning (Adaptive A* + {local_planner_name})')
    fig.tight_layout()

    results_dir = pathlib.Path(__file__).parent / 'results'
    figs_dir = pathlib.Path(__file__).parent / 'figs'
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)
    save_path_csv(list(zip(gy, gx)), results_dir / 'global_path.csv')
    if len(traj) > 0:
        np.savetxt(str(results_dir / 'local_trajectory.csv'), traj,
                   delimiter=',', header='x,y,theta', comments='')
    save_metrics_json(m, results_dir / 'metrics.json')
    fig.savefig(str(figs_dir / 'joint_planning.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
