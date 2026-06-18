"""
Timed Elastic Band (TEB) local path planning with L-BFGS-B optimization

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [TEB Algorithm](https://www.ri.cmu.edu/pub_files/2013/4/ROB-2013-01.pdf)
"""

import math
import sys
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
from utils.plot import plot_robot

show_animation = True


# === Phase 1: Configuration ===

class TEBConfig:
    """
    Configuration parameters for TEB planner.
    """
    n_poses = 20
    dt_ref = 0.3  # [s]
    max_vel = 2.5  # [m/s]
    max_acc = 2.0  # [m/s^2]
    max_steer = np.deg2rad(30)  # [rad]
    max_steer_rate = np.deg2rad(100)  # [rad/s]
    wheelbase = 0.3  # [m]
    robot_radius = 0.5  # [m]
    min_obstacle_dist = 0.8  # [m]
    weight_path = 1.0
    weight_obstacle = 10.0
    weight_vel = 1.0
    weight_acc = 5.0
    weight_curv = 2.0
    weight_kin = 10.0
    weight_time = 1.0
    n_opt_iter = 50


# === Phase 2: Cost Functions ===

def penalty(x):
    """
    Quadratic penalty for positive values, zero otherwise.

    :param x: (numpy.ndarray) Input array
    :return: (numpy.ndarray) Penalty values
    """
    return np.where(x > 0, x ** 2, 0.0)


def teb_objective(decision_vars, ref_path, obstacles, config):
    """
    TEB multi-objective cost function combining path, obstacle, velocity, acceleration, curvature, kinematics, and time.

    :param decision_vars: (numpy.ndarray) Concatenated poses (3N) and time steps (N-1)
    :param ref_path: (numpy.ndarray) Reference path of shape (N, 3)
    :param obstacles: (numpy.ndarray) Obstacle positions of shape (M, 2)
    :param config: (TEBConfig) TEB configuration
    :return: (float) Total weighted cost
    """
    N = config.n_poses
    poses = decision_vars[:3 * N].reshape(N, 3)
    dt = decision_vars[3 * N:]

    diff_path = poses - ref_path
    f_path = np.sum(diff_path ** 2)

    pos = poses[:, :2]
    obs_cutoff = config.min_obstacle_dist * 3.0
    dists = np.linalg.norm(
        pos[:, np.newaxis, :] - obstacles[np.newaxis, :, :], axis=2
    )
    near_mask = dists < obs_cutoff
    n_near_obs = np.sum(near_mask)
    decay_scale = max(config.min_obstacle_dist * 2.0, 1.0) * (1.0 + 0.1 * min(n_near_obs, 20))
    f_obs = np.sum(penalty(config.min_obstacle_dist - dists) * near_mask) + 0.2 * np.sum(np.exp(-dists / decay_scale) * near_mask)

    dp = poses[1:, :2] - poses[:-1, :2]
    v = np.linalg.norm(dp, axis=1) / np.maximum(dt, 1e-9)
    f_vel = np.sum(penalty(np.abs(v) - config.max_vel))

    acc = np.diff(v) / np.maximum(dt[:-1], 1e-9)
    f_acc = np.sum(penalty(np.abs(acc) - config.max_acc))

    dbeta = poses[1:, 2] - poses[:-1, 2]
    dp_norm = np.linalg.norm(dp, axis=1)
    kappa = 2.0 * np.sin(dbeta / 2.0) / np.maximum(dp_norm, 1e-6)
    dkappa = np.diff(kappa)
    f_curv = np.sum(dkappa ** 2)

    delta = np.arctan(config.wheelbase * kappa)
    kin_residual = dbeta - v * np.tan(delta) / config.wheelbase * dt
    f_kin = np.sum(penalty(np.abs(kin_residual) - 0.01))

    f_time = np.sum(dt) ** 2

    return (config.weight_path * f_path
            + config.weight_obstacle * f_obs
            + config.weight_vel * f_vel
            + config.weight_acc * f_acc
            + config.weight_curv * f_curv
            + config.weight_kin * f_kin
            + config.weight_time * f_time)


# === Phase 3: Optimization ===

def optimize_teb(ref_path, start_state, obstacles, config, goal=None):
    """
    Optimize TEB trajectory using L-BFGS-B with fallback on failure.

    :param ref_path: (numpy.ndarray) Reference path of shape (N, 3)
    :param start_state: (numpy.ndarray) Start state [x, y, yaw]
    :param obstacles: (numpy.ndarray) Obstacle positions of shape (M, 2)
    :param config: (TEBConfig) TEB configuration
    :param goal: (numpy.ndarray or None) Goal position [gx, gy] for endpoint constraint
    :return: (tuple) (poses, dt, plan_time_ms, traj_history, n_iterations)
    """
    N = config.n_poses
    t0 = time.perf_counter()

    poses_init = ref_path.copy()
    poses_init[0] = start_state

    dp = np.linalg.norm(np.diff(ref_path[:, :2], axis=0), axis=1)
    v_nominal = config.max_vel * 0.5
    dt_init = dp / np.maximum(v_nominal, 1e-9)
    dt_init = np.clip(dt_init, 0.05, 2.0)

    decision_vars = np.concatenate([poses_init.flatten(), dt_init])

    pose_bounds = []
    goal_near_ref_end = (goal is not None
                         and np.hypot(goal[0] - ref_path[-1, 0], goal[1] - ref_path[-1, 1]) < 6.0)
    for i in range(N):
        if i == 0:
            pose_bounds.extend([
                (start_state[0], start_state[0]),
                (start_state[1], start_state[1]),
                (start_state[2], start_state[2]),
            ])
        elif i == N - 1 and goal_near_ref_end:
            goal_yaw = math.atan2(goal[1] - ref_path[-2, 1], goal[0] - ref_path[-2, 0])
            pose_bounds.extend([
                (goal[0], goal[0]),
                (goal[1], goal[1]),
                (goal_yaw, goal_yaw),
            ])
        else:
            pose_bounds.extend([
                (None, None),
                (None, None),
                (-np.pi, np.pi),
            ])
    dt_bounds = [(0.01, 2.0)] * (N - 1)
    bounds = pose_bounds + dt_bounds

    traj_history = []

    def callback(xk):
        poses_k = xk[:3 * N].reshape(N, 3)
        traj_history.append(poses_k.copy())

    result = minimize(
        teb_objective,
        decision_vars,
        args=(ref_path, obstacles, config),
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={'maxiter': config.n_opt_iter, 'ftol': 1e-4},
    )

    if not result.success and result.fun > 1e10:
        poses_init = ref_path.copy()
        poses_init[0] = start_state
        dp = np.linalg.norm(np.diff(ref_path[:, :2], axis=0), axis=1)
        dt_fallback = dp / max(config.max_vel * 0.5, 1e-9)
        dt_fallback = np.clip(dt_fallback, 0.05, 2.0)
        plan_time_ms = (time.perf_counter() - t0) * 1000.0
        return poses_init, dt_fallback, plan_time_ms, traj_history, 0

    poses_opt = result.x[:3 * N].reshape(N, 3)
    dt_opt = result.x[3 * N:]

    poses_smooth, dt_smooth = smooth_acceleration(poses_opt, dt_opt, config, obstacles)

    if goal_near_ref_end:
        poses_smooth[-1, 0] = goal[0]
        poses_smooth[-1, 1] = goal[1]
        goal_yaw = math.atan2(goal[1] - poses_smooth[-2, 1], goal[0] - poses_smooth[-2, 0])
        poses_smooth[-1, 2] = goal_yaw

    plan_time_ms = (time.perf_counter() - t0) * 1000.0

    return poses_smooth, dt_smooth, plan_time_ms, traj_history, result.nit


# === Phase 4: Post-processing ===

def smooth_acceleration(poses, dt, config, obstacles=None):
    """
    Post-process trajectory to enforce acceleration and steering rate limits.

    :param poses: (numpy.ndarray) Optimized poses of shape (N, 3)
    :param dt: (numpy.ndarray) Time steps of shape (N-1,)
    :param config: (TEBConfig) TEB configuration
    :param obstacles: (numpy.ndarray or None) Obstacle positions for safety check
    :return: (tuple) (poses_smooth, dt) Smoothed poses and time steps
    """
    dp = poses[1:, :2] - poses[:-1, :2]
    v = np.linalg.norm(dp, axis=1) / np.maximum(dt, 1e-9)

    dv = np.diff(v)
    dv_max = config.max_acc * dt[:-1]
    dv_clipped = np.clip(dv, -dv_max, dv_max)

    v_smooth = np.empty_like(v)
    v_smooth[0] = v[0]
    v_smooth[1:] = v[0] + np.cumsum(dv_clipped)

    dbeta = poses[1:, 2] - poses[:-1, 2]
    dp_norm = np.linalg.norm(dp, axis=1)
    kappa = 2.0 * np.sin(dbeta / 2.0) / np.maximum(dp_norm, 1e-6)
    omega = v_smooth * kappa

    d_omega = np.diff(omega)
    max_omega_rate = config.max_vel * np.tan(config.max_steer) / max(config.wheelbase, 1e-9)
    d_omega_max = max_omega_rate * dt[:-1]
    d_omega_clipped = np.clip(d_omega, -d_omega_max, d_omega_max)

    omega_smooth = np.empty_like(omega)
    omega_smooth[0] = omega[0]
    omega_smooth[1:] = omega[0] + np.cumsum(d_omega_clipped)

    kappa_smooth = omega_smooth / np.maximum(v_smooth, 1e-9)
    dbeta_smooth = 2.0 * np.arcsin(np.clip(kappa_smooth * dp_norm / 2.0, -1.0, 1.0))
    headings_smooth = np.empty(len(poses))
    headings_smooth[0] = poses[0, 2]
    headings_smooth[1:] = poses[0, 2] + np.cumsum(dbeta_smooth)

    dx = v_smooth * np.cos(headings_smooth[:-1]) * dt
    dy = v_smooth * np.sin(headings_smooth[:-1]) * dt

    poses_smooth = poses.copy()
    poses_smooth[1:, 0] = poses_smooth[0, 0] + np.cumsum(dx)
    poses_smooth[1:, 1] = poses_smooth[0, 1] + np.cumsum(dy)

    dp_smooth = poses_smooth[1:, :2] - poses_smooth[:-1, :2]
    poses_smooth[1:, 2] = np.arctan2(dp_smooth[:, 1], dp_smooth[:, 0])

    if obstacles is not None and len(obstacles) > 0:
        pos_smooth = poses_smooth[:, :2]
        dists_smooth = np.linalg.norm(
            pos_smooth[:, np.newaxis, :] - obstacles[np.newaxis, :, :], axis=2
        )
        if np.min(dists_smooth) < config.min_obstacle_dist * 0.8:
            return poses, dt

    return poses_smooth, dt


# === Phase 5: Visualization and Main ===

def main():
    """
    Run TEB planner demo with optimization convergence visualization.
    """
    config = TEBConfig()

    t = np.linspace(0, 1, config.n_poses)
    ref_x = t * 6.0
    ref_y = 0.5 * np.sin(t * np.pi) + t * 2.0
    ref_beta = np.arctan2(np.gradient(ref_y), np.gradient(ref_x))
    ref_path = np.column_stack([ref_x, ref_y, ref_beta])

    obstacles = np.array([
        [2.0, 1.5],
        [3.5, 2.8],
        [5.0, 3.2],
    ])

    start_state = ref_path[0].copy()

    poses, dt, plan_time, traj_history, n_iter = optimize_teb(
        ref_path, start_state, obstacles, config
    )

    dp = poses[1:, :2] - poses[:-1, :2]
    v = np.linalg.norm(dp, axis=1) / np.maximum(dt, 1e-9)
    t_cum = np.concatenate([[0.0], np.cumsum(dt)])

    print(f"[TEB] plan_time={plan_time:.2f} ms")
    print(f"[TEB] T_exec={np.sum(dt):.3f} s")
    print(f"[TEB] v_max={np.max(np.abs(v)):.3f} m/s (limit={config.max_vel})")

    dists = np.linalg.norm(
        poses[:, np.newaxis, :2] - obstacles[np.newaxis, :, :], axis=2
    )
    min_dist = np.min(dists)
    print(f"[TEB] d_min={min_dist:.3f} m (safety={config.min_obstacle_dist})")
    print(f"[TEB] constraint_satisfied={min_dist >= config.min_obstacle_dist}")

    if not show_animation:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def on_key(event):
        if event.key == 'escape':
            plt.close()
            sys.exit()

    fig.canvas.mpl_connect('key_press_event', on_key)

    ax1.plot(ref_path[:, 0], ref_path[:, 1], 'b--', linewidth=1.5, label='Reference')
    ax1.plot(poses[:, 0], poses[:, 1], 'r-', linewidth=2, label='TEB trajectory')

    for obs in obstacles:
        circle = plt.Circle(obs, config.min_obstacle_dist, color='orange', alpha=0.4)
        ax1.add_patch(circle)
        ax1.plot(obs[0], obs[1], 'xk', markersize=10)

    plt.sca(ax1)
    plot_robot(poses[0, 0], poses[0, 1], poses[0, 2], color='blue')

    for i in range(1, len(poses), 3):
        plot_robot(poses[i, 0], poses[i, 1], poses[i, 2], color='green')

    ax1.set_aspect('equal')
    ax1.legend(frameon=True, fancybox=True)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('TEB Trajectory')

    ax2.plot(t_cum[:-1], v, 'r-', linewidth=1.5, label='velocity')
    ax2.axhline(y=config.max_vel, color='k', linestyle='--', label='v_max')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('velocity [m/s]')
    ax2.set_title('Velocity Profile')
    ax2.legend(frameon=True, fancybox=True)

    fig.tight_layout()

    if len(traj_history) > 1:
        fig2, ax3 = plt.subplots(figsize=(10, 8))
        fig2.canvas.mpl_connect('key_press_event', on_key)

        ax3.plot(ref_path[:, 0], ref_path[:, 1], 'b--', linewidth=1.5, label='Reference')

        for obs in obstacles:
            circle = plt.Circle(obs, config.min_obstacle_dist, color='orange', alpha=0.4)
            ax3.add_patch(circle)
            ax3.plot(obs[0], obs[1], 'xk', markersize=10)

        colors = plt.cm.viridis(np.linspace(0, 1, len(traj_history)))
        step = max(1, len(traj_history) // 5)
        for idx, traj in enumerate(traj_history):
            label = f'iter {idx}' if idx % step == 0 else None
            ax3.plot(traj[:, 0], traj[:, 1], color=colors[idx],
                     linewidth=1, alpha=0.6, label=label)

        ax3.plot(poses[:, 0], poses[:, 1], 'r-', linewidth=2, label='Final')
        ax3.set_aspect('equal')
        ax3.legend(frameon=True, fancybox=True, fontsize=8)
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('y [m]')
        ax3.set_title('TEB Optimization Convergence')
        fig2.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
