"""Animated visualization of path planning algorithms

author: Kat-yuan-eng (RuiWen Liao)
"""

import math
import random
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.grid_map import generate_random_map, inflate_obstacles

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from GlobalPlanning.AStar.a_star import AStarPlanner
from GlobalPlanning.Dijkstra.dijkstra import DijkstraPlanner
from GlobalPlanning.AdaptiveAStar.adaptive_a_star import AdaptiveAStarPlanner
from LocalPlanning.RRT.rrt import RRT
from LocalPlanning.DWA.dynamic_window_approach import (
    Config as DWAConfig, dwa_control, motion as dwa_motion,
    plot_robot as dwa_plot_robot, plot_arrow as dwa_plot_arrow,
)
from LocalPlanning.TEB.timed_elastic_band import TEBConfig, optimize_teb

show_animation = True


# === Phase 1: Global Planning Animation ===

def _setup_grid_axes(ax, grid, start, goal, title):
    """Configure grid axes with obstacles, start, and goal markers.

    :param ax: (matplotlib.axes.Axes) Axes to configure
    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param title: (str) Plot title
    """
    obs_y, obs_x = np.where(grid == 1)
    ax.plot(obs_x, obs_y, ".k", markersize=2)
    ax.plot(start[0], start[1], "og", markersize=8)
    ax.plot(goal[0], goal[1], "xb", markersize=8)
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlabel("x [cell]")
    ax.set_ylabel("y [cell]")
    ax.set_title(title)
    ax.legend(["obstacle", "start", "goal"], loc="upper right",
              frameon=True, fancybox=True)


def animate_global_planning(grid, start, goal, algorithm='adaptive_astar'):
    """Animate a single global planning algorithm on a grid.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param algorithm: (str) 'dijkstra', 'astar', or 'adaptive_astar'
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    algo_names = {
        'dijkstra': 'Dijkstra',
        'astar': 'A* (Euclidean)',
        'adaptive_astar': 'Adaptive A*',
    }
    _setup_grid_axes(ax, grid, start, goal,
                     algo_names.get(algorithm, algorithm))

    sx, sy = start
    gx, gy = goal

    if algorithm == 'dijkstra':
        planner = DijkstraPlanner(grid)
        rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elif algorithm == 'astar':
        planner = AStarPlanner(grid, heuristic='euclidean')
        rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    else:
        planner = AdaptiveAStarPlanner(grid)
        rx, ry, expanded, _ = planner.planning(sx, sy, gx, gy, heuristic_type="adaptive")

    if rx:
        ax.plot(rx, ry, "-r", linewidth=2, label="path")
        ax.legend(["obstacle", "start", "goal", "path"], loc="upper right",
                  frameon=True, fancybox=True)
        path_len = sum(math.hypot(rx[i+1] - rx[i], ry[i+1] - ry[i])
                       for i in range(len(rx) - 1))
        print(f"[{algo_names.get(algorithm, algorithm)}] "
              f"path_length={path_len:.2f}  expanded={expanded}")
    else:
        print(f"[{algo_names.get(algorithm, algorithm)}] No path found")

    plt.pause(0.001)
    plt.show()


def animate_global_comparison(grid, start, goal):
    """Animate side-by-side comparison of Dijkstra, A*, and Adaptive A*.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=100)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    sx, sy = start
    gx, gy = goal

    configs = [
        ('dijkstra', 'Dijkstra', DijkstraPlanner(grid), (sx, sy, gx, gy), {}),
        ('astar', 'A* (Euclidean)', AStarPlanner(grid, heuristic='euclidean'),
         (sx, sy, gx, gy), {}),
        ('adaptive_astar', 'Adaptive A*', AdaptiveAStarPlanner(grid),
         (sx, sy, gx, gy), {'heuristic_type': 'adaptive'}),
    ]

    for ax, (key, name, planner, args, kwargs) in zip(axes, configs):
        plt.sca(ax)
        _setup_grid_axes(ax, grid, start, goal, name)
        result = planner.planning(*args, **kwargs)
        rx, ry, expanded = result[0], result[1], result[2]
        if rx:
            ax.plot(rx, ry, "-r", linewidth=2, label="path")
            ax.legend(["obstacle", "start", "goal", "path"],
                      loc="upper right", frameon=True, fancybox=True)
            path_len = sum(math.hypot(rx[i+1] - rx[i], ry[i+1] - ry[i])
                           for i in range(len(rx) - 1))
            print(f"[{name}] path_length={path_len:.2f}  expanded={expanded}")
        else:
            print(f"[{name}] No path found")

    fig.tight_layout()
    plt.pause(0.001)
    plt.show()


# === Phase 2: RRT Animation ===

def animate_rrt(obstacle_list, start, goal, rand_area,
                expand_dis=3.0, path_resolution=0.5,
                goal_sample_rate=10, max_iter=500, robot_radius=0.8):
    """Animate RRT path planning with step-by-step tree growth.

    :param obstacle_list: (list) List of (ox, oy, radius) tuples
    :param start: (tuple) Start position (x, y)
    :param goal: (tuple) Goal position (x, y)
    :param rand_area: (tuple) (min_rand, max_rand) for sampling
    :param expand_dis: (float) Tree expansion step size
    :param path_resolution: (float) Path interpolation resolution
    :param goal_sample_rate: (int) Goal bias percentage
    :param max_iter: (int) Maximum iterations
    :param robot_radius: (float) Robot radius for collision checking
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    rrt = RRT(start=start, goal=goal,
              obstacle_list=obstacle_list,
              rand_area=rand_area,
              expand_dis=expand_dis,
              path_resolution=path_resolution,
              goal_sample_rate=goal_sample_rate,
              max_iter=max_iter,
              robot_radius=robot_radius,
              safety_margin=0.2)

    node_list = [rrt.start]
    rrt.node_list = [rrt.start]

    for i in range(max_iter):
        rnd_node = rrt.get_random_node()
        nearest_ind = rrt.get_nearest_node_index(rrt.node_list, rnd_node)
        nearest_node = rrt.node_list[nearest_ind]
        new_node = rrt.steer(nearest_node, rnd_node, expand_dis)

        if rrt.check_if_outside_play_area(new_node, rrt.play_area) and \
           rrt.check_collision(new_node, obstacle_list):
            rrt.node_list.append(new_node)

        if i % 5 == 0:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            if rnd_node is not None:
                plt.plot(rnd_node.x, rnd_node.y, "^k")

            for ox, oy, sz in obstacle_list:
                deg = list(range(0, 360, 5))
                deg.append(0)
                xl = [ox + sz * math.cos(np.deg2rad(d)) for d in deg]
                yl = [oy + sz * math.sin(np.deg2rad(d)) for d in deg]
                plt.plot(xl, yl, "-b")

            for node in rrt.node_list:
                if node.parent is not None:
                    plt.plot(node.path_x, node.path_y, "-g")

            plt.plot(start[0], start[1], "og", markersize=8)
            plt.plot(goal[0], goal[1], "xb", markersize=8)
            plt.axis("equal")
            plt.axis([rand_area[0], rand_area[1],
                      rand_area[0], rand_area[1]])
            plt.grid(True)
            plt.title(f"RRT (iter={i})")
            plt.pause(0.01)

        last_node = rrt.node_list[-1]
        if rrt.calc_dist_to_goal(last_node.x, last_node.y) <= expand_dis:
            final_node = rrt.steer(last_node, rrt.end, expand_dis)
            if rrt.check_collision(final_node, obstacle_list):
                rrt.node_list.append(final_node)
                path = rrt.generate_final_course(len(rrt.node_list) - 1)

                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]

                plt.cla()
                for ox, oy, sz in obstacle_list:
                    deg = list(range(0, 360, 5))
                    deg.append(0)
                    xl = [ox + sz * math.cos(np.deg2rad(d)) for d in deg]
                    yl = [oy + sz * math.sin(np.deg2rad(d)) for d in deg]
                    plt.plot(xl, yl, "-b")
                for node in rrt.node_list:
                    if node.parent is not None:
                        plt.plot(node.path_x, node.path_y, "-g")
                plt.plot(path_x, path_y, "-r", linewidth=2, label="path")
                plt.plot(start[0], start[1], "og", markersize=8)
                plt.plot(goal[0], goal[1], "xb", markersize=8)
                plt.axis("equal")
                plt.axis([rand_area[0], rand_area[1],
                          rand_area[0], rand_area[1]])
                plt.grid(True)
                plt.title("RRT - Path Found")
                plt.legend(frameon=True, fancybox=True)

                path_len = sum(math.hypot(path_x[i+1] - path_x[i],
                                          path_y[i+1] - path_y[i])
                               for i in range(len(path_x) - 1))
                print(f"[RRT] path_length={path_len:.2f}  "
                      f"nodes={len(rrt.node_list)}  iter={i}")
                plt.pause(0.001)
                plt.show()
                return

    print("[RRT] No path found within max_iter")
    plt.pause(0.001)
    plt.show()


# === Phase 3: DWA Animation ===

def animate_dwa(obstacle_list, start, goal,
                max_speed=2.5, min_speed=-0.5,
                max_yaw_rate=40.0 * math.pi / 180.0,
                max_accel=0.5, max_delta_yaw_rate=40.0 * math.pi / 180.0,
                v_res=0.01, yaw_rate_res=0.1 * math.pi / 180.0,
                config_dt=0.1, predict_time=3.0,
                to_goal_gain=0.15, speed_gain=1.0, obs_gain=1.0,
                stuck_flag_cons=0.001, robot_radius=0.3):
    """Animate DWA local planning with real-time trajectory prediction.

    :param obstacle_list: (list/array) Obstacle positions
    :param start: (tuple) Start position (x, y)
    :param goal: (tuple) Goal position (x, y)
    :param max_speed: (float) Maximum linear speed
    :param min_speed: (float) Minimum linear speed
    :param max_yaw_rate: (float) Maximum yaw rate
    :param max_accel: (float) Maximum acceleration
    :param max_delta_yaw_rate: (float) Maximum yaw rate change
    :param v_res: (float) Velocity resolution
    :param yaw_rate_res: (float) Yaw rate resolution
    :param config_dt: (float) Time step
    :param predict_time: (float) Trajectory prediction horizon
    :param to_goal_gain: (float) Goal cost weight
    :param speed_gain: (float) Speed cost weight
    :param obs_gain: (float) Obstacle cost weight
    :param stuck_flag_cons: (float) Stuck detection threshold
    :param robot_radius: (float) Robot radius
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    dwa_config = DWAConfig()
    dwa_config.max_speed = max_speed
    dwa_config.min_speed = min_speed
    dwa_config.max_steer = max_yaw_rate
    dwa_config.max_accel = max_accel
    dwa_config.max_steer_rate = max_delta_yaw_rate
    dwa_config.v_resolution = v_res
    dwa_config.steer_resolution = yaw_rate_res
    dwa_config.dt = config_dt
    dwa_config.predict_time = predict_time
    dwa_config.to_goal_cost_gain = to_goal_gain
    dwa_config.speed_cost_gain = speed_gain
    dwa_config.obstacle_cost_gain = obs_gain
    dwa_config.robot_stuck_flag_cons = stuck_flag_cons
    dwa_config.robot_radius = robot_radius

    ob = np.asarray(obstacle_list, dtype=float)
    x = np.array([start[0], start[1], math.pi / 8.0, 0.0, 0.0])
    goal_arr = np.array([goal[0], goal[1]])
    trajectory = np.array(x)

    step = 0
    while True:
        u, pred_traj = dwa_control(x, dwa_config, goal_arr, ob)
        x = dwa_motion(x, u, dwa_config.dt, dwa_config.wheelbase)
        trajectory = np.vstack((trajectory, x))
        step += 1

        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        plt.plot(pred_traj[:, 0], pred_traj[:, 1], "-g", label="predicted")
        plt.plot(x[0], x[1], "xr", markersize=8)
        plt.plot(goal_arr[0], goal_arr[1], "xb", markersize=10)
        plt.plot(ob[:, 0], ob[:, 1], "ok", markersize=6)

        dwa_plot_robot(x[0], x[1], x[2], dwa_config)
        dwa_plot_arrow(x[0], x[1], x[2])

        plt.axis("equal")
        plt.grid(True)
        plt.title(f"DWA (step={step})")
        plt.legend(frameon=True, fancybox=True, loc="upper right")
        plt.pause(0.0001)

        if math.hypot(x[0] - goal_arr[0], x[1] - goal_arr[1]) <= robot_radius:
            print("[DWA] Goal reached!")
            break
        if step > 5000:
            print("[DWA] Max steps exceeded")
            break

    plt.cla()
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r", linewidth=2,
             label="trajectory")
    plt.plot(ob[:, 0], ob[:, 1], "ok", markersize=6)
    plt.plot(start[0], start[1], "og", markersize=10, label="start")
    plt.plot(goal_arr[0], goal_arr[1], "xb", markersize=10, label="goal")
    plt.axis("equal")
    plt.grid(True)
    plt.title("DWA - Goal Reached")
    plt.legend(frameon=True, fancybox=True)
    plt.pause(0.001)
    plt.show()


# === Phase 4: Joint Planning Animation ===

def animate_joint_planning(grid, start, goal, n_ref=20,
                           teb_n_poses=20, teb_max_vel=2.5,
                           teb_max_acc=2.0, teb_min_obs_dist=0.3,
                           teb_wheelbase=0.3, teb_n_opt_iter=5,
                           teb_w_path=1.0, teb_w_obs=10.0,
                           teb_w_vel=1.0, teb_w_kin=100.0,
                           teb_w_time=1.0):
    """Animate joint global (A*) + local (TEB) planning pipeline.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sx, sy) start position
    :param goal: (tuple) (gx, gy) goal position
    :param n_ref: (int) Number of reference path points for TEB
    :param teb_n_poses: (int) TEB pose count
    :param teb_max_vel: (float) TEB max velocity
    :param teb_max_acc: (float) TEB max acceleration
    :param teb_min_obs_dist: (float) TEB minimum obstacle distance
    :param teb_wheelbase: (float) TEB wheelbase
    :param teb_n_opt_iter: (int) TEB optimization iterations
    :param teb_w_path: (float) TEB path weight
    :param teb_w_obs: (float) TEB obstacle weight
    :param teb_w_vel: (float) TEB velocity weight
    :param teb_w_kin: (float) TEB kinematics weight
    :param teb_w_time: (float) TEB time weight
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    obs_y, obs_x = np.where(grid == 1)
    ax.plot(obs_x, obs_y, ".k", markersize=2)
    ax.plot(start[0], start[1], "og", markersize=10, label="start")
    ax.plot(goal[0], goal[1], "xb", markersize=10, label="goal")
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlabel("x [cell]")
    ax.set_ylabel("y [cell]")
    ax.set_title("Phase 1: Global A* Search")
    ax.legend(frameon=True, fancybox=True)

    sx, sy = start
    gx, gy = goal

    print("[Phase 1] Running global A* search...")
    planner = AStarPlanner(grid, heuristic='euclidean')
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)

    if not rx:
        print("[Joint] Global planning failed")
        plt.show()
        return

    ax.plot(rx, ry, "-b", linewidth=2, label="global path")
    ax.legend(frameon=True, fancybox=True)
    plt.pause(0.5)

    print("[Phase 2] Running local TEB optimization...")
    ax.set_title("Phase 2: Local TEB Optimization")

    px = np.asarray(rx, dtype=float)
    py = np.asarray(ry, dtype=float)
    dists = (px - start[0]) ** 2 + (py - start[1]) ** 2
    idx_closest = int(np.argmin(dists))
    idx_end = min(idx_closest + n_ref, len(px))
    ref_x = px[idx_closest:idx_end]
    ref_y = py[idx_closest:idx_end]

    start_state = np.array([float(start[0]), float(start[1]), 0.0])
    if len(ref_x) >= 2:
        start_state[2] = math.atan2(ref_y[1] - ref_y[0],
                                     ref_x[1] - ref_x[0])

    obs_static = (np.column_stack([obs_x, obs_y]).astype(float)
                  if len(obs_x) > 0 else np.empty((0, 2)))

    teb_config = TEBConfig()
    teb_config.n_poses = teb_n_poses
    teb_config.max_vel = teb_max_vel
    teb_config.max_acc = teb_max_acc
    teb_config.min_obstacle_dist = teb_min_obs_dist
    teb_config.wheelbase = teb_wheelbase
    teb_config.n_opt_iter = teb_n_opt_iter
    teb_config.weight_path = teb_w_path
    teb_config.weight_obstacle = teb_w_obs
    teb_config.weight_vel = teb_w_vel
    teb_config.weight_kin = teb_w_kin
    teb_config.weight_time = teb_w_time

    n_pts = teb_config.n_poses
    if len(ref_x) >= 2:
        t_wp = np.linspace(0, 1, len(ref_x))
        t_fine = np.linspace(0, 1, n_pts)
        interp_x = np.interp(t_fine, t_wp, ref_x)
        interp_y = np.interp(t_fine, t_wp, ref_y)
        interp_beta = np.arctan2(np.gradient(interp_y), np.gradient(interp_x))
        ref_path = np.column_stack([interp_x, interp_y, interp_beta])
    else:
        ref_path = np.column_stack([ref_x, ref_y, np.zeros(len(ref_x))])

    poses, dt, _, _, _ = optimize_teb(ref_path, start_state, obs_static, teb_config)

    if len(poses) > 0:
        ax.plot(poses[:, 0], poses[:, 1], "-r", linewidth=2,
                label="TEB trajectory")
        ax.legend(frameon=True, fancybox=True)

        for i in range(0, len(poses), max(1, len(poses) // 8)):
            circle = plt.Circle((poses[i, 0], poses[i, 1]), 0.4,
                                color="green", alpha=0.5)
            ax.add_patch(circle)
            out_x = poses[i, 0] + np.cos(poses[i, 2]) * 0.4
            out_y = poses[i, 1] + np.sin(poses[i, 2]) * 0.4
            ax.plot([poses[i, 0], out_x], [poses[i, 1], out_y], "-k",
                    linewidth=1)
            plt.pause(0.05)

        ax.set_title("Joint Planning: A* + TEB")
        print(f"[Joint] global_expanded={expanded}  "
              f"teb_poses={len(poses)}")

    plt.pause(0.001)
    plt.show()


# === Phase 5: Main ===

def main():
    """Run interactive animation demo with algorithm selection."""
    N_ROW, N_COL = 50, 50
    OBS_RATIO = 0.2

    grid = generate_random_map(N_ROW, N_COL, OBS_RATIO, seed=42)

    start = (0, 0)
    goal = (N_COL - 1, N_ROW - 1)
    grid = inflate_obstacles(grid, radius=1, protect_positions=[start, goal])

    assert grid[start[1], start[0]] == 0, "Start blocked after inflation"
    assert grid[goal[1], goal[0]] == 0, "Goal blocked after inflation"

    obstacle_list_rrt = [
        (5, 5, 3), (10, 15, 3), (20, 10, 4), (25, 25, 4),
        (35, 20, 3), (30, 35, 3), (15, 30, 3), (40, 40, 3),
    ]

    obstacle_list_dwa = np.array([
        [-1, -1], [0, 2], [4.0, 2.0], [5.0, 4.0], [5.0, 5.0],
        [5.0, 6.0], [5.0, 9.0], [8.0, 9.0], [7.0, 9.0],
        [8.0, 10.0], [9.0, 11.0], [12.0, 13.0], [12.0, 12.0],
        [15.0, 15.0], [13.0, 13.0],
    ])

    print("=== Path Planning Animation Demo ===")
    print("1: Global planning (Adaptive A*)")
    print("2: Global comparison (Dijkstra / A* / Adaptive A*)")
    print("3: RRT animation")
    print("4: DWA animation")
    print("5: Joint planning (A* + TEB)")
    print("0: Run global comparison (default)")

    choice = input("Select animation [0-5]: ").strip()
    choice = choice or "0"

    if choice == "1":
        animate_global_planning(grid, start, goal, 'adaptive_astar')
    elif choice == "2":
        animate_global_comparison(grid, start, goal)
    elif choice == "3":
        animate_rrt(obstacle_list_rrt, (5, 5), (50, 50), (0, 60))
    elif choice == "4":
        animate_dwa(obstacle_list_dwa, (0.0, 0.0), (10.0, 10.0))
    elif choice == "5":
        animate_joint_planning(grid, start, goal)
    else:
        animate_global_comparison(grid, start, goal)


if __name__ == '__main__':
    main()
