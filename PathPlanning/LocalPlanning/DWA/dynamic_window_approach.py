"""
Dynamic Window Approach for local path planning with Ackermann bicycle model

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [Dynamic Window Approach](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_2/fox_dieter_1997_2.pdf)
"""

import math

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class Config:
    """
    Configuration parameters for DWA planner with Ackermann bicycle model.
    """

    def __init__(self):
        """
        Initialize DWA configuration with default parameters.
        """
        self.max_speed = 5.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_accel = 2.0  # [m/s^2]
        self.max_steer = math.radians(30)  # [rad]
        self.max_steer_rate = math.radians(100)  # [rad/s]
        self.wheelbase = 3.0  # [m]
        self.v_resolution = 0.2  # [m/s]
        self.steer_resolution = math.radians(5)  # [rad]
        self.dt = 0.2  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 5.0
        self.speed_cost_gain = 5.0
        self.obstacle_cost_gain = 5.0
        self.robot_stuck_flag_cons = 0.01
        self.robot_radius = 1.0  # [m]
        self.safety_margin = 0.5  # [m]


def motion(x, u, dt, wheelbase):
    """
    Compute next state using Ackermann bicycle kinematic model.

    :param x: (numpy.ndarray) Current state [x, y, yaw, v, steer]
    :param u: (list) Control input [v, steer]
    :param dt: (float) Time step
    :param wheelbase: (float) Wheelbase length
    :return: (numpy.ndarray) Next state [x, y, yaw, v, steer]
    """
    yaw = x[2] + u[0] * math.tan(u[1]) / wheelbase * dt
    px = x[0] + u[0] * math.cos(yaw) * dt
    py = x[1] + u[0] * math.sin(yaw) * dt
    return np.array([px, py, yaw, u[0], u[1]])


def calc_dynamic_window(x, config):
    """
    Calculate dynamic window based on velocity/steering limits and current state.

    :param x: (numpy.ndarray) Current state [x, y, yaw, v, steer]
    :param config: (Config) DWA configuration
    :return: (list) Dynamic window [v_min, v_max, steer_min, steer_max]
    """
    Vs = [config.min_speed, config.max_speed,
          -config.max_steer, config.max_steer]

    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_steer_rate * config.dt,
          x[4] + config.max_steer_rate * config.dt]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def calc_trajectory(x_init, v, steer, config):
    """
    Simulate trajectory for given velocity and steering over prediction time.

    :param x_init: (numpy.ndarray) Initial state [x, y, yaw, v, steer]
    :param v: (float) Velocity
    :param steer: (float) Steering angle
    :param config: (Config) DWA configuration
    :return: (numpy.ndarray) Trajectory array of shape (N, 5)
    """
    x = np.array(x_init)
    traj_list = [x.copy()]
    time = 0.0
    while time <= config.predict_time:
        x = motion(x, [v, steer], config.dt, config.wheelbase)
        traj_list.append(x.copy())
        time += config.dt
    return np.array(traj_list)


def calc_to_goal_cost(trajectory, goal):
    """
    Calculate cost based on heading angle and distance to goal.

    :param trajectory: (numpy.ndarray) Predicted trajectory
    :param goal: (numpy.ndarray) Goal position [gx, gy]
    :return: (float) Goal cost
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    angle_cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    dist_cost = math.hypot(dx, dy)
    return angle_cost + 0.5 * dist_cost


def calc_obstacle_cost(trajectory, ob, config):
    """
    Calculate obstacle proximity cost with pre-filtering for efficiency.

    :param trajectory: (numpy.ndarray) Predicted trajectory
    :param ob: (numpy.ndarray) Obstacles, shape (N, 2) or (N, 3) with radius
    :param config: (Config) DWA configuration
    :return: (float) Obstacle cost (inf if collision)
    """
    safety_dist = config.robot_radius + config.safety_margin
    if len(ob) == 0:
        return 0.0
    if ob.shape[1] == 3:
        ox = ob[:, 0]
        oy = ob[:, 1]
        cr = ob[:, 2]
        bubble_r = float(np.max(cr)) + safety_dist + config.predict_time * config.max_speed
        traj_mid_x = (trajectory[0, 0] + trajectory[-1, 0]) * 0.5
        traj_mid_y = (trajectory[0, 1] + trajectory[-1, 1]) * 0.5
        traj_span = math.hypot(trajectory[-1, 0] - trajectory[0, 0],
                               trajectory[-1, 1] - trajectory[0, 1]) + config.predict_time * config.max_speed
        pre_mask = np.hypot(ox - traj_mid_x, oy - traj_mid_y) < traj_span + bubble_r
        if not np.any(pre_mask):
            return 0.0
        ob_near = ob[pre_mask]
        cr_n = ob_near[:, 2]
        dx = trajectory[:, 0][:, None] - ob_near[:, 0][None, :]
        dy = trajectory[:, 1][:, None] - ob_near[:, 1][None, :]
        surface_dist = np.hypot(dx, dy) - cr_n[None, :]
        min_r = np.min(surface_dist)
    else:
        ox = ob[:, 0]
        oy = ob[:, 1]
        bubble_r = safety_dist + 1.0
        traj_mid_x = (trajectory[0, 0] + trajectory[-1, 0]) * 0.5
        traj_mid_y = (trajectory[0, 1] + trajectory[-1, 1]) * 0.5
        traj_span = math.hypot(trajectory[-1, 0] - trajectory[0, 0],
                               trajectory[-1, 1] - trajectory[0, 1]) + config.predict_time * config.max_speed
        pre_mask = np.hypot(ox - traj_mid_x, oy - traj_mid_y) < traj_span + bubble_r
        if not np.any(pre_mask):
            return 0.0
        ob_near = ob[pre_mask]
        dx = trajectory[:, 0][:, None] - ob_near[:, 0][None, :]
        dy = trajectory[:, 1][:, None] - ob_near[:, 1][None, :]
        r = np.hypot(dx, dy)
        min_r = np.min(r)

    if min_r < safety_dist:
        return float('inf')
    elif min_r < safety_dist * 2.0:
        return (safety_dist * 2.0 - min_r) ** 2 / max(min_r, 1e-9)
    else:
        return safety_dist / (min_r ** 2 + 1e-6)


def dwa_control(x, config, goal, ob):
    """
    Select optimal control input by evaluating all candidate trajectories.

    :param x: (numpy.ndarray) Current state [x, y, yaw, v, steer]
    :param config: (Config) DWA configuration
    :param goal: (numpy.ndarray) Goal position [gx, gy]
    :param ob: (numpy.ndarray) Obstacles
    :return: (tuple) (best_u, best_trajectory) optimal control and trajectory
    """
    dw = calc_dynamic_window(x, config)

    x_init = x
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for s in np.arange(dw[2], dw[3], config.steer_resolution):
            trajectory = calc_trajectory(x_init, v, s, config)

            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            if trajectory[-1, 3] < 0:
                speed_cost *= 3.0
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            if min_cost > final_cost:
                min_cost = final_cost
                best_u = [v, s]
                best_trajectory = trajectory

    if min_cost == float("inf"):
        goal_angle = math.atan2(goal[1] - x[1], goal[0] - x[0])
        angle_diff = math.atan2(math.sin(goal_angle - x[2]),
                                math.cos(goal_angle - x[2]))
        best_u[0] = config.min_speed
        best_u[1] = config.max_steer if angle_diff > 0 else -config.max_steer
    elif abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        goal_angle = math.atan2(goal[1] - x[1], goal[0] - x[0])
        angle_diff = math.atan2(math.sin(goal_angle - x[2]),
                                math.cos(goal_angle - x[2]))
        best_u[0] = config.min_speed
        best_u[1] = config.max_steer if angle_diff > 0 else -config.max_steer

    return best_u, best_trajectory


def plot_arrow(x, y, yaw, length=2.0, width=0.3):
    """
    Plot an arrow indicating robot heading.

    :param x: (float) X position
    :param y: (float) Y position
    :param yaw: (float) Heading angle in radians
    :param length: (float) Arrow length
    :param width: (float) Arrow head width
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):
    """
    Plot robot as a circle with heading indicator.

    :param x: (float) X position
    :param y: (float) Y position
    :param yaw: (float) Heading angle in radians
    :param config: (Config) DWA configuration
    """
    circle = plt.Circle((x, y), config.robot_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=50.0, gy=50.0):
    """
    Run DWA planner demo with waypoint navigation and circular obstacles.
    """
    print(__file__ + " start!!")

    x = np.array([0.0, 0.0, math.pi / 4.0, 0.0, 0.0])
    goal = np.array([gx, gy])
    config = Config()

    obstacle_circles = np.array([
        [15, 8, 3], [10, 20, 3], [25, 12, 4],
        [30, 28, 4], [40, 18, 3], [35, 38, 3],
        [20, 42, 3], [45, 42, 3],
    ])
    ob = obstacle_circles

    waypoints = [
        np.array([5.0, 8.0]),
        np.array([3.0, 15.0]),
        np.array([3.0, 25.0]),
        np.array([10.0, 32.0]),
        np.array([20.0, 35.0]),
        np.array([28.0, 42.0]),
        np.array([40.0, 45.0]),
        np.array([gx, gy]),
    ]

    traj_list = [x.copy()]
    total_step = 0

    if show_animation:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    for wp_idx, wp_goal in enumerate(waypoints):
        print(f"[WP {wp_idx+1}/{len(waypoints)}] -> ({wp_goal[0]:.1f}, {wp_goal[1]:.1f})")
        stuck_count = 0
        prev_pos = x[:2].copy()

        for step in range(500):
            u, predicted_trajectory = dwa_control(x, config, wp_goal, ob)
            x = motion(x, u, config.dt, config.wheelbase)
            traj_list.append(x.copy())
            total_step += 1

            if show_animation:
                plt.cla()
                for ci in range(ob.shape[0]):
                    circle = plt.Circle(
                        (ob[ci, 0], ob[ci, 1]), ob[ci, 2],
                        color="gray", alpha=0.4)
                    plt.gcf().gca().add_artist(circle)
                wp_arr = np.array(waypoints)
                plt.plot(wp_arr[:, 0], wp_arr[:, 1], "s--c", ms=6, label="waypoints")
                plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
                plt.plot(x[0], x[1], "xr")
                plt.plot(wp_goal[0], wp_goal[1], "xb", ms=10)
                plot_robot(x[0], x[1], x[2], config)
                plot_arrow(x[0], x[1], x[2])
                plt.axis("equal")
                plt.xlim(-5, 55)
                plt.ylim(-5, 55)
                plt.grid(True)
                plt.legend(loc="upper left", frameon=True, fancybox=True)
                plt.title(f"WP {wp_idx+1}/{len(waypoints)} step={total_step}")
                plt.pause(0.0001)

            dist_to_wp = math.hypot(x[0] - wp_goal[0], x[1] - wp_goal[1])
            if dist_to_wp <= config.robot_radius + 1.0:
                print(f"  WP {wp_idx+1} reached at step {total_step}")
                break

            if np.linalg.norm(x[:2] - prev_pos) < 0.2:
                stuck_count += 1
                if stuck_count > 50:
                    print(f"  Stuck at step {total_step}, pos=({x[0]:.1f},{x[1]:.1f})")
                    break
            else:
                stuck_count = 0
                prev_pos = x[:2].copy()

    dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
    if dist_to_goal <= config.robot_radius + 1.0:
        print("Goal!!")
    else:
        print(f"Final pos=({x[0]:.1f},{x[1]:.1f}), dist_to_goal={dist_to_goal:.2f}")

    trajectory = np.array(traj_list)
    print(f"Done in {total_step} steps")
    if show_animation:
        plt.cla()
        for ci in range(ob.shape[0]):
            circle = plt.Circle(
                (ob[ci, 0], ob[ci, 1]), ob[ci, 2],
                color="gray", alpha=0.4)
            plt.gcf().gca().add_artist(circle)
        wp_arr = np.array(waypoints)
        plt.plot(wp_arr[:, 0], wp_arr[:, 1], "s--c", ms=6, label="waypoints")
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r", label="trajectory")
        plt.plot(goal[0], goal[1], "xb", ms=10, label="goal")
        plt.plot(trajectory[0, 0], trajectory[0, 1], "og", ms=8, label="start")
        plt.axis("equal")
        plt.xlim(-5, 55)
        plt.ylim(-5, 55)
        plt.grid(True)
        plt.legend(loc="upper left", frameon=True, fancybox=True)
        plt.title("DWA Final Trajectory")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
