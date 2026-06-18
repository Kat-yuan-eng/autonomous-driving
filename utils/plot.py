"""
Plotting utilities for vehicle, path, trajectory and obstacle visualization

author: Kat-yuan-eng (RuiWen Liao)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from utils.angle import rot_mat_2d


def plot_arrow(x, y, yaw, arrow_length=1.0,
               origin_point_plot_style="xr",
               head_width=0.1, fc="r", ec="k", **kwargs):
    """
    Plot an arrow or arrows based on 2D state (x, y, yaw).

    :param x: (float or array_like) Arrow origin x position.
    :param y: (float or array_like) Arrow origin y position.
    :param yaw: (float or array_like) Arrow yaw angle (orientation).
    :param arrow_length: (float) Arrow length. Default is 1.0.
    :param origin_point_plot_style: (str or None) Origin point plot style. If None, not plotting. Default is "xr".
    :param head_width: (float) Arrow head width. Default is 0.1.
    :param fc: (str) Face color. Default is "r".
    :param ec: (str) Edge color. Default is "k".
    """
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, arrow_length=arrow_length,
                       origin_point_plot_style=origin_point_plot_style,
                       head_width=head_width,
                       fc=fc, ec=ec, **kwargs)
    else:
        plt.arrow(x, y,
                  arrow_length * math.cos(yaw),
                  arrow_length * math.sin(yaw),
                  head_width=head_width,
                  fc=fc, ec=ec,
                  **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


def plot_robot(x, y, yaw, length=0.5, width=0.3, color="blue"):
    """
    Draw a rectangle representing the robot/vehicle at position (x, y) with heading yaw.

    :param x: (float) X position of the robot center.
    :param y: (float) Y position of the robot center.
    :param yaw: (float) Heading angle of the robot in radians.
    :param length: (float) Length of the robot. Default is 0.5.
    :param width: (float) Width of the robot. Default is 0.3.
    :param color: (str) Face color of the robot rectangle. Default is "blue".
    """
    corners = np.array([
        [-length / 2.0, -width / 2.0],
        [length / 2.0, -width / 2.0],
        [length / 2.0, width / 2.0],
        [-length / 2.0, width / 2.0],
    ])
    rot = rot_mat_2d(yaw)
    corners = (rot @ corners.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    robot = plt.Polygon(corners, color=color, alpha=0.7)
    plt.gca().add_patch(robot)
    plot_arrow(x, y, yaw, arrow_length=length * 0.6,
               head_width=width * 0.2, fc="white", ec="white",
               origin_point_plot_style=None)


def plot_grid_map(ax, grid, start, goal):
    """
    Draw grid map on given axes. Obstacles as black squares, free space white, start as green circle, goal as blue circle.

    :param ax: (matplotlib.axes.Axes) Axes to draw the grid map on.
    :param grid: (numpy.ndarray) 2D occupancy grid array. 1 for obstacle, 0 for free space.
    :param start: (tuple) (x, y) start position in grid coordinates.
    :param goal: (tuple) (x, y) goal position in grid coordinates.
    """
    ax.imshow(grid, cmap="binary", origin="lower")
    ax.plot(start[0], start[1], "og", markersize=10, label="start")
    ax.plot(goal[0], goal[1], "ob", markersize=10, label="goal")
    ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Grid Map")


def plot_path(ax, path_x, path_y, color="-r"):
    """
    Plot path line on given axes.

    :param ax: (matplotlib.axes.Axes) Axes to draw the path on.
    :param path_x: (array_like) X position list of the path.
    :param path_y: (array_like) Y position list of the path.
    :param color: (str) Color and line style of the path. Default is "-r".
    """
    ax.plot(path_x, path_y, color, label="path")
    ax.legend(loc="upper right")


def plot_expanded(ax, expanded_x, expanded_y):
    """
    Plot expanded nodes as cyan dots.

    :param ax: (matplotlib.axes.Axes) Axes to draw the expanded nodes on.
    :param expanded_x: (array_like) X position list of expanded nodes.
    :param expanded_y: (array_like) Y position list of expanded nodes.
    """
    ax.plot(expanded_x, expanded_y, "xc", markersize=3, label="expanded")
    ax.legend(loc="upper right")


def plot_trajectory(ax, traj_x, traj_y, color="-g"):
    """
    Plot trajectory line.

    :param ax: (matplotlib.axes.Axes) Axes to draw the trajectory on.
    :param traj_x: (array_like) X position list of the trajectory.
    :param traj_y: (array_like) Y position list of the trajectory.
    :param color: (str) Color and line style of the trajectory. Default is "-g".
    """
    ax.plot(traj_x, traj_y, color, label="trajectory")
    ax.legend(loc="upper right")


def plot_obstacles(ax, ox, oy, size=0.3):
    """
    Plot circular obstacles.

    :param ax: (matplotlib.axes.Axes) Axes to draw the obstacles on.
    :param ox: (array_like) X position list of obstacles.
    :param oy: (array_like) Y position list of obstacles.
    :param size: (float) Radius of the obstacle circles. Default is 0.3.
    """
    for (i_x, i_y) in zip(ox, oy):
        circle = plt.Circle((i_x, i_y), size, color="k")
        ax.add_patch(circle)
    ax.set_aspect("equal")


def plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.01, c="-c", label="Curvature"):
    """
    Plot curvature on 2D path. Lateral distance from the original path shows curvature magnitude.

    :param x_list: (array_like) X position list of the path.
    :param y_list: (array_like) Y position list of the path.
    :param heading_list: (array_like) Heading list of the path.
    :param curvature: (array_like) Curvature list of the path.
    :param k: (float) Curvature scale factor to calculate distance from the original path. Default is 0.01.
    :param c: (str) Color of the plot. Default is "-c".
    :param label: (str) Label of the plot. Default is "Curvature".
    """
    cx = [x + d * k * np.cos(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]
    cy = [y + d * k * np.sin(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]

    plt.plot(cx, cy, c, label=label)
    for ix, iy, icx, icy in zip(x_list, y_list, cx, cy):
        plt.plot([ix, icx], [iy, icy], c)


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # === plot_arrow demo ===
    ax = axes[0, 0]
    plt.sca(ax)
    plot_arrow(0, 0, math.radians(45), arrow_length=2.0)
    plot_arrow(1, 0, math.radians(90), arrow_length=1.5)
    plot_arrow([3, 3, 3], [0, 1, 2],
               [math.radians(0), math.radians(30), math.radians(60)])
    ax.set_title("plot_arrow")
    ax.axis("equal")

    # === plot_robot demo ===
    ax = axes[0, 1]
    plt.sca(ax)
    plot_robot(0, 0, 0, color="blue")
    plot_robot(2, 1, math.radians(45), color="red")
    plot_robot(4, 0, math.radians(90), color="green")
    ax.set_title("plot_robot")
    ax.axis("equal")

    # === plot_grid_map demo ===
    ax = axes[0, 2]
    grid = np.zeros((10, 10))
    grid[3:7, 3:7] = 1
    plot_grid_map(ax, grid, (1, 1), (8, 8))

    # === plot_path + plot_expanded demo ===
    ax = axes[1, 0]
    path_x = np.linspace(0, 10, 50)
    path_y = np.sin(path_x)
    expanded_x = np.random.rand(30) * 10
    expanded_y = np.sin(expanded_x) + np.random.randn(30) * 0.3
    plot_expanded(ax, expanded_x, expanded_y)
    plot_path(ax, path_x, path_y)
    ax.set_title("plot_path + plot_expanded")

    # === plot_trajectory + plot_obstacles demo ===
    ax = axes[1, 1]
    ox, oy = [2, 5, 8], [1, 3, 2]
    plot_obstacles(ax, ox, oy, size=0.5)
    traj_x = np.linspace(0, 10, 100)
    traj_y = 0.5 * np.sin(traj_x) + 2
    plot_trajectory(ax, traj_x, traj_y)
    ax.set_title("plot_trajectory + plot_obstacles")

    # === plot_curvature demo ===
    ax = axes[1, 2]
    plt.sca(ax)
    x_list = np.linspace(0, 2 * np.pi, 100)
    y_list = np.sin(x_list)
    dx = np.gradient(x_list)
    dy = np.gradient(y_list)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    heading_list = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12)
    plt.plot(x_list, y_list, "-k", label="path")
    plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.5, c="-c", label="Curvature")
    ax.set_title("plot_curvature")
    ax.axis("equal")
    ax.legend()

    plt.tight_layout()
    plt.show()
