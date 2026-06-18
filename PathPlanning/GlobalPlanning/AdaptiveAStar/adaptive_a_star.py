"""
Adaptive A* with Sigmoid-based heuristic switching

author: Kat-yuan-eng (RuiWen Liao)
"""

import heapq
import math
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

from utils.grid_map import generate_random_map

show_animation = True

MOTION_8CONN = [
    (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

THETA_B_DEG = 22.5  # [deg]
THETA_MID_DEG = 45.0  # [deg]
THETA_B2_DEG = 67.5  # [deg]


# === Phase 1: Adaptive Heuristic ===

def adaptive_heuristic(nx, ny, gx, gy, k_sigmoid=0.5):
    """
    Compute adaptive heuristic with Sigmoid soft-switching between Chebyshev and Euclidean.

    :param nx: (int) Current node x
    :param ny: (int) Current node y
    :param gx: (int) Goal x
    :param gy: (int) Goal y
    :param k_sigmoid: (float) Sigmoid sharpness parameter
    :return: (float) Heuristic value
    """
    dx = abs(gx - nx)
    dy = abs(gy - ny)

    h_e = math.hypot(dx, dy)
    h_c = max(dx, dy)
    h_m = dx + dy

    if h_e < 1e-12:
        return 0.0

    theta_rad = math.atan2(gy - ny, gx - nx)
    theta_deg = math.degrees(theta_rad)
    theta_mod = abs(theta_deg) % 90.0

    w = 1.0 / (1.0 + math.exp(-k_sigmoid * (theta_mod - THETA_B_DEG)))

    if theta_mod < THETA_MID_DEG:
        return (1.0 - w) * h_c + w * h_e

    w2 = 1.0 / (1.0 + math.exp(-k_sigmoid * (theta_mod - THETA_B2_DEG)))
    return (1.0 - w2) * h_e + w2 * h_c


def euclidean_heuristic(nx, ny, gx, gy):
    """
    Compute Euclidean distance heuristic.

    :param nx: (int) Current node x
    :param ny: (int) Current node y
    :param gx: (int) Goal x
    :param gy: (int) Goal y
    :return: (float) Euclidean distance
    """
    return math.hypot(gx - nx, gy - ny)


# === Phase 2: A* Planner ===

class AdaptiveAStarPlanner:
    """
    Adaptive A* grid path planner with Sigmoid-based heuristic switching.
    """

    class Node:
        """
        Search node for Adaptive A* algorithm.
        """

        def __init__(self, x, y, cost, parent_index):
            """
            :param x: (int) Column index (horizontal position).
            :param y: (int) Row index (vertical position).
            :param cost: (float) Accumulated cost from start to this node.
            :param parent_index: (int) Index of parent node in closed_set (-1 for start).
            """
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def __init__(self, grid):
        """
        Initialize Adaptive A* planner with a grid map.

        :param grid: (numpy.ndarray) 2-D array, 0 = free cell, 1 = obstacle
        """
        self.grid = grid
        self.n_row, self.n_col = grid.shape
        self.motion = MOTION_8CONN

    def planning(self, sx, sy, gx, gy, heuristic_type="adaptive",
                 k_sigmoid=0.5):
        """
        A* path search with selectable heuristic.

        :param sx: (int) Start x (column index)
        :param sy: (int) Start y (row index)
        :param gx: (int) Goal x (column index)
        :param gy: (int) Goal y (row index)
        :param heuristic_type: (str) "adaptive" or "euclidean"
        :param k_sigmoid: (float) Sigmoid sharpness for adaptive heuristic
        :return: (tuple) (rx, ry, expanded_nodes, planning_time_ms)
        """
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

        open_set = {}
        closed_set = {}
        heap = []
        counter = 0

        start_id = self._grid_index(start_node)
        open_set[start_id] = start_node
        f_start = start_node.cost + self._calc_h(
            start_node, goal_node, heuristic_type, k_sigmoid)
        heapq.heappush(heap, (f_start, counter, start_id))
        counter += 1

        expanded_count = 0
        t_start = time.perf_counter()

        while heap:
            _, _, c_id = heapq.heappop(heap)

            if c_id not in open_set:
                continue

            current = open_set[c_id]

            if show_animation and heuristic_type == "adaptive":
                plt.plot(current.x, current.y, "xc")
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(
                        0) if event.key == 'escape' else None])
                if len(closed_set) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current
            expanded_count += 1

            for dx, dy, move_cost in self.motion:
                node = self.Node(
                    current.x + dx, current.y + dy,
                    current.cost + move_cost, c_id)
                n_id = self._grid_index(node)

                if not self._verify_node(node):
                    continue
                if not self._can_move_diagonal(current.x, current.y, dx, dy):
                    continue
                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                    f_new = node.cost + self._calc_h(
                        node, goal_node, heuristic_type, k_sigmoid)
                    heapq.heappush(heap, (f_new, counter, n_id))
                    counter += 1
                elif open_set[n_id].cost > node.cost:
                    open_set[n_id] = node
                    f_new = node.cost + self._calc_h(
                        node, goal_node, heuristic_type, k_sigmoid)
                    heapq.heappush(heap, (f_new, counter, n_id))
                    counter += 1
        else:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return [], [], expanded_count, t_elapsed_ms

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        rx, ry = self._calc_final_path(goal_node, closed_set)
        return rx, ry, expanded_count, t_elapsed_ms

    def _calc_h(self, node, goal, heuristic_type, k_sigmoid):
        """
        Compute heuristic value for a node.

        :param node: (Node) Current node
        :param goal: (Node) Goal node
        :param heuristic_type: (str) "adaptive" or "euclidean"
        :param k_sigmoid: (float) Sigmoid sharpness parameter
        :return: (float) Heuristic value
        """
        if heuristic_type == "adaptive":
            return adaptive_heuristic(
                node.x, node.y, goal.x, goal.y, k_sigmoid)
        return euclidean_heuristic(node.x, node.y, goal.x, goal.y)

    def _grid_index(self, node):
        """
        Compute a unique integer index for a grid node.

        :param node: (Node) Node with x and y attributes
        :return: (int) Unique index = y * n_col + x
        """
        return node.y * self.n_col + node.x

    def _verify_node(self, node):
        """
        Check if a node is within map bounds and not on an obstacle.

        :param node: (Node) Node to verify
        :return: (bool) True if node is valid
        """
        if node.x < 0 or node.x >= self.n_col:
            return False
        if node.y < 0 or node.y >= self.n_row:
            return False
        if self.grid[node.y, node.x] == 1:
            return False
        return True

    def _can_move_diagonal(self, cx, cy, dx, dy):
        """
        Check if diagonal movement is possible without corner cutting.

        :param cx: (int) Current x position
        :param cy: (int) Current y position
        :param dx: (int) X direction of movement
        :param dy: (int) Y direction of movement
        :return: (bool) True if diagonal move is valid
        """
        if dx == 0 or dy == 0:
            return True
        nx_h, ny_h = cx + dx, cy
        nx_v, ny_v = cx, cy + dy
        if not (0 <= nx_h < self.n_col and 0 <= ny_h < self.n_row):
            return False
        if not (0 <= nx_v < self.n_col and 0 <= ny_v < self.n_row):
            return False
        return self.grid[ny_h, nx_h] == 0 and self.grid[ny_v, nx_v] == 0

    def _calc_final_path(self, goal_node, closed_set):
        """
        Trace back from goal to start via parent indices.

        :param goal_node: (Node) Goal node with parent_index set
        :param closed_set: (dict) Mapping from node index to Node for all visited nodes
        :return: (tuple) (rx, ry) x and y coordinate lists of the path
        """
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]


# === Phase 3: Comparison & Visualization ===

def calc_path_length(rx, ry):
    """
    Calculate total Euclidean path length.

    :param rx: (list) X coordinates of the path
    :param ry: (list) Y coordinates of the path
    :return: (float) Total path length
    """
    return sum(
        math.hypot(rx[i + 1] - rx[i], ry[i + 1] - ry[i])
        for i in range(len(rx) - 1))


def main():
    """
    Run Adaptive A* planner demo comparing euclidean vs adaptive heuristic.
    """
    print(__file__ + " start!!")

    N_ROW, N_COL = 50, 50
    OBS_RATIO = 0.2

    grid = generate_random_map(N_ROW, N_COL, OBS_RATIO, seed=42)

    sx, sy = 0, 0
    gx, gy = N_COL - 1, N_ROW - 1

    assert grid[sy, sx] == 0, "Start cell must be free"
    assert grid[gy, gx] == 0, "Goal cell must be free"

    planner = AdaptiveAStarPlanner(grid)

    rx_std, ry_std, n_std, t_std = planner.planning(
        sx, sy, gx, gy, heuristic_type="euclidean")

    if show_animation:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(grid, cmap="binary", origin="lower")
        ax.plot(sx, sy, "og", markersize=10, label="start")
        ax.plot(gx, gy, "xb", markersize=10, label="goal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Adaptive A* Search")
        ax.legend(loc="upper right")

    rx_adp, ry_adp, n_adp, t_adp = planner.planning(
        sx, sy, gx, gy, heuristic_type="adaptive")

    L_std = calc_path_length(rx_std, ry_std)
    L_adp = calc_path_length(rx_adp, ry_adp)

    print("=== Standard A* (Euclidean) vs Adaptive A* ===")
    print(f"  Path length:     Std={L_std:.4f}  Adp={L_adp:.4f}")
    print(f"  Expanded nodes:  Std={n_std}  Adp={n_adp}")
    print(f"  Planning time:   Std={t_std:.2f}ms  Adp={t_adp:.2f}ms")

    delta_L = abs(L_std - L_adp)
    if delta_L < 1e-6:
        print(f"  Path length diff: {delta_L:.2e} (< 1e-6, PASS)")
    else:
        print(f"  Path length diff: {delta_L:.2e} (>= 1e-6, FAIL)")

    if show_animation:
        ax.plot(rx_adp, ry_adp, "-r", label="adaptive path")
        ax.legend(loc="upper right")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
