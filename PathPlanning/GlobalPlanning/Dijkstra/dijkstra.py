"""
Dijkstra grid path planning with 8-connectivity

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
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


class DijkstraPlanner:
    """
    Dijkstra grid path planner with 8-connectivity.
    """

    def __init__(self, grid):
        """
        Initialize Dijkstra planner with a grid map.

        :param grid: (numpy.ndarray) 2-D array of shape (n_row, n_col). 0 = free cell, 1 = obstacle
        """
        self.grid = grid
        self.n_row, self.n_col = grid.shape
        self.motion = self.get_motion_model()

    class Node:
        """
        Search node for Dijkstra algorithm.
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

        def __lt__(self, other):
            return self.cost < other.cost

    def planning(self, sx, sy, gx, gy):
        """
        Dijkstra path search on an 8-connected grid.

        :param sx: (int) Start x position (column index)
        :param sy: (int) Start y position (row index)
        :param gx: (int) Goal x position (column index)
        :param gy: (int) Goal y position (row index)
        :return: (tuple) (rx, ry, expanded) where rx/ry are path coordinate lists and expanded is node count
        """
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = dict()
        node_map = dict()
        start_id = self.calc_node_index(start_node)
        node_map[start_id] = start_node

        expanded = 0

        while open_set:
            current = heapq.heappop(open_set)
            c_id = self.calc_node_index(current)

            if c_id in closed_set:
                continue

            closed_set[c_id] = current
            expanded += 1

            if show_animation:
                plt.plot(current.x, current.y, "xc")
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if expanded % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            for dx, dy, move_cost in self.motion:
                nx, ny = current.x + dx, current.y + dy
                node = self.Node(nx, ny, current.cost + move_cost, c_id)
                n_id = self.calc_node_index(node)

                if not self.verify_node(node):
                    continue
                if not self._can_move_diagonal(current.x, current.y, dx, dy):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in node_map or node_map[n_id].cost > node.cost:
                    node_map[n_id] = node
                    heapq.heappush(open_set, node)

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry, expanded

    def calc_final_path(self, goal_node, closed_set):
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
        rx.reverse()
        ry.reverse()
        return rx, ry

    def calc_node_index(self, node):
        """
        Compute a unique integer index for a grid node.

        :param node: (Node) Node with x (column) and y (row) attributes
        :return: (int) Unique index = y * n_col + x
        """
        return node.y * self.n_col + node.x

    def verify_node(self, node):
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

    @staticmethod
    def get_motion_model():
        """
        Return 8-connected motion model: (dx, dy, cost).

        :return: (list) List of (dx, dy, cost) tuples
        """
        motion = [
            (1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2)),
        ]
        return motion


def main():
    """
    Run Dijkstra planner demo on a random grid map.
    """
    print(__file__ + " start!!")

    n_row, n_col = 50, 50
    obs_ratio = 0.2

    grid = generate_random_map(n_row, n_col, obs_ratio, seed=42)

    sx, sy = 0, 0
    gx, gy = n_col - 1, n_row - 1

    if show_animation:
        obs_y, obs_x = np.where(grid == 1)
        plt.plot(obs_x, obs_y, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    planner = DijkstraPlanner(grid)

    t0 = time.time()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed = time.time() - t0

    path_length = sum(
        math.hypot(rx[i + 1] - rx[i], ry[i + 1] - ry[i])
        for i in range(len(rx) - 1)
    )

    print(f"Path length : {path_length:.2f}")
    print(f"Expanded    : {expanded}")
    print(f"Time        : {elapsed:.4f} s")

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
