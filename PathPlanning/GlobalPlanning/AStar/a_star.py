"""
A* grid path planning with 8-connectivity and multiple heuristics

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
"""

import heapq
import math
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

show_animation = True


class AStarPlanner:
    """
    A* grid path planner with 8-connectivity and selectable heuristic.
    """

    def __init__(self, grid_map, heuristic="euclidean"):
        """
        Initialize A* planner on a binary grid map.

        :param grid_map: (numpy.ndarray) 2D array, True/1 = obstacle, False/0 = free
        :param heuristic: (str) 'euclidean', 'manhattan', or 'chebyshev'
        """
        assert heuristic in ("euclidean", "manhattan", "chebyshev"), \
            f"heuristic must be 'euclidean', 'manhattan', or 'chebyshev', got '{heuristic}'"
        self.grid_map = np.array(grid_map, dtype=bool)
        self.y_size, self.x_size = self.grid_map.shape
        self.heuristic = heuristic
        self.motion = self._get_motion_model()

    class Node:
        """
        Search node for A* algorithm.
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
        A* path search on 8-connected grid.

        :param sx: (int) Start x grid index
        :param sy: (int) Start y grid index
        :param gx: (int) Goal x grid index
        :param gy: (int) Goal y grid index
        :return: (tuple) (rx, ry, expanded) where rx/ry are path coordinate lists and expanded is node count
        """
        assert 0 <= sx < self.x_size and 0 <= sy < self.y_size, \
            f"start ({sx},{sy}) out of map bounds ({self.x_size},{self.y_size})"
        assert 0 <= gx < self.x_size and 0 <= gy < self.y_size, \
            f"goal ({gx},{gy}) out of map bounds ({self.x_size},{self.y_size})"
        assert not self.grid_map[sy, sx], f"start ({sx},{sy}) is on obstacle"
        assert not self.grid_map[gy, gx], f"goal ({gx},{gy}) is on obstacle"

        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

        open_list = []
        heapq.heappush(open_list, (0.0, 0, start_node))
        closed_set = {}
        open_set = {}
        open_set[self._calc_index(start_node)] = start_node
        expanded = 0
        counter = 1

        while open_list:
            _, _, current = heapq.heappop(open_list)
            c_id = self._calc_index(current)

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
                n_id = self._calc_index(node)

                if not self._verify_node(node):
                    continue
                if not self._can_move_diagonal(current.x, current.y, dx, dy):
                    continue
                if n_id in closed_set:
                    continue

                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node
                    h = self._calc_heuristic(node, goal_node)
                    heapq.heappush(open_list, (node.cost + h, counter, node))
                    counter += 1
        else:
            print("Open set is empty.. no path found")
            return [], [], expanded

        rx, ry = self._calc_final_path(goal_node, closed_set)
        return rx, ry, expanded

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
        rx.reverse()
        ry.reverse()
        return rx, ry

    def _calc_heuristic(self, n1, n2):
        """
        Compute heuristic cost between two nodes.

        :param n1: (Node) First node
        :param n2: (Node) Second node
        :return: (float) Heuristic distance
        """
        dx = abs(n1.x - n2.x)
        dy = abs(n1.y - n2.y)
        if self.heuristic == "euclidean":
            return math.hypot(dx, dy)
        elif self.heuristic == "manhattan":
            return dx + dy
        else:
            return max(dx, dy)

    def _calc_index(self, node):
        """
        Compute a unique integer index for a grid node.

        :param node: (Node) Node with x and y attributes
        :return: (int) Unique index = y * x_size + x
        """
        return node.y * self.x_size + node.x

    def _verify_node(self, node):
        """
        Check if a node is within map bounds and not on an obstacle.

        :param node: (Node) Node to verify
        :return: (bool) True if node is valid
        """
        if node.x < 0 or node.x >= self.x_size:
            return False
        if node.y < 0 or node.y >= self.y_size:
            return False
        if self.grid_map[node.y, node.x]:
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
        if not (0 <= nx_h < self.x_size and 0 <= ny_h < self.y_size):
            return False
        if not (0 <= nx_v < self.x_size and 0 <= ny_v < self.y_size):
            return False
        return self.grid_map[ny_h, nx_h] == 0 and self.grid_map[ny_v, nx_v] == 0

    @staticmethod
    def _get_motion_model():
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


def calc_path_length(rx, ry):
    """
    Calculate total Euclidean path length.

    :param rx: (list) X coordinates of the path
    :param ry: (list) Y coordinates of the path
    :return: (float) Total path length
    """
    total = 0.0
    for i in range(1, len(rx)):
        total += math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1])
    return total


def main():
    """
    Run A* planner demo with multiple heuristics comparison.
    """
    print(__file__ + " start!!")

    np.random.seed(42)

    x_size, y_size = 50, 50
    obstacle_ratio = 0.20
    grid_map = np.random.rand(y_size, x_size) < obstacle_ratio

    sx, sy = 2, 2
    gx, gy = x_size - 3, y_size - 3
    grid_map[sy, sx] = False
    grid_map[gy, gx] = False

    heuristics = ["euclidean", "manhattan", "chebyshev"]
    results = {}

    for h_name in heuristics:
        planner = AStarPlanner(grid_map, heuristic=h_name)
        t0 = time.perf_counter()
        rx, ry, expanded = planner.planning(sx, sy, gx, gy)
        dt = time.perf_counter() - t0
        path_len = calc_path_length(rx, ry) if rx else float("inf")
        results[h_name] = {"path_length": path_len, "expanded": expanded, "time": dt}
        print(f"[{h_name}] path_length={path_len:.2f}, expanded={expanded}, time={dt:.4f}s")

    print("\n=== Comparison ===")
    print(f"{'Heuristic':<12} {'Path Length':>12} {'Expanded':>10} {'Time (s)':>10}")
    for h_name in heuristics:
        r = results[h_name]
        print(f"{h_name:<12} {r['path_length']:>12.2f} {r['expanded']:>10} {r['time']:>10.4f}")

    if show_animation:
        planner_anim = AStarPlanner(grid_map, heuristic="euclidean")
        plt.figure(figsize=(8, 8))
        obs_y, obs_x = np.where(grid_map)
        plt.plot(obs_x, obs_y, ".k", markersize=2)
        plt.plot(sx, sy, "og", markersize=8)
        plt.plot(gx, gy, "xb", markersize=8)
        plt.grid(True)
        plt.axis("equal")
        plt.title("A* with euclidean heuristic")
        rx, ry, _ = planner_anim.planning(sx, sy, gx, gy)
        if rx:
            plt.plot(rx, ry, "-r", linewidth=2)
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
