"""
RRT and RRT* path planning with path smoothing

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [RRT Algorithm](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)
"""

import math
import random
import time
import numpy as np


class RRT:
    """
    RRT / RRT* path planner with collision checking and path smoothing.
    """

    class Node:
        """
        Search node for RRT algorithm.
        """

        def __init__(self, x, y):
            """
            :param x: (float) X position
            :param y: (float) Y position
            """
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0.0

    class AreaBounds:
        """
        Bounding area for random sampling.
        """

        def __init__(self, area):
            """
            :param area: (list) [xmin, xmax, ymin, ymax]
            """
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=0.5,
                 path_resolution=0.5,
                 goal_sample_rate=10,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.3,
                 safety_margin=0.2,
                 use_rrt_star=False,
                 connect_circle_dist=50.0,
                 ):
        """
        Initialize RRT / RRT* planner.

        :param start: (tuple) Start position (x, y)
        :param goal: (tuple) Goal position (x, y)
        :param obstacle_list: (list) List of (ox, oy, radius) tuples
        :param rand_area: (tuple) (min_rand, max_rand) for random sampling
        :param expand_dis: (float) Step size for tree expansion
        :param path_resolution: (float) Resolution for path interpolation
        :param goal_sample_rate: (int) Percentage of sampling goal directly
        :param max_iter: (int) Maximum number of iterations
        :param play_area: (list) [xmin, xmax, ymin, ymax] or None
        :param robot_radius: (float) Robot radius for collision checking
        :param safety_margin: (float) Safety margin around obstacles
        :param use_rrt_star: (bool) Enable RRT* optimization
        :param connect_circle_dist: (float) Connection radius for RRT* rewiring
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.use_rrt_star = use_rrt_star
        self.connect_circle_dist = connect_circle_dist

    def planning(self):
        """
        Execute RRT / RRT* path planning.

        :return: (tuple) (path, n_nodes, elapsed_time) or (None, n_nodes, elapsed_time) if no path found
        """
        self.node_list = [self.start]
        t_start = time.time()

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(new_node, self.obstacle_list):

                if self.use_rrt_star:
                    new_node.cost = nearest_node.cost + math.hypot(
                        new_node.x - nearest_node.x, new_node.y - nearest_node.y)
                    near_inds = self.find_near_nodes(new_node)
                    node_with_updated_parent = self.choose_parent(
                        new_node, near_inds, self.obstacle_list)
                    if node_with_updated_parent:
                        new_node = node_with_updated_parent
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds, self.obstacle_list)
                else:
                    self.node_list.append(new_node)

            if self.use_rrt_star:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    t_end = time.time()
                    path = self.generate_final_course(last_index)
                    return path, len(self.node_list), t_end - t_start
            else:
                if self.calc_dist_to_goal(self.node_list[-1].x,
                                          self.node_list[-1].y) <= self.expand_dis:
                    final_node = self.steer(self.node_list[-1], self.end,
                                            self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        t_end = time.time()
                        path = self.generate_final_course(len(self.node_list) - 1)
                        return path, len(self.node_list), t_end - t_start

        if self.use_rrt_star:
            last_index = self.search_best_goal_node()
            if last_index is not None:
                t_end = time.time()
                path = self.generate_final_course(last_index)
                return path, len(self.node_list), t_end - t_start

        t_end = time.time()
        return None, len(self.node_list), t_end - t_start

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Extend from_node toward to_node by extend_length.

        :param from_node: (Node) Source node
        :param to_node: (Node) Target node
        :param extend_length: (float) Maximum extension distance
        :return: (Node) New node after steering
        """
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.ceil(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        """
        Generate path from start to goal by tracing parent links.

        :param goal_ind: (int) Index of goal node in node_list
        :return: (list) Path as list of [x, y] coordinates
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        """
        Calculate Euclidean distance from (x, y) to goal.

        :param x: (float) X position
        :param y: (float) Y position
        :return: (float) Distance to goal
        """
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        """
        Generate a random node with goal bias.

        :return: (Node) Randomly sampled node
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """
        Find index of the nearest node in node_list to rnd_node.

        :param node_list: (list) List of Node objects
        :param rnd_node: (Node) Target node
        :return: (int) Index of nearest node
        """
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        """
        Check if node is within the play area bounds.

        :param node: (Node) Node to check
        :param play_area: (AreaBounds) Play area bounds or None
        :return: (bool) True if inside play area
        """
        if play_area is None:
            return True
        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False
        else:
            return True

    def check_collision(self, node, obstacle_list):
        """
        Check if node path collides with any obstacle.

        :param node: (Node) Node with path_x, path_y to check
        :param obstacle_list: (list) List of (ox, oy, radius) tuples
        :return: (bool) True if no collision
        """
        if node is None:
            return False
        px = np.array(node.path_x)
        py = np.array(node.path_y)
        for (ox, oy, size) in obstacle_list:
            dx = px - ox
            dy = py - oy
            d_sq = dx * dx + dy * dy
            if np.any(d_sq <= (size + self.robot_radius + self.safety_margin) ** 2):
                return False
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        Calculate Euclidean distance and angle between two nodes.

        :param from_node: (Node) Source node
        :param to_node: (Node) Target node
        :return: (tuple) (distance, angle_in_radians)
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def choose_parent(self, new_node, near_inds, obstacle_list):
        """
        Choose the best parent for new_node from nearby nodes (RRT*).

        :param new_node: (Node) New node to find parent for
        :param near_inds: (list) Indices of nearby nodes
        :param obstacle_list: (list) List of (ox, oy, radius) tuples
        :return: (Node or None) New node with updated parent, or None
        """
        if not near_inds:
            return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds, obstacle_list):
        """
        Rewire nearby nodes through new_node if cost improves (RRT*).

        :param new_node: (Node) Newly added node
        :param near_inds: (list) Indices of nearby nodes
        :param obstacle_list: (list) List of (ox, oy, radius) tuples
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)
            no_collision = self.check_collision(edge_node, obstacle_list)
            improved_cost = near_node.cost > edge_node.cost
            if no_collision and improved_cost:
                near_node.parent = new_node
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                self.propagate_cost_to_leaves(near_node)

    def find_near_nodes(self, node):
        """
        Find indices of nodes within connect_circle_dist of given node.

        :param node: (Node) Target node
        :return: (list) Indices of nearby nodes
        """
        n = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(n) / n)
        r = min(r, self.connect_circle_dist)
        dist_list = [(nd.x - node.x)**2 + (nd.y - node.y)**2
                     for nd in self.node_list]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r**2]
        return near_inds

    def propagate_cost_to_leaves(self, parent_node):
        """
        Recursively update cost of all descendant nodes.

        :param parent_node: (Node) Parent node whose cost changed
        """
        stack = [parent_node]
        while stack:
            current = stack.pop()
            for node in self.node_list:
                if node.parent == current:
                    node.cost = self.calc_new_cost(current, node)
                    stack.append(node)

    def calc_new_cost(self, from_node, to_node):
        """
        Calculate cost of reaching to_node via from_node.

        :param from_node: (Node) Potential parent node
        :param to_node: (Node) Target node
        :return: (float) New cost
        """
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def search_best_goal_node(self):
        """
        Find the best goal node reachable from the tree (RRT*).

        :return: (int or None) Index of best goal node, or None
        """
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y)
                             for n in self.node_list]
        goal_inds = [i for i, d in enumerate(dist_to_goal_list)
                     if d <= self.expand_dis]
        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.end)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)
        if not safe_goal_inds:
            return None
        min_cost = min(self.node_list[i].cost +
                       self.calc_dist_to_goal(self.node_list[i].x,
                                              self.node_list[i].y)
                       for i in safe_goal_inds)
        for i in safe_goal_inds:
            if self.node_list[i].cost + self.calc_dist_to_goal(
                    self.node_list[i].x, self.node_list[i].y) == min_cost:
                return i
        return None

    def smooth_path(self, path, max_iter=300):
        """
        Smooth path by randomly shortcutting segments.

        :param path: (list) Path as list of [x, y] coordinates
        :param max_iter: (int) Maximum smoothing iterations
        :return: (list) Smoothed path
        """
        if path is None or len(path) <= 2:
            return path
        le = 0.0
        for i in range(len(path) - 1):
            le += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for _ in range(max_iter):
            pick = sorted([random.uniform(0, le), random.uniform(0, le)])
            first = self._path_target_point(path, pick[0])
            second = self._path_target_point(path, pick[1])
            if first[2] <= 0 or second[2] <= 0 or second[2] == first[2]:
                continue
            if second[2] + 1 > len(path):
                continue
            if not self._line_collision_check(first, second):
                continue
            path = path[:first[2] + 1] + [[first[0], first[1]], [second[0], second[1]]] + path[second[2] + 1:]
            le = 0.0
            for i in range(len(path) - 1):
                le += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        return path

    def _path_target_point(self, path, target_l):
        """
        Find point on path at given arc length.

        :param path: (list) Path as list of [x, y] coordinates
        :param target_l: (float) Target arc length
        :return: (list) [x, y, segment_index]
        """
        le = 0.0
        for i in range(len(path) - 1):
            d = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if le + d >= target_l:
                ratio = (target_l - le) / max(d, 1e-9)
                x = path[i][0] + (path[i + 1][0] - path[i][0]) * ratio
                y = path[i][1] + (path[i + 1][1] - path[i][1]) * ratio
                return [x, y, i]
            le += d
        return [path[-1][0], path[-1][1], len(path) - 2]

    def _line_collision_check(self, first, second, sample_step=0.5):
        """
        Check if line segment between two points collides with obstacles.

        :param first: (list) [x, y, seg_index] first point
        :param second: (list) [x, y, seg_index] second point
        :param sample_step: (float) Sampling step size along the line
        :return: (bool) True if no collision
        """
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return True
        n_steps = max(1, int(length / sample_step))
        for i in range(n_steps + 1):
            t = i / n_steps
            px = first[0] + t * dx
            py = first[1] + t * dy
            for ox, oy, r in self.obstacle_list:
                if math.hypot(px - ox, py - oy) <= r + self.robot_radius + self.safety_margin:
                    return False
        return True


def main():
    """
    Run RRT* planner demo with path smoothing.
    """
    print("start " + __file__)

    obstacle_list = [
        (10, 8, 3),
        (20, 10, 4),
        (10, 20, 3),
        (25, 25, 4),
        (35, 20, 3),
        (30, 35, 3),
        (15, 40, 3),
        (40, 40, 3),
    ]

    start = (2, 2)
    goal = (50, 50)
    rand_area = (0, 60)

    rrt = RRT(
        start=start,
        goal=goal,
        rand_area=rand_area,
        obstacle_list=obstacle_list,
        expand_dis=1.5,
        path_resolution=0.5,
        goal_sample_rate=15,
        max_iter=3000,
        play_area=[0, 60, 0, 60],
        robot_radius=0.8,
        safety_margin=1.0,
        use_rrt_star=True,
        connect_circle_dist=50.0,
    )

    path, n_expanded, t_elapsed = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        path_length = sum(math.hypot(path[i][0] - path[i - 1][0],
                                     path[i][1] - path[i - 1][1])
                          for i in range(1, len(path)))

        smoothed = rrt.smooth_path(path, max_iter=300)
        smooth_length = sum(math.hypot(smoothed[i][0] - smoothed[i - 1][0],
                                       smoothed[i][1] - smoothed[i - 1][1])
                            for i in range(1, len(smoothed)))
        reduction = (1 - smooth_length / max(path_length, 1e-9)) * 100

        print(f"original path length: {path_length:.2f}")
        print(f"smoothed path length: {smooth_length:.2f}")
        print(f"reduction: {reduction:.1f}%")
        print(f"expanded nodes: {n_expanded}")
        print(f"planning time: {t_elapsed:.4f} s")


if __name__ == '__main__':
    main()
