"""Graph-based SLAM with pose graph optimization

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [GraphSLAM](https://www.ri.cmu.edu/pub_files/2014/7/kaess_icra14.pdf)
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

from SLAM.config import (WHEELBASE, UKF_DT,
    GRAPHSLAM_N_OPTIM_ITER, GRAPHSLAM_LOOP_MIN_INDEX_GAP, GRAPHSLAM_LOOP_DIST_THRESHOLD)
from SLAM.slam_sim import angle_mod, bicycle_dynamics

# === Phase 1: SE(2) utility functions ===

show_animation = True

WHEELBASE_G = WHEELBASE
UKF_DT_G = UKF_DT
LOOP_DIST_THRESHOLD = 1.5
ODOM_NOISE_XY = 0.05
ODOM_NOISE_THETA = np.deg2rad(2.0)
MIN_LOOP_GAP = 30


def se2_inverse(pose):
    """Compute inverse of SE(2) pose

    :param pose: (ndarray) Pose [x, y, theta]
    :return: (ndarray) Inverse pose
    """
    c, s = np.cos(pose[2]), np.sin(pose[2])
    inv_pose = np.array([
        -c * pose[0] - s * pose[1],
        s * pose[0] - c * pose[1],
        -pose[2]
    ])
    return inv_pose


def se2_compose(pose1, pose2):
    """Compose two SE(2) poses

    :param pose1: (ndarray) First pose [x, y, theta]
    :param pose2: (ndarray) Second pose [x, y, theta]
    :return: (ndarray) Composed pose
    """
    c, s = np.cos(pose1[2]), np.sin(pose1[2])
    result = np.array([
        pose1[0] + c * pose2[0] - s * pose2[1],
        pose1[1] + s * pose2[0] + c * pose2[1],
        pose1[2] + pose2[2]
    ])
    result[2] = angle_mod(result[2])
    return result


def se2_difference(pose1, pose2):
    """Compute relative pose from pose1 to pose2

    :param pose1: (ndarray) First pose [x, y, theta]
    :param pose2: (ndarray) Second pose [x, y, theta]
    :return: (ndarray) Relative pose [dx, dy, dtheta]
    """
    c, s = np.cos(pose1[2]), np.sin(pose1[2])
    dx = c * (pose2[0] - pose1[0]) + s * (pose2[1] - pose1[1])
    dy = -s * (pose2[0] - pose1[0]) + c * (pose2[1] - pose1[1])
    dtheta = angle_mod(pose2[2] - pose1[2])
    return np.array([dx, dy, dtheta])


# === Phase 2: GraphSLAM class ===

class GraphSLAM:

    def __init__(self, dt):
        """Initialize empty pose graph

        :param dt: (float) Time step [s]
        """
        self.dt = dt
        self.nodes = []
        self.edges = []
        self._kdtree = None
        self._kdtree_count = 0

    def add_node(self, pose):
        """Add a pose node to the graph

        :param pose: (ndarray) Pose [x, y, theta]
        """
        self.nodes.append(pose.copy())
        self._kdtree = None  # 使缓存失效

    def add_odometry_edge(self, i, j, relative_pose, information):
        """Add odometry constraint

        :param i: (int) Start node index
        :param j: (int) End node index
        :param relative_pose: (ndarray) Relative pose [dx, dy, dtheta]
        :param information: (ndarray) 3x3 information matrix
        """
        self.edges.append((i, j, relative_pose.copy(), information.copy(), False))

    def add_loop_edge(self, i, j, relative_pose, information):
        """Add loop closure constraint

        :param i: (int) Start node index
        :param j: (int) End node index
        :param relative_pose: (ndarray) Relative pose [dx, dy, dtheta]
        :param information: (ndarray) 3x3 information matrix
        """
        self.edges.append((i, j, relative_pose.copy(), information.copy(), True))

    def detect_loop(self, current_pose, min_gap=GRAPHSLAM_LOOP_MIN_INDEX_GAP,
                    dist_threshold=GRAPHSLAM_LOOP_DIST_THRESHOLD):
        """使用缓存 KDTree 进行高效回环检测

        :param current_pose: (ndarray) 当前位姿 [x, y, theta]
        :param min_gap: (int) 最小索引间隔
        :param dist_threshold: (float) 回环检测距离阈值 [m]
        :return: (int or None) 匹配节点索引，或 None
        """
        n = len(self.nodes)
        if n < min_gap + 1:
            return None

        # 增量构建/复用 KDTree
        search_count = n - min_gap
        if self._kdtree is None or self._kdtree_count != search_count:
            positions = np.array(self.nodes[:search_count])[:, :2]
            if len(positions) == 0:
                return None
            self._kdtree = KDTree(positions)
            self._kdtree_count = search_count

        dist, idx = self._kdtree.query(current_pose[:2])

        if dist < dist_threshold:
            return idx
        return None

    def optimize(self, n_iter=GRAPHSLAM_N_OPTIM_ITER):
        """位姿图优化（稀疏矩阵版本）

        :param n_iter: (int) 优化迭代次数
        :return: (ndarray) 优化后节点位姿, shape (n_nodes, 3)
        """
        n = len(self.nodes)
        assert n >= 2, "need at least 2 nodes to optimize"

        x = np.zeros(3 * n)
        for k in range(n):
            x[3 * k:3 * k + 3] = self.nodes[k]

        for iteration in range(n_iter):
            H = sparse.lil_matrix((3 * n, 3 * n))
            b = np.zeros(3 * n)

            for (i, j, z, Omega, is_loop) in self.edges:
                xi = x[3 * i:3 * i + 3]
                xj = x[3 * j:3 * j + 3]
                e, J_i, J_j = _compute_edge_error_and_jacobians(xi, xj, z)

                si = slice(3 * i, 3 * i + 3)
                sj = slice(3 * j, 3 * j + 3)

                H_ii = J_i.T @ Omega @ J_i
                H_ij = J_i.T @ Omega @ J_j
                H_jj = J_j.T @ Omega @ J_j
                b_i = J_i.T @ Omega @ e
                b_j = J_j.T @ Omega @ e

                H[si, si] += H_ii
                H[si, sj] += H_ij
                H[sj, si] += H_ij.T
                H[sj, sj] += H_jj
                b[si] -= b_i
                b[sj] -= b_j

            H[0:3, 0:3] += np.eye(3) * 1e6

            H_csc = H.tocsc()
            dx = spsolve(H_csc, b)

            x += dx

            x[2::3] = angle_mod(x[2::3])

        poses = x.reshape(n, 3)
        for k in range(n):
            self.nodes[k] = poses[k].copy()
        return poses

    def get_trajectory(self):
        """Return current node poses as trajectory

        :return: (ndarray) Node poses, shape (n_nodes, 3)
        """
        return np.array(self.nodes)


def _compute_edge_error_and_jacobians(xi, xj, z):
    """Compute edge error and Jacobians for SE(2) pose graph

    :param xi: (ndarray) Pose of node i [x, y, theta]
    :param xj: (ndarray) Pose of node j [x, y, theta]
    :param z: (ndarray) Measured relative pose [dx, dy, dtheta]
    :return: (tuple) (error, J_i, J_j)
    """
    c_i, s_i = np.cos(xi[2]), np.sin(xi[2])
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]

    e_x = c_i * dx + s_i * dy - z[0]
    e_y = -s_i * dx + c_i * dy - z[1]
    e_theta = angle_mod(xj[2] - xi[2] - z[2])
    e = np.array([e_x, e_y, e_theta])

    J_i = np.array([
        [-c_i, -s_i, -s_i * dx + c_i * dy],
        [s_i, -c_i, -c_i * dx - s_i * dy],
        [0.0, 0.0, -1.0]
    ])

    J_j = np.array([
        [c_i, s_i, 0.0],
        [-s_i, c_i, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return e, J_i, J_j


# === Phase 3: Demo ===

def main():
    dt = UKF_DT_G
    n_steps = 500
    r = 5.0
    omega = 0.2

    t = np.arange(n_steps) * dt
    true_traj = np.column_stack([
        r * np.cos(omega * t),
        r * np.sin(omega * t),
        angle_mod(omega * t + np.pi / 2)
    ])

    slam = GraphSLAM(dt)
    slam.add_node(true_traj[0])

    odom_traj = [true_traj[0].copy()]
    odom_info = np.diag([1.0 / ODOM_NOISE_XY**2,
                         1.0 / ODOM_NOISE_XY**2,
                         1.0 / ODOM_NOISE_THETA**2])
    loop_info = np.diag([1.0 / (ODOM_NOISE_XY * 2)**2,
                         1.0 / (ODOM_NOISE_XY * 2)**2,
                         1.0 / (ODOM_NOISE_THETA * 2)**2])

    loop_edges_vis = []

    for i in range(1, n_steps):
        rel = se2_difference(true_traj[i - 1], true_traj[i])
        noise = np.array([
            np.random.randn() * ODOM_NOISE_XY,
            np.random.randn() * ODOM_NOISE_XY,
            np.random.randn() * ODOM_NOISE_THETA
        ])
        noisy_rel = rel + noise
        noisy_rel[2] = angle_mod(noisy_rel[2])

        new_pose = se2_compose(odom_traj[-1], noisy_rel)
        slam.add_node(new_pose)
        odom_traj.append(new_pose.copy())
        slam.add_odometry_edge(i - 1, i, noisy_rel, odom_info)

        match_idx = slam.detect_loop(new_pose)
        if match_idx is not None:
            loop_rel = se2_difference(true_traj[match_idx], true_traj[i])
            loop_rel += np.array([
                np.random.randn() * ODOM_NOISE_XY * 2,
                np.random.randn() * ODOM_NOISE_XY * 2,
                np.random.randn() * ODOM_NOISE_THETA * 2
            ])
            loop_rel[2] = angle_mod(loop_rel[2])
            slam.add_loop_edge(match_idx, i, loop_rel, loop_info)
            loop_edges_vis.append((match_idx, i))

    odom_traj = np.array(odom_traj)

    n_loop = sum(1 for e in slam.edges if e[4])
    print(f"[GraphSLAM] nodes={len(slam.nodes)}, odom_edges={len(slam.edges)-n_loop}, loop_edges={n_loop}")

    optimized = slam.optimize(n_iter=20)

    pos_err_before = np.sqrt(np.mean((odom_traj[:, 0] - true_traj[:, 0])**2 +
                                      (odom_traj[:, 1] - true_traj[:, 1])**2))
    pos_err_after = np.sqrt(np.mean((optimized[:, 0] - true_traj[:, 0])**2 +
                                     (optimized[:, 1] - true_traj[:, 1])**2))
    print(f"[GraphSLAM] RMSE before opt={pos_err_before:.4f} m, after opt={pos_err_after:.4f} m")

    if not show_animation:
        return

    fig_dir = pathlib.Path(__file__).parent.parent / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Graph-based SLAM Demo")
    ax.grid(True, alpha=0.3)

    true_line, = ax.plot([], [], "-k", linewidth=2, label="True trajectory")
    odom_line, = ax.plot([], [], "--r", linewidth=1.2, alpha=0.7, label="Odometry (before opt)")
    opt_line, = ax.plot([], [], "-b", linewidth=1.5, label="Optimized")
    node_dots, = ax.plot([], [], "b.", markersize=3, alpha=0.5)
    loop_lines = []

    def init():
        true_line.set_data([], [])
        odom_line.set_data([], [])
        opt_line.set_data([], [])
        node_dots.set_data([], [])
        return true_line, odom_line, opt_line, node_dots

    def update(frame):
        step = min(frame + 1, n_steps)
        true_line.set_data(true_traj[:step, 0], true_traj[:step, 1])
        odom_line.set_data(odom_traj[:step, 0], odom_traj[:step, 1])

        if frame >= n_steps - 1:
            opt_line.set_data(optimized[:, 0], optimized[:, 1])
            node_dots.set_data(optimized[:, 0], optimized[:, 1])
            for ln in loop_lines:
                ln.remove()
            loop_lines.clear()
            for (mi, mj) in loop_edges_vis:
                ln, = ax.plot([optimized[mi, 0], optimized[mj, 0]],
                              [optimized[mi, 1], optimized[mj, 1]],
                              "-g", linewidth=0.8, alpha=0.5)
                loop_lines.append(ln)

        all_x = np.concatenate([true_traj[:step, 0], odom_traj[:step, 0]])
        all_y = np.concatenate([true_traj[:step, 1], odom_traj[:step, 1]])
        margin = 1.0
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.legend(loc="upper right", frameon=True, fancybox=True)
        return true_line, odom_line, opt_line, node_dots

    ani = FuncAnimation(fig, update, frames=n_steps + 30, init_func=init,
                        interval=30, blit=False, repeat=False)

    save_path = fig_dir / "slam_graphslam_demo.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"[save] figure saved to {save_path}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
