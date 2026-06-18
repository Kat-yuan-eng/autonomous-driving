"""Sparse pose adjustment (SPA) graph optimizer

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: SPA pose graph optimization ===
# === Phase 2: Sparse matrix construction ===
import copy
import numpy as np

from SLAM.config import N_SUBMAP_SCANS, CAUCHY_C_ODOM, CAUCHY_C_LOOP
from SLAM.mapping.loop_closure_detector import cauchy_loss


def build_pose_graph(nodes, odometry_constraints, loop_constraints):
    """
    Construct pose graph data structure from nodes and edge constraints.

    :param nodes: (list) List of node poses, each np.ndarray shape (3,)
    :param odometry_constraints: (list) List of (i, j, z, Omega, c_param, is_loop) tuples
    :param loop_constraints: (list) List of (i, j, z, Omega, c_param, is_loop) tuples
    :return: (dict) Graph with 'nodes', 'odom', and 'loops' keys
    """
    assert len(nodes) > 0, "pose graph must have at least one node"
    graph = {'nodes': nodes, 'odom': odometry_constraints, 'loops': loop_constraints}
    return graph


def optimize_pose_graph_spa(graph, n_iter=10, damping=1.0):
    """
    Optimize pose graph using Sparse Pose Adjustment with Gauss-Newton and Cauchy robust kernel.

    :param graph: (dict) Pose graph with 'nodes', 'odom', 'loops' keys
    :param n_iter: (int) Maximum number of optimization iterations
    :param damping: (float) Levenberg-Marquardt damping factor
    :return: (list) Optimized node poses
    """
    nodes = copy.deepcopy(graph['nodes'])
    n = len(nodes)
    if n < 2:
        return nodes
    for iteration in range(n_iter):
        H = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)
        total_residual = 0.0
        for (i, j, z, Omega, c_param, is_loop) in graph['odom']:
            e, J_i, J_j = _compute_edge_error(nodes[i], nodes[j], z)
            rho_weight = _robust_weight(np.dot(e, Omega @ e), c_param)
            ei = e.reshape(-1, 1)
            H_i = J_i.T @ Omega @ J_i
            H_j = J_j.T @ Omega @ J_j
            H_ij = J_i.T @ Omega @ J_j
            for idx_a, H_a in [(i, H_i), (j, H_j)]:
                r3a = slice(3 * idx_a, 3 * idx_a + 3)
                H[r3a, r3a] += H_a * rho_weight
            r3i = slice(3 * i, 3 * i + 3)
            r3j = slice(3 * j, 3 * j + 3)
            H[r3i, r3j] += H_ij * rho_weight
            H[r3j, r3i] += H_ij.T * rho_weight
            b[3*i:3*i+3] += (J_i.T @ Omega @ ei.flatten()) * rho_weight
            b[3*j:3*j+3] += (J_j.T @ Omega @ ei.flatten()) * rho_weight
            total_residual += cauchy_loss(np.dot(e, Omega @ e), c_param)
        for (i, j, z, Omega, c_param, _) in graph['loops']:
            e, J_i, J_j = _compute_edge_error(nodes[i], nodes[j], z)
            rho_weight = _robust_weight(np.dot(e, Omega @ e), c_param)
            ei = e.reshape(-1, 1)
            for idx_a, J_a in [(i, J_i), (j, J_j)]:
                r3a = slice(3 * idx_a, 3 * idx_a + 3)
                H[r3a, r3a] += (J_a.T @ Omega @ J_a) * rho_weight
            r3i = slice(3 * i, 3 * i + 3)
            r3j = slice(3 * j, 3 * j + 3)
            J_iTOJ_j = J_i.T @ Omega @ J_j
            H[r3i, r3j] += J_iTOJ_j * rho_weight
            H[r3j, r3i] += J_iTOJ_j.T * rho_weight
            b[3*i:3*i+3] += (J_i.T @ Omega @ ei.flatten()) * rho_weight
            b[3*j:3*j+3] += (J_j.T @ Omega @ ei.flatten()) * rho_weight
            total_residual += cauchy_loss(np.dot(e, Omega @ e), c_param)
        H += damping * np.eye(3 * n) * 1e-6
        H[0:3, 0:3] += np.eye(3) * 1e6
        try:
            dx = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(dx)) < 1e-6:
            break
        for k in range(n):
            nodes[k][0] += dx[3*k]
            nodes[k][1] += dx[3*k+1]
            nodes[k][2] = _norm_angle(nodes[k][2] + dx[3*k+2])
    return nodes


def _compute_edge_error(pose_i, pose_j, z):
    """
    Compute edge error and Jacobians for a pose graph constraint.

    :param pose_i: (np.ndarray) First node pose, shape (3,)
    :param pose_j: (np.ndarray) Second node pose, shape (3,)
    :param z: (np.ndarray) Measured relative pose [dx, dy, dtheta], shape (3,)
    :return: (tuple) (error, J_i, J_j) error vector and Jacobians
    """
    dx = pose_j[0] - pose_i[0]
    dy = pose_j[1] - pose_i[1]
    dt = _norm_angle(pose_j[2] - pose_i[2])
    pred = np.array([dx, dy, dt])
    e = np.array([dx - z[0], dy - z[1], _norm_angle(dt - z[2])])
    J_i = -np.eye(3)
    J_j = np.eye(3)
    return e, J_i, J_j


def _robust_weight(s, c):
    """
    Compute Cauchy robust weight for down-weighting outlier constraints.

    :param s: (float) Squared Mahalanobis distance
    :param c: (float) Cauchy scale parameter
    :return: (float) Robust weight in (0, 1]
    """
    w = 1.0 / (1.0 + s / c**2)
    return w


def _norm_angle(a):
    """
    Normalize angle to [-pi, pi].

    :param a: (float) Angle in radians
    :return: (float) Normalized angle
    """
    return np.arctan2(np.sin(a), np.cos(a))