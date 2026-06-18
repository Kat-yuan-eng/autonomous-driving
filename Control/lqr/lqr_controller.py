"""
LQR steering controller with adaptive Q/R matrices and gain precomputation

author: Kat-yuan-eng (RuiWen Liao)
"""

import os

import numpy as np
from Control.config import (WHEELBASE, V_MAX, DELTA_MAX, A_MAX, DELTA_DOT_MAX,
    DT, DV_TABLE, V_TABLE_MIN, V_TABLE_MAX, Q_LQR, R_LQR, T_LA_FF, L_LA_MIN,
    KAPPA_LOW, KAPPA_HIGH, Q_LAT_MIN, Q_LAT_MAX, Q_THETA_MIN, Q_THETA_MAX,
    R_DELTA_MIN, R_DELTA_MAX, LOOKAHEAD_RANGE_FACTOR, LOOKAHEAD_DECAY_RATE)

# === Phase 1: Offline Precomputation ===

def solve_dare(A, B, Q, R, eps=1e-9, max_iter=10000):
    """
    Solve the Discrete Algebraic Riccati Equation iteratively.

    :param A: (ndarray) State transition matrix (n x n)
    :param B: (ndarray) Control input matrix (n x m)
    :param Q: (ndarray) State weight matrix (n x n)
    :param R: (ndarray) Control weight matrix (m x m)
    :param eps: (float) Convergence tolerance
    :param max_iter: (int) Maximum iteration count
    :return: (ndarray) Solution matrix P (n x n)
    """
    n_state = A.shape[0]
    n_ctrl = B.shape[1]
    assert A.shape == (n_state, n_state), f"A must be square, got {A.shape}"
    assert B.shape[0] == n_state, f"B row dim must match A, got {B.shape}"
    assert Q.shape == (n_state, n_state), f"Q must be {n_state}x{n_state}, got {Q.shape}"
    assert R.shape == (n_ctrl, n_ctrl), f"R must be {n_ctrl}x{n_ctrl}, got {R.shape}"
    P = Q.copy().astype(float)
    R_reg = R + 1e-12 * np.eye(n_ctrl)
    for _ in range(max_iter):
        BtPB = B.T @ P @ B
        P_new = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R_reg + BtPB) @ B.T @ P @ A + Q
        diff = np.max(np.abs(P_new - P))
        if diff < eps:
            return P_new
        P = P_new
    if diff >= eps:
        raise RuntimeError(f"DARE did not converge after {max_iter} iterations, residual={diff:.2e}")
    return P

def build_state_matrix(v, dt=0.005, wheelbase=0.3):
    """
    Build linearized bicycle model state-space matrices A and B.

    :param v: (float) Current longitudinal speed [m/s]
    :param dt: (float) Discretization time step [s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :return: (tuple) (A, B) state and control matrices
    """
    assert v >= 0, f"v must be non-negative, got {v}"
    assert dt > 0, f"dt must be positive, got {dt}"
    assert wheelbase > 0, f"wheelbase must be positive, got {wheelbase}"
    A = np.array([
        [1.0, dt, 0.0, 0.0, 0.0],
        [0.0, 0.0, v, 0.0, 0.0],
        [0.0, 0.0, 1.0, dt, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [v / max(wheelbase, 1e-9), 0.0],
        [0.0, dt],
    ])
    return A, B

def adaptive_Q(kappa, Q_base, kappa_low=KAPPA_LOW, kappa_high=KAPPA_HIGH,
               q_lat_min=Q_LAT_MIN, q_lat_max=Q_LAT_MAX,
               q_theta_min=Q_THETA_MIN, q_theta_max=Q_THETA_MAX):
    """
    Compute curvature-adaptive state weight matrix Q.

    :param kappa: (float) Path curvature [1/m]
    :param Q_base: (ndarray) Base Q matrix (5 x 5)
    :param kappa_low: (float) Low curvature threshold [1/m]
    :param kappa_high: (float) High curvature threshold [1/m]
    :param q_lat_min: (float) Minimum lateral error weight
    :param q_lat_max: (float) Maximum lateral error weight
    :param q_theta_min: (float) Minimum heading error weight
    :param q_theta_max: (float) Maximum heading error weight
    :return: (ndarray) Adapted Q matrix (5 x 5)
    """
    alpha_k = np.clip((np.abs(kappa) - kappa_low) / (kappa_high - kappa_low + 1e-12), 0.0, 1.0)
    Q = Q_base.copy()
    Q[0, 0] = q_lat_min + alpha_k * (q_lat_max - q_lat_min)
    Q[2, 2] = q_theta_min + alpha_k * (q_theta_max - q_theta_min)
    return Q

def adaptive_R(kappa, R_base, r_delta_min=R_DELTA_MIN, r_delta_max=R_DELTA_MAX,
               kappa_low=KAPPA_LOW, kappa_high=KAPPA_HIGH):
    """
    Compute curvature-adaptive control weight matrix R.

    :param kappa: (float) Path curvature [1/m]
    :param R_base: (ndarray) Base R matrix (2 x 2)
    :param r_delta_min: (float) Minimum steering weight
    :param r_delta_max: (float) Maximum steering weight
    :param kappa_low: (float) Low curvature threshold [1/m]
    :param kappa_high: (float) High curvature threshold [1/m]
    :return: (ndarray) Adapted R matrix (2 x 2)
    """
    alpha_k = np.clip((np.abs(kappa) - kappa_low) / (kappa_high - kappa_low + 1e-12), 0.0, 1.0)
    R = R_base.copy()
    R[0, 0] = r_delta_max - alpha_k * (r_delta_max - r_delta_min)
    return R

def precompute_lqr_gains(v_min=0.1, v_max=2.5, dv=0.1, dt=0.005, wheelbase=0.3, Q=None, R=None):
    """
    Precompute LQR feedback gains over a velocity table.

    :param v_min: (float) Minimum velocity in table [m/s]
    :param v_max: (float) Maximum velocity in table [m/s]
    :param dv: (float) Velocity step [m/s]
    :param dt: (float) Discretization time step [s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :param Q: (ndarray or None) State weight matrix, defaults to Q_LQR
    :param R: (ndarray or None) Control weight matrix, defaults to R_LQR
    :return: (dict) Velocity-to-gain mapping {v: K}
    """
    assert v_min > 0, f"v_min must be positive, got {v_min}"
    assert v_max > v_min, f"v_max must exceed v_min, got v_max={v_max} v_min={v_min}"
    assert dv > 0, f"dv must be positive, got {dv}"
    Q_default = Q_LQR
    R_default = R_LQR
    Q = Q_default if Q is None else Q
    R = R_default if R is None else R
    gains = {}
    n_v = int(round((v_max - v_min) / dv)) + 1
    for i in range(n_v):
        v_k = round(v_min + i * dv, 10)
        A, B = build_state_matrix(v_k, dt, wheelbase)
        P = solve_dare(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        gains[v_k] = K
    return gains

def precompute_adaptive_gains(v_min=0.1, v_max=2.5, dv=0.1, dt=0.005, wheelbase=0.3):
    """
    Precompute adaptive LQR gains for straight and curved path segments.

    :param v_min: (float) Minimum velocity in table [m/s]
    :param v_max: (float) Maximum velocity in table [m/s]
    :param dv: (float) Velocity step [m/s]
    :param dt: (float) Discretization time step [s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :return: (tuple) (gains_straight, gains_curve) velocity-to-gain dicts
    """
    assert v_min > 0, f"v_min must be positive, got {v_min}"
    assert v_max > v_min, f"v_max must exceed v_min, got v_max={v_max} v_min={v_min}"
    gains_straight = {}
    gains_curve = {}
    Q_straight = adaptive_Q(0.0, Q_LQR)
    Q_curve = adaptive_Q(KAPPA_HIGH + 1.0, Q_LQR)
    R_straight = adaptive_R(0.0, R_LQR)
    R_curve = adaptive_R(KAPPA_HIGH + 1.0, R_LQR)
    n_v = int(round((v_max - v_min) / dv)) + 1
    for i in range(n_v):
        v_k = round(v_min + i * dv, 10)
        A, B = build_state_matrix(v_k, dt, wheelbase)
        P_s = solve_dare(A, B, Q_straight, R_straight)
        K_s = np.linalg.inv(R_straight + B.T @ P_s @ B) @ B.T @ P_s @ A
        gains_straight[v_k] = K_s
        P_c = solve_dare(A, B, Q_curve, R_curve)
        K_c = np.linalg.inv(R_curve + B.T @ P_c @ B) @ B.T @ P_c @ A
        gains_curve[v_k] = K_c
    return gains_straight, gains_curve

def save_gains(gains_dict, filepath):
    """
    Save precomputed LQR gains to a .npz file.

    :param gains_dict: (dict) Velocity-to-gain mapping
    :param filepath: (str) Output file path
    """
    assert len(gains_dict) > 0, "gains_dict must not be empty"
    v_arr = np.array(sorted(gains_dict.keys()))
    K_arr = np.stack([gains_dict[vk] for vk in sorted(gains_dict.keys())])
    np.savez(filepath, v_arr=v_arr, K_arr=K_arr)

def load_gains(filepath):
    """
    Load precomputed LQR gains from a .npz file.

    :param filepath: (str) Input file path
    :return: (dict) Velocity-to-gain mapping
    """
    assert os.path.isfile(filepath), f"gains file not found: {filepath}, run precompute first"
    gains_pack_raw = np.load(filepath)
    v_arr = gains_pack_raw['v_arr']
    K_arr = gains_pack_raw['K_arr']
    return {round(float(v), 10): K_arr[i] for i, v in enumerate(v_arr)}

def save_adaptive_gains(gains_straight, gains_curve, filepath):
    """
    Save adaptive LQR gains (straight + curve) to a .npz file.

    :param gains_straight: (dict) Straight-path velocity-to-gain mapping
    :param gains_curve: (dict) Curved-path velocity-to-gain mapping
    :param filepath: (str) Output file path
    """
    assert len(gains_straight) > 0, "gains_straight must not be empty"
    assert len(gains_curve) > 0, "gains_curve must not be empty"
    v_arr = np.array(sorted(gains_straight.keys()))
    K_s_arr = np.stack([gains_straight[vk] for vk in sorted(gains_straight.keys())])
    K_c_arr = np.stack([gains_curve[vk] for vk in sorted(gains_curve.keys())])
    np.savez(filepath, v_arr=v_arr, K_straight_arr=K_s_arr, K_curve_arr=K_c_arr)

def load_adaptive_gains(filepath):
    """
    Load adaptive LQR gains (straight + curve) from a .npz file.

    :param filepath: (str) Input file path
    :return: (tuple) (gains_straight, gains_curve) velocity-to-gain dicts
    """
    assert os.path.isfile(filepath), f"gains file not found: {filepath}, run precompute first"
    pack = np.load(filepath)
    v_arr = pack['v_arr']
    K_s_arr = pack['K_straight_arr']
    K_c_arr = pack['K_curve_arr']
    gains_straight = {round(float(v), 10): K_s_arr[i] for i, v in enumerate(v_arr)}
    gains_curve = {round(float(v), 10): K_c_arr[i] for i, v in enumerate(v_arr)}
    return gains_straight, gains_curve

# === Phase 2: Error Computation ===

def find_nearest_point(x, y, x_ref, y_ref, idx_prev, n_search=50):
    """
    Find the nearest reference point index using local search.

    :param x: (float) Current x position [m]
    :param y: (float) Current y position [m]
    :param x_ref: (ndarray) Reference x positions
    :param y_ref: (ndarray) Reference y positions
    :param idx_prev: (int) Previous nearest point index
    :param n_search: (int) Search window size
    :return: (int) Nearest reference point index
    """
    n = len(x_ref)
    assert n > 0, "reference trajectory must not be empty"
    assert 0 <= idx_prev < n, f"idx_prev out of range [0, {n}), got {idx_prev}"
    idx_end = min(idx_prev + n_search, n)
    dx = x_ref[idx_prev:idx_end] - x
    dy = y_ref[idx_prev:idx_end] - y
    dist_sq = dx * dx + dy * dy
    idx_local = np.argmin(dist_sq)
    return idx_prev + int(idx_local)

def compute_lateral_error(x, y, theta_ref, x_ref, y_ref):
    """
    Compute signed lateral error relative to reference heading.

    :param x: (float) Current x position [m]
    :param y: (float) Current y position [m]
    :param theta_ref: (float) Reference heading [rad]
    :param x_ref: (float) Reference x position [m]
    :param y_ref: (float) Reference y position [m]
    :return: (float) Signed lateral error [m]
    """
    nx = -np.sin(theta_ref)
    ny = np.cos(theta_ref)
    return (x - x_ref) * nx + (y - y_ref) * ny

def compute_heading_error(theta, theta_ref):
    """
    Compute normalized heading error wrapped to [-pi, pi].

    :param theta: (float) Current heading [rad]
    :param theta_ref: (float) Reference heading [rad]
    :return: (float) Heading error [rad]
    """
    e = theta - theta_ref
    return np.arctan2(np.sin(e), np.cos(e))

def compute_error_state(x, y, theta, v, x_ref, y_ref, theta_ref, v_ref, e_lat_prev, e_theta_prev, de_lat_prev=0.0, de_theta_prev=0.0, dt=0.005, alpha_f=0.3):
    """
    Compute full 5-dim error state vector with filtered derivatives.

    :param x: (float) Current x position [m]
    :param y: (float) Current y position [m]
    :param theta: (float) Current heading [rad]
    :param v: (float) Current speed [m/s]
    :param x_ref: (float) Reference x position [m]
    :param y_ref: (float) Reference y position [m]
    :param theta_ref: (float) Reference heading [rad]
    :param v_ref: (float) Reference speed [m/s]
    :param e_lat_prev: (float) Previous lateral error [m]
    :param e_theta_prev: (float) Previous heading error [rad]
    :param de_lat_prev: (float) Previous filtered lateral error rate [m/s]
    :param de_theta_prev: (float) Previous filtered heading error rate [rad/s]
    :param dt: (float) Time step [s]
    :param alpha_f: (float) Low-pass filter coefficient
    :return: (tuple) (e, e_lat, e_theta, de_lat, de_theta)
    """
    assert dt > 0, f"dt must be positive, got {dt}"
    e_lat = compute_lateral_error(x, y, theta_ref, x_ref, y_ref)
    e_theta = compute_heading_error(theta, theta_ref)
    e_v = v - v_ref
    de_lat_raw = (e_lat - e_lat_prev) / (dt + 1e-12)
    de_theta_raw = (e_theta - e_theta_prev) / (dt + 1e-12)
    de_lat = alpha_f * de_lat_raw + (1 - alpha_f) * de_lat_prev
    de_theta = alpha_f * de_theta_raw + (1 - alpha_f) * de_theta_prev
    e = np.array([e_lat, de_lat, e_theta, de_theta, e_v])
    return e, e_lat, e_theta, de_lat, de_theta

# === Phase 3: LQR Control ===

def interpolate_gain(v, gains_dict, dv=0.1):
    """
    Linearly interpolate LQR gain for a given velocity.

    :param v: (float) Current speed [m/s]
    :param gains_dict: (dict) Velocity-to-gain mapping
    :param dv: (float) Velocity step used in precomputation [m/s]
    :return: (ndarray) Interpolated gain matrix K
    """
    assert len(gains_dict) >= 2, "gains_dict must contain at least 2 entries for interpolation"
    v_arr = np.array(sorted(gains_dict.keys()))
    K_arr = np.stack([gains_dict[vk] for vk in sorted(gains_dict.keys())])
    v_min, v_max = v_arr[0], v_arr[-1]
    v_clip = np.clip(v, v_min, v_max)
    idx = int((v_clip - v_min) / (dv + 1e-12))
    idx = min(idx, len(v_arr) - 2)
    alpha = (v_clip - v_arr[idx]) / (v_arr[idx + 1] - v_arr[idx] + 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)
    K = (1 - alpha) * K_arr[idx] + alpha * K_arr[idx + 1]
    return K

def interpolate_adaptive_gain(v, kappa, gains_straight, gains_curve, dv=0.1,
                              kappa_low=KAPPA_LOW, kappa_high=KAPPA_HIGH):
    """
    Interpolate adaptive LQR gain blending straight and curve tables.

    :param v: (float) Current speed [m/s]
    :param kappa: (float) Path curvature [1/m]
    :param gains_straight: (dict) Straight-path velocity-to-gain mapping
    :param gains_curve: (dict) Curved-path velocity-to-gain mapping
    :param dv: (float) Velocity step [m/s]
    :param kappa_low: (float) Low curvature threshold [1/m]
    :param kappa_high: (float) High curvature threshold [1/m]
    :return: (ndarray) Blended gain matrix K
    """
    K_s = interpolate_gain(v, gains_straight, dv)
    K_c = interpolate_gain(v, gains_curve, dv)
    alpha_k = np.clip((np.abs(kappa) - kappa_low) / (kappa_high - kappa_low + 1e-12), 0.0, 1.0)
    K = (1 - alpha_k) * K_s + alpha_k * K_c
    return K

def feedforward_compensate(kappa_ref, v_ref, v, tau_ff=0.5, wheelbase=0.3):
    """
    Compute feedforward steering and acceleration from curvature.

    :param kappa_ref: (float) Reference curvature [1/m]
    :param v_ref: (float) Reference speed [m/s]
    :param v: (float) Current speed [m/s]
    :param tau_ff: (float) Feedforward time constant [s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :return: (tuple) (delta_ff, a_ff) feedforward steering [rad] and acceleration [m/s^2]
    """
    assert tau_ff > 0, f"tau_ff must be positive, got {tau_ff}"
    assert wheelbase > 0, f"wheelbase must be positive, got {wheelbase}"
    delta_ff = np.arctan(wheelbase * kappa_ref)
    a_ff = (v_ref - v) / (tau_ff + 1e-12)
    return delta_ff, a_ff

def lookahead_curvature(kappa_ref, idx, v, s_arr, t_la=0.4, l_min=0.3, range_factor=LOOKAHEAD_RANGE_FACTOR, decay_rate=LOOKAHEAD_DECAY_RATE):
    """
    Compute weighted lookahead curvature for feedforward compensation.

    :param kappa_ref: (ndarray) Reference curvature array [1/m]
    :param idx: (int) Current nearest point index
    :param v: (float) Current speed [m/s]
    :param s_arr: (ndarray) Cumulative arc-length array [m]
    :param t_la: (float) Lookahead time [s]
    :param l_min: (float) Minimum lookahead distance [m]
    :param range_factor: (float) Search range multiplier
    :param decay_rate: (float) Exponential decay rate for weighting
    :return: (float) Weighted lookahead curvature [1/m]
    """
    assert idx >= 0, f"idx must be non-negative, got {idx}"
    assert len(kappa_ref) == len(s_arr), f"kappa_ref and s_arr must have same length"
    s_la = v * t_la + l_min
    s_from_idx = s_arr - s_arr[idx]
    mask = (s_from_idx >= 0) & (s_from_idx <= s_la * range_factor)
    if not np.any(mask):
        return kappa_ref[min(idx, len(kappa_ref) - 1)]
    weights = np.exp(-decay_rate * s_from_idx[mask] / (s_la + 1e-12))
    kappa_la = np.sum(weights * kappa_ref[mask]) / (np.sum(weights) + 1e-12)
    return kappa_la

def lqr_control_adaptive(e, v, kappa, gains_straight, gains_curve, dv=0.1, delta_ff=0.0, a_ff=0.0):
    """
    Compute adaptive LQR control with feedforward compensation.

    :param e: (ndarray) 5-dim error state vector
    :param v: (float) Current speed [m/s]
    :param kappa: (float) Path curvature [1/m]
    :param gains_straight: (dict) Straight-path velocity-to-gain mapping
    :param gains_curve: (dict) Curved-path velocity-to-gain mapping
    :param dv: (float) Velocity step [m/s]
    :param delta_ff: (float) Feedforward steering [rad]
    :param a_ff: (float) Feedforward acceleration [m/s^2]
    :return: (tuple) (delta, accel) steering [rad] and acceleration [m/s^2]
    """
    assert e.shape == (5,), f"error state must be 5-dim, got {e.shape}"
    K = interpolate_adaptive_gain(v, kappa, gains_straight, gains_curve, dv)
    u_ff = np.array([delta_ff, a_ff])
    u_cmd = u_ff - K @ e
    delta = np.clip(u_cmd[0], -DELTA_MAX, DELTA_MAX)
    accel = np.clip(u_cmd[1], -A_MAX, A_MAX)
    return float(delta), float(accel)

def main():
    print("LQR Controller Demo")
    v_sample = 1.0
    A, B = build_state_matrix(v_sample)
    print(f"State matrix A at v={v_sample} m/s:\n{A}")
    print(f"Input matrix B at v={v_sample} m/s:\n{B}")
    gains_straight, gains_curve = precompute_adaptive_gains()
    print(f"Precomputed {len(gains_straight)} gain entries")
    for v_k in [0.5, 1.0, 2.0]:
        K = interpolate_adaptive_gain(v_k, 0.0, gains_straight, gains_curve)
        print(f"K at v={v_k}, kappa=0.0:\n{K}")

if __name__ == '__main__':
    main()
