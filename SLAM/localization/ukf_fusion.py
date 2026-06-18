"""Unscented Kalman Filter for multi-source pose fusion

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: UKF initialization ===
# === Phase 2: Sigma point generation ===
# === Phase 3: Prediction step (50Hz) ===
# === Phase 4: Multi-source observation update ===
# === Phase 5: Adaptive noise tuning ===
import numpy as np

from SLAM.config import (UKF_ALPHA, UKF_BETA, UKF_KAPPA, UKF_DIM, UKF_DT,
    UKF_Q, UKF_R_CARTO_BASE, UKF_BETA_A, UKF_BETA_W,
    UKF_Q_SCALE_HIGH, UKF_Q_SCALE_LOW, UKF_INNOVATION_THRESH, UKF_Q_SCALE_NORMAL,
    UKF_R_CARTO_TIME_DECAY, UKF_R_CARTO_TIME_MAX,
    UKF_ALPHA_LOW, UKF_ALPHA_HIGH, UKF_OMEGA_HIGH_THRESH)

def ukf_init(x0=np.zeros(UKF_DIM), P0=None):
    if P0 is None:
        P0 = np.eye(UKF_DIM) * 1.0
    assert x0.shape == (UKF_DIM,), f"initial state must be {UKF_DIM}-dim, got {x0.shape}"
    assert P0.shape == (UKF_DIM, UKF_DIM), f"initial cov must be {UKF_DIM}x{UKF_DIM}, got {P0.shape}"
    n = UKF_DIM
    lam = UKF_ALPHA**2 * (n + UKF_KAPPA) - n
    W_m = np.full(2 * n + 1, 0.5 / (n + lam))
    W_c = np.full(2 * n + 1, 0.5 / (n + lam))
    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - UKF_ALPHA**2 + UKF_BETA)
    return x0.copy(), P0.copy(), W_m, W_c


def _ensure_positive_definite(P, eps=1e-6):
    n = P.shape[0]
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def ukf_adaptive_alpha(omega):
    omega_abs = abs(omega)
    ratio = min(omega_abs / UKF_OMEGA_HIGH_THRESH, 1.0)
    alpha = UKF_ALPHA_LOW + (UKF_ALPHA_HIGH - UKF_ALPHA_LOW) * ratio
    return round(alpha, 6)


def ukf_generate_sigma(x, P, W_m, W_c, omega=0.0):
    n = len(x)
    alpha = ukf_adaptive_alpha(omega)
    lam = alpha**2 * (n + UKF_KAPPA) - n
    P_reg = _ensure_positive_definite(P, eps=1e-9)
    scaled = (n + lam) * P_reg
    sqrt_P = None
    for reg in [1e-9, 1e-6, 1e-4, 1e-3, 1e-2]:
        P_try = scaled + np.eye(n) * reg
        eigvals = np.linalg.eigvalsh(P_try)
        if np.all(eigvals > 0):
            sqrt_P = np.linalg.cholesky(P_try)
            break
    if sqrt_P is None:
        sqrt_P = np.linalg.cholesky(np.eye(n))
    sigma = np.zeros((2 * n + 1, n))
    sigma[0] = x
    for i in range(n):
        sigma[i + 1] = x + sqrt_P[:, i]
        sigma[n + i + 1] = x - sqrt_P[:, i]
    return sigma


def ukf_predict(sigma, W_m, W_c, dt=UKF_DT, innovation_prev=None):
    n = UKF_DIM
    sigma_pred = np.zeros_like(sigma)
    for i in range(2 * n + 1):
        s = sigma[i]
        sigma_pred[i] = _motion_model(s, dt)
    x_pred = np.dot(W_m, sigma_pred)
    P_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        diff = sigma_pred[i] - x_pred
        diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
        P_pred += W_c[i] * np.outer(diff, diff)
    Q_use = ukf_adaptive_Q(innovation_prev, P_pred, UKF_Q) if innovation_prev is not None else UKF_Q
    P_pred += Q_use
    P_pred = 0.5 * (P_pred + P_pred.T)
    P_pred = _ensure_positive_definite(P_pred, eps=1e-9)
    return x_pred, P_pred, sigma_pred


def ukf_update_carto(x_pred, P_pred, sigma_pred, W_m, W_c, z_carto, R_carto=None):
    if R_carto is None:
        R_carto = UKF_R_CARTO_BASE
    return _ukf_update_generic(x_pred, P_pred, sigma_pred, W_m, W_c,
                               z_carto, R_carto, _h_carto)


def ukf_adaptive_Q(innovation, P_pred, Q_base):
    assert innovation.shape == (UKF_DIM,), f"innovation must be {UKF_DIM}-dim, got {innovation.shape}"
    assert P_pred.shape == (UKF_DIM, UKF_DIM), f"P_pred must be {UKF_DIM}x{UKF_DIM}, got {P_pred.shape}"
    assert Q_base.shape == (UKF_DIM, UKF_DIM), f"Q_base must be {UKF_DIM}x{UKF_DIM}, got {Q_base.shape}"
    innov_norm = np.linalg.norm(innovation)
    sigma_pred = np.sqrt(np.trace(P_pred) / P_pred.shape[0])
    ratio = innov_norm / max(sigma_pred, 1e-9)
    if ratio > UKF_INNOVATION_THRESH:
        scale = UKF_Q_SCALE_HIGH
    elif ratio < UKF_INNOVATION_THRESH / 3.0:
        scale = UKF_Q_SCALE_LOW
    else:
        scale = UKF_Q_SCALE_NORMAL
    Q_adapt = np.round(Q_base * scale, 6)
    assert Q_adapt.shape == Q_base.shape, f"Q_adapt shape mismatch: {Q_adapt.shape} vs {Q_base.shape}"
    return Q_adapt


def ukf_adaptive_R_carto(score, time_since_match):
    time_clipped = min(time_since_match, UKF_R_CARTO_TIME_MAX)
    scale_pos = 1.0 + (1.0 - score) * 1.5 + time_clipped * UKF_R_CARTO_TIME_DECAY
    scale_heading = 0.10 + (1.0 - score) * 0.5 + time_clipped * UKF_R_CARTO_TIME_DECAY * 0.4
    R_base = UKF_R_CARTO_BASE.copy()
    R_adapt = np.diag([R_base[0, 0] * scale_pos,
                       R_base[1, 1] * scale_pos,
                       R_base[2, 2] * scale_heading])
    return np.round(R_adapt, 9)


def _ukf_update_generic(x_pred, P_pred, sigma_pred, W_m, W_c, z, R, h_func):
    n = UKF_DIM
    n_sigma = 2 * n + 1
    z_dim = len(z)
    Z = np.zeros((n_sigma, z_dim))
    for i in range(n_sigma):
        Z[i] = h_func(sigma_pred[i])
    z_pred = np.dot(W_m, Z)
    S = np.zeros((z_dim, z_dim))
    C = np.zeros((n, z_dim))
    for i in range(n_sigma):
        dz = Z[i] - z_pred
        dx = sigma_pred[i] - x_pred
        dx[2] = np.arctan2(np.sin(dx[2]), np.cos(dx[2]))
        if z_dim >= 3:
            dz[2] = np.arctan2(np.sin(dz[2]), np.cos(dz[2]))
        S += W_c[i] * np.outer(dz, dz)
        C += W_c[i] * np.outer(dx, dz)
    S += R
    S = 0.5 * (S + S.T) + np.eye(z_dim) * 1e-9
    assert not (np.any(np.isnan(S)) or np.any(np.isinf(S))), "S became NaN/Inf"
    K = C @ np.linalg.inv(S)
    innovation = z - z_pred
    if z_dim >= 3:
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))
    x_upd = x_pred + K @ innovation
    P_upd = P_pred - K @ S @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    P_upd = _ensure_positive_definite(P_upd, eps=1e-6)
    x_upd[2] = np.arctan2(np.sin(x_upd[2]), np.cos(x_upd[2]))
    return x_upd, P_upd


def _motion_model(x, dt):
    return np.array([
        x[0] + x[3] * dt,
        x[1] + x[4] * dt,
        x[2] + x[5] * dt,
        x[3], x[4], x[5]
    ])


def _motion_model_odom(x, v_m, w_m, dt):
    return np.array([
        x[0] + v_m * np.cos(x[2]) * dt,
        x[1] + v_m * np.sin(x[2]) * dt,
        x[2] + w_m * dt,
        v_m * np.cos(x[2]),
        v_m * np.sin(x[2]),
        w_m
    ])


def ukf_predict_odom(sigma, W_m, W_c, v_m, w_m, dt=UKF_DT, innovation_prev=None):
    n = UKF_DIM
    sigma_pred = np.zeros_like(sigma)
    for i in range(2 * n + 1):
        sigma_pred[i] = _motion_model_odom(sigma[i], v_m, w_m, dt)
    x_pred = np.dot(W_m, sigma_pred)
    P_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        diff = sigma_pred[i] - x_pred
        diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
        P_pred += W_c[i] * np.outer(diff, diff)
    Q_use = ukf_adaptive_Q(innovation_prev, P_pred, UKF_Q) if innovation_prev is not None else UKF_Q
    P_pred += Q_use
    P_pred = 0.5 * (P_pred + P_pred.T)
    P_pred = _ensure_positive_definite(P_pred, eps=1e-9)
    return x_pred, P_pred, sigma_pred


def _h_carto(x):
    return x[0:3]
