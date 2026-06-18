"""
Reference trajectory extraction utilities

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np

# === Phase 1: Reference Trajectory Extraction ===

def compute_curvature(x_ref, y_ref):
    """
    Compute discrete curvature along a reference path using three-point formula.

    :param x_ref: (ndarray) Reference x positions [m]
    :param y_ref: (ndarray) Reference y positions [m]
    :return: (ndarray) Curvature array [1/m]
    """
    assert len(x_ref) == len(y_ref), "x_ref and y_ref must have same length"
    n = len(x_ref)
    if n < 3:
        return np.zeros(n)
    dx01 = np.diff(x_ref[:-1])
    dy01 = np.diff(y_ref[:-1])
    dx12 = np.diff(x_ref[1:])
    dy12 = np.diff(y_ref[1:])
    dx02 = x_ref[2:] - x_ref[:-2]
    dy02 = y_ref[2:] - y_ref[:-2]
    d01 = np.sqrt(dx01**2 + dy01**2) + 1e-12
    d12 = np.sqrt(dx12**2 + dy12**2) + 1e-12
    d02 = np.sqrt(dx02**2 + dy02**2) + 1e-12
    area = np.abs(dx01 * dy02 - dx02 * dy01)
    kappa_inner = 2.0 * area / (d01 * d12 * d02 + 1e-18)
    kappa_ref = np.empty(n)
    kappa_ref[1:-1] = kappa_inner
    kappa_ref[0] = kappa_inner[0]
    kappa_ref[-1] = kappa_inner[-1]
    return kappa_ref


def extract_reference(poses, dt_arr, wheelbase=0.3):
    assert poses.ndim == 2 and poses.shape[1] == 3, "poses must be Nx3 (x, y, theta)"
    assert len(dt_arr) == len(poses) - 1, "dt_arr must be len(poses)-1"
    assert np.all(dt_arr > 0), "dt_arr must be positive"

    x_ref = poses[:, 0]
    y_ref = poses[:, 1]
    theta_ref = poses[:, 2]

    dp = np.diff(poses[:, :2], axis=0)
    dist = np.linalg.norm(dp, axis=1)
    dist = np.maximum(dist, 1e-9)

    s_arr = np.concatenate([[0], np.cumsum(dist)])

    v_ref_inner = dist / dt_arr
    v_ref = np.empty(len(poses))
    v_ref[1:-1] = 0.5 * (v_ref_inner[:-1] + v_ref_inner[1:])
    v_ref[0] = v_ref_inner[0]
    v_ref[-1] = v_ref_inner[-1]

    dtheta = np.diff(theta_ref)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    kappa_ref = compute_curvature(x_ref, y_ref)

    return {
        "x_ref": x_ref,
        "y_ref": y_ref,
        "theta_ref": theta_ref,
        "v_ref": v_ref,
        "kappa_ref": kappa_ref,
        "s_arr": s_arr,
    }
