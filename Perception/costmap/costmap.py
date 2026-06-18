"""
Costmap generation with static inflation and dynamic Gaussian layers

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
from scipy.ndimage import binary_dilation
from config import R_INFLATE, SIGMA_DYN, DT_PRED, W_PRED, RESOLUTION

# === Phase 4: Costmap ===

def inflate_static_layer(grid, r_inflate=R_INFLATE, resolution=RESOLUTION):
    """
    Inflate occupied cells in the static grid by morphological dilation.

    :param grid: (np.ndarray) 2-D binary occupancy grid (1=occupied)
    :param r_inflate: (float) Inflation radius in meters
    :param resolution: (float) Grid resolution in m/cell
    :return: (np.ndarray) 2-D inflated static cost layer (0 or 1)
    """
    assert grid.ndim == 2, f"grid must 2-D, got ndim={grid.ndim}"
    inflate_pixels = max(1, int(r_inflate / resolution))
    size = 2 * inflate_pixels + 1
    struct = np.ones((size, size), dtype=bool)
    inflated = binary_dilation(grid.astype(bool), structure=struct).astype(np.float64)
    c_static = (inflated > 0.5).astype(np.float64)
    return c_static

def gaussian_dynamic_layer(trackers_confirmed, grid_shape, grid_origin,
                           resolution=RESOLUTION, sigma_dyn=SIGMA_DYN, dt_pred=DT_PRED,
                           w_pred=W_PRED):
    """
    Generate dynamic cost layer from confirmed trackers using Gaussian kernels.

    :param trackers_confirmed: (list) List of confirmed tracker dicts
    :param grid_shape: (tuple) (n_row, n_col) of the costmap grid
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :param resolution: (float) Grid resolution in m/cell
    :param sigma_dyn: (float) Gaussian spread for dynamic obstacles in meters
    :param dt_pred: (float) Prediction time horizon for velocity extrapolation in seconds
    :param w_pred: (float) Weight for the predicted position Gaussian layer
    :return: (np.ndarray) 2-D dynamic cost layer with same shape as grid_shape
    """
    n_row, n_col = grid_shape
    yy, xx = np.mgrid[0:n_row, 0:n_col]
    wx = grid_origin[0] + xx * resolution
    wy = grid_origin[1] + yy * resolution
    n_trk = len(trackers_confirmed)
    if n_trk == 0:
        return np.zeros((n_row, n_col), dtype=np.float64)
    pos = np.array([[trk['x_trk'][0], trk['x_trk'][1]] for trk in trackers_confirmed])
    vel = np.array([[trk['x_trk'][2], trk['x_trk'][3]] for trk in trackers_confirmed])
    cx, cy = pos[:, 0], pos[:, 1]
    d2 = (wx[None, :, :] - cx[:, None, None]) ** 2 + (wy[None, :, :] - cy[:, None, None]) ** 2
    c_dyn = np.exp(-d2 / (2 * sigma_dyn ** 2 + 1e-18)).sum(axis=0)
    speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
    moving = speed > 0.1
    if moving.any():
        cx_pred = cx[moving] + vel[moving, 0] * dt_pred
        cy_pred = cy[moving] + vel[moving, 1] * dt_pred
        d2_pred = (wx[None, :, :] - cx_pred[:, None, None]) ** 2 + (wy[None, :, :] - cy_pred[:, None, None]) ** 2
        c_dyn += w_pred * np.exp(-d2_pred / (2 * sigma_dyn ** 2 + 1e-18)).sum(axis=0)
    return c_dyn

def fuse_costmap(c_static, c_dyn):
    """
    Fuse static and dynamic cost layers via element-wise maximum.

    :param c_static: (np.ndarray) 2-D static cost layer
    :param c_dyn: (np.ndarray) 2-D dynamic cost layer
    :return: (np.ndarray) 2-D fused costmap
    """
    assert c_static.shape == c_dyn.shape, f"shape mismatch: {c_static.shape} vs {c_dyn.shape}"
    return np.maximum(c_static, c_dyn)

def main():
    print("Costmap Demo")
    static = np.zeros((20, 20))
    static[8:12, 8:12] = 1.0
    inflated = inflate_static_layer(static, r_inflate=2, resolution=1.0)
    print(f"Static layer: {static.sum():.0f} occupied -> Inflated: {inflated.sum():.0f} occupied")
    grid_origin = np.array([0.0, 0.0])
    trackers_confirmed = [{'x_trk': np.array([5.0, 5.0, 0.0, 0.0])}]
    dyn_layer = gaussian_dynamic_layer(trackers_confirmed, (20, 20), grid_origin, resolution=1.0, sigma_dyn=2.0)
    fused = fuse_costmap(inflated, dyn_layer)
    print(f"Fused costmap shape: {fused.shape}, max: {fused.max():.3f}")

if __name__ == '__main__':
    main()
