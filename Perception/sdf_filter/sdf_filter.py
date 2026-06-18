"""
SDF (Signed Distance Field) adaptive filter for obstacle detection

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

# === Phase 1: SDF adaptive filter ===

def compute_sdf(grid, resolution=0.05):
    """
    Compute signed distance field from a binary occupancy grid.

    :param grid: (np.ndarray) 2-D occupancy grid (1=occupied, 0=free)
    :param resolution: (float) Grid resolution in meters per cell
    :return: (np.ndarray) 2-D SDF with same shape as grid
    """
    assert grid.ndim == 2, f"grid must 2-D, got ndim={grid.ndim}"
    occupied = (grid > 0.5).astype(np.float64)
    dist_free = distance_transform_edt(1.0 - occupied) * resolution
    dist_occ = distance_transform_edt(occupied) * resolution
    sdf = dist_free - dist_occ
    return sdf

def query_sdf(points_global, sdf, grid_origin, resolution):
    """
    Query SDF values at given global-frame points via nearest-cell lookup.

    :param points_global: (np.ndarray) Nx2+ query points in global frame
    :param sdf: (np.ndarray) 2-D signed distance field
    :param grid_origin: (np.ndarray) Length-2 origin [x0, y0] of the grid
    :param resolution: (float) Grid resolution in meters per cell
    :return: (tuple) (dists, valid) — dists: Nx1 SDF values, valid: Nx1 boolean mask
    """
    assert points_global.ndim == 2 and points_global.shape[1] >= 2
    idx_x = ((points_global[:, 0] - grid_origin[0]) / resolution).astype(np.int32)
    idx_y = ((points_global[:, 1] - grid_origin[1]) / resolution).astype(np.int32)
    n_row, n_col = sdf.shape
    valid = (idx_x >= 0) & (idx_x < n_col) & (idx_y >= 0) & (idx_y < n_row)
    dists = np.full(len(points_global), np.inf)
    dists[valid] = sdf[idx_y[valid], idx_x[valid]]
    return dists, valid

def sdf_adaptive_filter(points_global, sdf, grid_origin, resolution=0.05,
                        tau_sdf=0.15, beta_sdf=3.0, d_near=0.5):
    """
    Filter points using SDF with proximity-adaptive threshold.

    :param points_global: (np.ndarray) Nx2+ points in global frame
    :param sdf: (np.ndarray) 2-D signed distance field
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :param resolution: (float) Grid resolution in m/cell
    :param tau_sdf: (float) Base SDF threshold in meters
    :param beta_sdf: (float) Proximity sensitivity factor
    :param d_near: (float) Near-wall distance for adaptive relaxation in meters
    :return: (tuple) (filtered_points, mask) — kept points and boolean mask
    """
    assert d_near > 0, "d_near must be positive"
    assert tau_sdf > 0, "tau_sdf must be positive"
    assert sdf.ndim == 2, f"sdf must 2-D, got ndim={sdf.ndim}"
    d_sdf, valid = query_sdf(points_global, sdf, grid_origin, resolution)
    proximity = np.maximum(0.0, d_near - d_sdf) / d_near
    tau_adaptive = tau_sdf / (1.0 + beta_sdf * proximity)
    mask_primary = valid & (d_sdf > tau_adaptive)
    mask_recovery = valid & (d_sdf > 0) & (d_sdf <= tau_adaptive) & (proximity > 0.5)
    mask = mask_primary | mask_recovery
    return points_global[mask], mask

def fixed_threshold_filter(points_global, sdf, grid_origin, resolution=0.05, tau=0.15):
    """
    Filter points using a fixed SDF threshold.

    :param points_global: (np.ndarray) Nx2+ points in global frame
    :param sdf: (np.ndarray) 2-D signed distance field
    :param grid_origin: (np.ndarray) Length-2 grid origin [x0, y0]
    :param resolution: (float) Grid resolution in m/cell
    :param tau: (float) Fixed SDF threshold in meters
    :return: (tuple) (filtered_points, mask) — kept points and boolean mask
    """
    d_sdf, valid = query_sdf(points_global, sdf, grid_origin, resolution)
    mask = valid & (d_sdf > tau)
    return points_global[mask], mask

def main():
    print("SDF Filter Demo")
    grid = np.zeros((20, 20))
    grid[5:15, 5:15] = 1.0
    sdf = compute_sdf(grid)
    print(f"SDF computed, shape: {sdf.shape}, min: {sdf.min():.3f}, max: {sdf.max():.3f}")
    points = np.array([[10.0, 10.0], [2.0, 2.0], [5.0, 5.0]])
    grid_origin = np.array([0.0, 0.0])
    filtered, mask = sdf_adaptive_filter(points, sdf, grid_origin, resolution=1.0)
    print(f"Points: {len(points)} -> Filtered: {len(filtered)}")

if __name__ == '__main__':
    main()
