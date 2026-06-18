"""
Voxel grid filter for point cloud downsampling and coordinate transformation

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np

# === Phase 1: Voxel filter ===

def voxel_filter(points_raw, r_voxel=0.05):
    """
    Downsample point cloud via voxel grid averaging.

    :param points_raw: (np.ndarray) Nx2+ raw point cloud
    :param r_voxel: (float) Voxel side length in meters
    :return: (np.ndarray) Mx2 centroid coordinates per occupied voxel
    """
    assert len(points_raw) > 0, "points_raw must not be empty"
    assert points_raw.ndim == 2 and points_raw.shape[1] >= 2, f"points_raw shape err: {points_raw.shape}"
    assert r_voxel > 0, f"r_voxel must > 0, got {r_voxel}"
    voxel_idx = np.floor(points_raw[:, :2] / r_voxel).astype(np.int32)
    _, unique_inv = np.unique(voxel_idx, axis=0, return_inverse=True)
    n_voxels = unique_inv.max() + 1
    sums_x = np.bincount(unique_inv, weights=points_raw[:, 0], minlength=n_voxels)
    sums_y = np.bincount(unique_inv, weights=points_raw[:, 1], minlength=n_voxels)
    counts = np.bincount(unique_inv, minlength=n_voxels).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    centroids = np.stack([sums_x / counts_safe, sums_y / counts_safe], axis=1)
    return centroids

def transform_to_global(points_local, pose):
    """
    Transform local-frame points to global frame via 2D rigid transform.

    :param points_local: (np.ndarray) Nx2+ points in local frame
    :param pose: (np.ndarray) Length-3 array [x, y, theta] of robot pose
    :return: (np.ndarray) Nx2 points in global frame
    """
    assert points_local.ndim == 2 and points_local.shape[1] >= 2, f"points_local shape err: {points_local.shape}"
    assert pose.shape == (3,), f"pose shape err: {pose.shape}"
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    points_global = (R @ points_local[:, :2].T).T + pose[:2]
    return points_global

def main():
    print("Voxel Filter Demo")
    np.random.seed(42)
    points = np.random.randn(1000, 3) * 2.0
    filtered = voxel_filter(points, r_voxel=0.5)
    print(f"Original: {len(points)} points -> Filtered: {len(filtered)} points")

if __name__ == '__main__':
    main()
