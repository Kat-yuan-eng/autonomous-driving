"""
DBSCAN-based Euclidean clustering for obstacle segmentation

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
from sklearn.cluster import DBSCAN
from config import EPSILON_CLUSTER, N_MIN, R_MARGIN, A_MAX_CLUSTER, R_VOXEL, RESOLUTION

# === Phase 2: Euclidean Clustering ===

def cluster_euclidean(points_dyn, epsilon=EPSILON_CLUSTER, n_min=N_MIN):
    """
    Cluster dynamic points using DBSCAN with Euclidean distance.

    :param points_dyn: (np.ndarray) Nx2+ dynamic obstacle points
    :param epsilon: (float) DBSCAN neighborhood radius in meters
    :param n_min: (int) Minimum cluster size
    :return: (list) List of np.ndarray, each Mx2 cluster points
    """
    assert points_dyn.ndim == 2 and points_dyn.shape[1] >= 2, f"points_dyn shape err: {points_dyn.shape}"
    assert epsilon > 0 and n_min >= 1, f"epsilon={epsilon}, n_min={n_min} invalid"
    n_pts = len(points_dyn)
    if n_pts == 0:
        return []
    labels = DBSCAN(eps=epsilon, min_samples=n_min).fit_predict(points_dyn[:, :2])
    unique_labels = np.unique(labels[labels >= 0])
    clusters = [points_dyn[labels == c] for c in unique_labels if (labels == c).sum() >= n_min]
    return clusters

def compute_cluster_attrs(clusters, r_voxel=R_VOXEL, r_margin=R_MARGIN, a_max=A_MAX_CLUSTER):
    """
    Compute bounding attributes for each cluster (center, radius, point count).

    :param clusters: (list) List of np.ndarray, each Mx2 cluster points
    :param r_voxel: (float) Voxel size for area estimation in meters
    :param r_margin: (float) Safety margin added to cluster radius in meters
    :param a_max: (float) Maximum cluster area threshold in m²
    :return: (list) List of dicts with keys 'center', 'radius', 'n_pts'
    """
    assert len(clusters) >= 0
    results = []
    for cluster_pts in clusters:
        assert len(cluster_pts) >= 1
        center = cluster_pts[:, :2].mean(axis=0)
        radius = np.linalg.norm(cluster_pts[:, :2] - center, axis=1).max() + r_margin
        area = len(cluster_pts) * r_voxel ** 2
        if area < a_max:
            results.append({"center": center, "radius": radius, "n_pts": len(cluster_pts)})
    return results

def main():
    print("Euclidean Cluster Demo")
    np.random.seed(42)
    pts_a = np.random.randn(30, 2) + np.array([2, 2])
    pts_b = np.random.randn(30, 2) + np.array([-2, -2])
    points = np.vstack([pts_a, pts_b])
    clusters = cluster_euclidean(points, epsilon=1.0, n_min=3)
    print(f"Points: {len(points)} -> Clusters: {len(clusters)}")

if __name__ == '__main__':
    main()
