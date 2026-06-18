"""Submap builder with voxel filtering and occupancy grid

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Voxel filter ===
# === Phase 2: Probability grid update ===
# === Phase 3: Submap switching ===
import numpy as np
from collections import deque

from SLAM.config import (VOXEL_SIZE, OCC_PROB_OCCUPIED, OCC_PROB_FREE,
    PROB_OCCUPIED_INIT, N_SUBMAP_SCANS)


def voxel_filter(points, voxel_size=VOXEL_SIZE):
    """
    Downsample point cloud by keeping one point per voxel cell.

    :param points: (np.ndarray) Input point cloud, shape (N, 2+)
    :param voxel_size: (float) Voxel grid size in meters
    :return: (np.ndarray) Filtered point cloud
    """
    assert points.ndim == 2 and points.shape[1] >= 2, f"points must be Nx2+, got shape {points.shape}"
    voxel_indices = np.floor(points[:, :2] / voxel_size).astype(int)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_idx]


def prob_to_log_odds(p):
    """
    Convert probability to log-odds representation.

    :param p: (float or np.ndarray) Probability value(s) in [0, 1]
    :return: (float or np.ndarray) Log-odds value(s)
    """
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def log_odds_to_prob(l):
    """
    Convert log-odds to probability representation.

    :param l: (float or np.ndarray) Log-odds value(s)
    :return: (float or np.ndarray) Probability value(s) in [0, 1]
    """
    return 1.0 - 1.0 / (1.0 + np.exp(np.clip(l, -100, 100)))


class Submap:
    """
    Single submap with log-odds occupancy grid and Bresenham ray-tracing update.
    """

    def __init__(self, width, height, resolution, origin_x, origin_y):
        """
        Initialize submap with uniform prior occupancy grid.

        :param width: (int) Grid width in cells
        :param height: (int) Grid height in cells
        :param resolution: (float) Grid resolution in meters
        :param origin_x: (float) Grid origin x in meters
        :param origin_y: (float) Grid origin y in meters
        """
        assert width > 0 and height > 0, f"map dimensions must be positive: ({width}, {height})"
        assert resolution > 0, f"resolution must be positive, got {resolution}"
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.log_odds = np.full((height, width), prob_to_log_odds(PROB_OCCUPIED_INIT))
        self.n_scans = 0
        self.is_frozen = False

    def update(self, scan_points, pose, free_radius=0.05):
        """
        Update occupancy grid with scan points using Bresenham ray-tracing.

        :param scan_points: (np.ndarray) Scan points in sensor frame, shape (N, 2)
        :param pose: (np.ndarray) Robot pose [x, y, theta], shape (3,)
        :param free_radius: (float) Free space radius around sensor in meters
        """
        assert not self.is_frozen, "cannot update frozen submap"
        self.n_scans += 1
        ox, oy, otheta = pose
        rot = np.array([[np.cos(otheta), -np.sin(otheta)], [np.sin(otheta), np.cos(otheta)]])
        pts_w = (rot @ scan_points.T).T + np.array([ox, oy])
        ipx = np.round((pts_w[:, 0] - self.origin_x) / self.resolution).astype(int)
        ipy = np.round((pts_w[:, 1] - self.origin_y) / self.resolution).astype(int)
        ray_ends = np.column_stack([ipx, ipy])
        ox_pix = int((ox - self.origin_x) / self.resolution)
        oy_pix = int((oy - self.origin_y) / self.resolution)
        for end_x, end_y in ray_ends:
            if not (0 <= end_x < self.width and 0 <= end_y < self.height):
                continue
            self.log_odds[end_y, end_x] += prob_to_log_odds(OCC_PROB_OCCUPIED)
            free_cells = _bresenham(ox_pix, oy_pix, end_x, end_y)
            for fx, fy in free_cells[:-1]:
                if 0 <= fx < self.width and 0 <= fy < self.height:
                    self.log_odds[fy, fx] += prob_to_log_odds(OCC_PROB_FREE)
        self.log_odds = np.clip(self.log_odds, -100, 100)

    def get_prob_grid(self):
        """
        Convert log-odds grid to probability grid.

        :return: (np.ndarray) Probability occupancy grid, shape (H, W)
        """
        return log_odds_to_prob(self.log_odds)


def _bresenham(x0, y0, x1, y1):
    """
    Bresenham line algorithm returning all grid cells along the ray.

    :param x0: (int) Start cell x
    :param y0: (int) Start cell y
    :param x1: (int) End cell x
    :param y1: (int) End cell y
    :return: (list) List of (x, y) cell coordinates along the ray
    """
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


class SubmapCollection:
    """
    Manages multiple submaps with automatic submap switching when scan count exceeds threshold.
    """

    def __init__(self, width=200, height=200, resolution=VOXEL_SIZE, n_submap_scans=N_SUBMAP_SCANS):
        """
        Initialize submap collection with first active submap.

        :param width: (int) Grid width in cells
        :param height: (int) Grid height in cells
        :param resolution: (float) Grid resolution in meters
        :param n_submap_scans: (int) Maximum scans per submap before creating a new one
        """
        self.resolution = resolution
        self.n_submap_scans = n_submap_scans
        self.submaps = []
        self.active_submap = None
        self.map_origin_x = -width * resolution / 2
        self.map_origin_y = -height * resolution / 2
        self.width = width
        self.height = height
        self._create_new_submap()

    def _create_new_submap(self):
        """
        Freeze current submap and create a new active submap.
        """
        sm = Submap(self.width, self.height, self.resolution, self.map_origin_x, self.map_origin_y)
        if self.active_submap is not None:
            self.active_submap.is_frozen = True
        self.submaps.append(sm)
        self.active_submap = sm

    def insert_scan(self, scan_points, pose):
        """
        Insert scan into active submap, switching to new submap if scan limit reached.

        :param scan_points: (np.ndarray) Scan points in sensor frame, shape (N, 2)
        :param pose: (np.ndarray) Robot pose [x, y, theta], shape (3,)
        """
        if self.active_submap.n_scans >= self.n_submap_scans:
            self._create_new_submap()
        self.active_submap.update(scan_points, pose)

    def get_all_prob_grids(self):
        """
        Get probability grids from all submaps.

        :return: (list) List of probability grids, each shape (H, W)
        """
        return [sm.get_prob_grid() for sm in self.submaps]

    def get_combined_grid(self):
        """
        Merge all submap grids by taking element-wise maximum probability.

        :return: (np.ndarray) Combined probability grid, shape (H, W)
        """
        if not self.submaps:
            return np.zeros((self.height, self.width))
        combined = np.zeros_like(self.submaps[0].get_prob_grid())
        for sm in self.submaps:
            combined = np.maximum(combined, sm.get_prob_grid())
        return combined