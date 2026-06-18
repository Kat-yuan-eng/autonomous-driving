"""
Grid map utilities for loading, generating, and manipulating occupancy grid maps

author: Kat-yuan-eng (RuiWen Liao)
"""

import csv
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

sys.path.append(str(pathlib.Path(__file__).parent.parent))


def load_grid_map(csv_path):
    """Load a grid map from a CSV file. Each cell must be 0 (free) or 1 (obstacle).

    :param csv_path: (str or pathlib.Path) Path to the CSV file containing the grid map.
    :return: (numpy.ndarray) 2-D integer array of shape (n_row, n_col) with values in {0, 1}.
    """
    csv_path = pathlib.Path(csv_path)
    assert csv_path.is_file(), f"CSV file not found: {csv_path}"
    grid = np.loadtxt(str(csv_path), delimiter=",", dtype=int)
    assert grid.ndim == 2, f"Grid must be 2-D, got ndim={grid.ndim}"
    assert np.all((grid == 0) | (grid == 1)), "Grid values must be 0 or 1"
    return grid


def generate_random_map(n_row, n_col, obs_ratio, seed=42):
    """Generate a random grid map with a given obstacle ratio. Start (0,0) and goal (n_row-1, n_col-1) are always free.

    :param n_row: (int) Number of rows (must be > 0).
    :param n_col: (int) Number of columns (must be > 0).
    :param obs_ratio: (float) Obstacle ratio in [0, 1].
    :param seed: (int) Random seed for reproducibility. Defaults to 42.
    :return: (numpy.ndarray) 2-D integer array of shape (n_row, n_col).
    """
    assert n_row > 0 and n_col > 0, f"Map size must be positive: ({n_row}, {n_col})"
    assert 0.0 <= obs_ratio <= 1.0, f"obs_ratio must be in [0, 1]: {obs_ratio}"
    rng = np.random.RandomState(seed)
    grid = (rng.rand(n_row, n_col) < obs_ratio).astype(int)
    grid[0, 0] = 0
    grid[n_row - 1, n_col - 1] = 0
    return grid


def generate_maze_map(n_row, n_col, seed=42):
    """Generate a maze grid map using recursive backtracking. Start is (1, 1) and goal is (n_row-2, n_col-2).

    :param n_row: (int) Number of rows (must be >= 5).
    :param n_col: (int) Number of columns (must be >= 5).
    :param seed: (int) Random seed for reproducibility. Defaults to 42.
    :return: (numpy.ndarray) 2-D integer array of shape (n_row, n_col).
    """
    assert n_row >= 5 and n_col >= 5, f"Maze size must be >= 5x5: ({n_row}, {n_col})"
    rng = np.random.RandomState(seed)
    grid = np.ones((n_row, n_col), dtype=int)
    grid[1::2, 1::2] = 0

    n_cell_r = (n_row - 1) // 2
    n_cell_c = (n_col - 1) // 2
    visited = np.zeros((n_cell_r, n_cell_c), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        cr, cc = stack[-1]
        neighbors = [
            (cr + dr, cc + dc, dr, dc)
            for dr, dc in dirs
            if 0 <= cr + dr < n_cell_r
            and 0 <= cc + dc < n_cell_c
            and not visited[cr + dr, cc + dc]
        ]
        if not neighbors:
            stack.pop()
            continue
        nr, nc, dr, dc = neighbors[rng.randint(len(neighbors))]
        grid[1 + cr * 2 + dr, 1 + cc * 2 + dc] = 0
        visited[nr, nc] = True
        stack.append((nr, nc))

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[1, 1] = 0
    grid[n_row - 2, n_col - 2] = 0
    return grid


def inflate_obstacles(grid, radius, protect_positions=None):
    """Inflate obstacles in a grid map by a given radius using binary dilation.

    :param grid: (numpy.ndarray) 2-D integer grid map with values in {0, 1}.
    :param radius: (int) Inflation radius in cells (must be >= 0).
    :param protect_positions: (list of tuple or None) List of (x, y) positions to protect from inflation. Defaults to None.
    :return: (numpy.ndarray) Inflated 2-D integer grid map.
    """
    grid = np.asarray(grid, dtype=int)
    if radius == 0:
        return grid.copy()
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    struct = (x ** 2 + y ** 2) <= radius ** 2
    result = binary_dilation(grid.astype(bool), structure=struct).astype(int)
    if protect_positions is not None:
        for (px, py) in protect_positions:
            y_lo = max(0, py - radius)
            y_hi = min(result.shape[0], py + radius + 1)
            x_lo = max(0, px - radius)
            x_hi = min(result.shape[1], px + radius + 1)
            result[y_lo:y_hi, x_lo:x_hi] = 0
    return result


def build_8connected_neighbors(pos, n_row, n_col, grid=None):
    """Return 8-connected neighbors of pos within an n_row x n_col grid. Diagonal corner-cutting is prevented when grid is provided.

    :param pos: (tuple of (int, int)) (row, col) position.
    :param n_row: (int) Number of rows in the grid.
    :param n_col: (int) Number of columns in the grid.
    :param grid: (numpy.ndarray or None) 2-D grid map. When provided, diagonal neighbors are excluded if both orthogonal neighbors are obstacles. Defaults to None.
    :return: (list of tuple) List of valid (row, col) neighbor positions.
    """
    r, c = int(pos[0]), int(pos[1])
    assert 0 <= r < n_row and 0 <= c < n_col, (
        f"Position out of bounds: pos=({r}, {c}), size=({n_row}, {n_col})"
    )
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1],  [1, 0],  [1, 1],
    ])
    candidates = np.array([r, c]) + offsets
    mask = (
        (candidates[:, 0] >= 0)
        & (candidates[:, 0] < n_row)
        & (candidates[:, 1] >= 0)
        & (candidates[:, 1] < n_col)
    )
    if grid is not None:
        free_mask = grid[candidates[mask, 0], candidates[mask, 1]] == 0
        valid_diag = np.ones(mask.sum(), dtype=bool)
        diag_idx = np.where((np.abs(offsets[mask, 0]) == 1) & (np.abs(offsets[mask, 1]) == 1))[0]
        for di in diag_idx:
            dr, dc = offsets[mask][di]
            r_h, c_h = r, c + dc
            r_v, c_v = r + dr, c
            if 0 <= r_h < n_row and 0 <= c_h < n_col and 0 <= r_v < n_row and 0 <= c_v < n_col:
                if grid[r_h, c_h] == 1 and grid[r_v, c_v] == 1:
                    valid_diag[di] = False
        free_mask &= valid_diag
        return [(int(nr), int(nc)) for nr, nc in candidates[mask][free_mask]]
    return [(int(nr), int(nc)) for nr, nc in candidates[mask]]


def save_path_csv(path, filepath):
    """Save a path (list of (row, col) tuples) to a CSV file.

    :param path: (list of tuple) Path as a sequence of (row, col) positions.
    :param filepath: (str or pathlib.Path) Output CSV file path.
    """
    assert len(path) > 0, "Path is empty, cannot save"
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(str(filepath), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col"])
        for r, c in path:
            writer.writerow([r, c])


def save_metrics_json(metrics, filepath):
    """Save a metrics dictionary to a JSON file.

    :param metrics: (dict) Metrics to serialize.
    :param filepath: (str or pathlib.Path) Output JSON file path.
    """
    assert isinstance(metrics, dict), f"metrics must be dict, got {type(metrics)}"
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(str(filepath), "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(filepath):
    """Load a metrics dictionary from a JSON file.

    :param filepath: (str or pathlib.Path) Input JSON file path.
    :return: (dict) Loaded metrics.
    """
    filepath = pathlib.Path(filepath)
    assert filepath.is_file(), f"JSON file not found: {filepath}"
    with open(str(filepath), "r") as f:
        return json.load(f)


def _show_grid(grid, title, ax):
    """Display a grid map on the given axes.

    :param grid: (numpy.ndarray) 2-D integer grid map.
    :param title: (str) Title of the plot.
    :param ax: (matplotlib.axes.Axes) Axes to draw the grid on.
    """
    ax.imshow(grid, cmap="gray_r", origin="upper")
    n_row, n_col = grid.shape
    ax.scatter(0, 0, c="green", s=80, marker="s", label="start", zorder=5)
    ax.scatter(n_col - 1, n_row - 1, c="blue", s=80, marker="s", label="goal", zorder=5)
    ax.legend(frameon=True, fancybox=True)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)


def main():
    """Demo: generate and display a random map and a maze."""
    n_row, n_col = 31, 31

    grid_random = generate_random_map(n_row, n_col, obs_ratio=0.3, seed=42)
    grid_maze = generate_maze_map(n_row, n_col, seed=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    _show_grid(grid_random, f"Random Map ({n_row}x{n_col}, obs=30%)", ax1)
    _show_grid(grid_maze, f"Maze Map ({n_row}x{n_col})", ax2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
