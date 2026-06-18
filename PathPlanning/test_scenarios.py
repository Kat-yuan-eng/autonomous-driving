"""Test scenarios for path planning algorithms

author: Kat-yuan-eng (RuiWen Liao)
"""

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.grid_map import generate_random_map, generate_maze_map, inflate_obstacles, save_metrics_json

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from GlobalPlanning.AStar.a_star import calc_path_length
from GlobalPlanning.Dijkstra.dijkstra import DijkstraPlanner
from GlobalPlanning.AStar.a_star import AStarPlanner
from GlobalPlanning.AdaptiveAStar.adaptive_a_star import AdaptiveAStarPlanner


# === Phase 1: Global Planner Wrappers ===

def dijkstra(grid, start, goal):
    """Run Dijkstra planning via DijkstraPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = DijkstraPlanner(grid)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def astar(grid, start, goal, heuristic='euclidean'):
    """Run A* planning via AStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :param heuristic: (str) 'euclidean', 'manhattan', or 'chebyshev'
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = AStarPlanner(grid, heuristic=heuristic)
    t0 = time.perf_counter()
    rx, ry, expanded = planner.planning(sx, sy, gx, gy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rx, ry, expanded, elapsed_ms


def adaptive_astar(grid, start, goal):
    """Run Adaptive A* planning via AdaptiveAStarPlanner.

    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (tuple) (path_x, path_y, expanded, elapsed_ms)
    """
    sy, sx = start
    gy, gx = goal
    planner = AdaptiveAStarPlanner(grid)
    rx, ry, expanded, elapsed_ms = planner.planning(sx, sy, gx, gy, heuristic_type="adaptive")
    return rx, ry, expanded, elapsed_ms


# === Phase 2: Utility Functions ===

def check_path_no_collision(path_x, path_y, grid):
    """Verify that no path point falls on an obstacle cell.

    :param path_x: (list) X coordinates of the path
    :param path_y: (list) Y coordinates of the path
    :param grid: (numpy.ndarray) 2-D grid, 0=free 1=obstacle
    :return: (bool) True if collision-free
    """
    for px, py in zip(path_x, path_y):
        if grid[py, px] == 1:
            return False
    return True


def check_path_endpoints(path_x, path_y, start, goal):
    """Verify that path starts at start and ends at goal.

    :param path_x: (list) X coordinates of the path
    :param path_y: (list) Y coordinates of the path
    :param start: (tuple) (sy, sx) start position
    :param goal: (tuple) (gy, gx) goal position
    :return: (bool) True if endpoints match
    """
    sy, sx = start
    gy, gx = goal
    return path_x[0] == sx and path_y[0] == sy and path_x[-1] == gx and path_y[-1] == gy


# === Phase 3: Test Scenarios ===

def test_t1_simple_no_obstacle():
    """Test global planners on an empty 50x50 grid."""
    grid = generate_random_map(50, 50, 0.0, seed=1)
    start, goal = (0, 0), (49, 49)

    px_d, py_d, exp_d, t_d = dijkstra(grid, start, goal)
    px_a, py_a, exp_a, t_a = astar(grid, start, goal)
    px_ad, py_ad, exp_ad, t_ad = adaptive_astar(grid, start, goal)

    assert len(px_d) > 0, "[T1] Dijkstra: no path found"
    assert len(px_a) > 0, "[T1] A*: no path found"
    assert len(px_ad) > 0, "[T1] AdaptiveA*: no path found"
    assert check_path_endpoints(px_d, py_d, start, goal), "[T1] Dijkstra: wrong endpoints"
    assert check_path_endpoints(px_a, py_a, start, goal), "[T1] A*: wrong endpoints"
    assert check_path_endpoints(px_ad, py_ad, start, goal), "[T1] AdaptiveA*: wrong endpoints"

    print("[T1] simple_no_obstacle PASS")
    return {
        "test": "t1_simple_no_obstacle", "status": "PASS",
        "dijkstra": {"expanded": exp_d, "time_ms": round(t_d, 2), "path_len": round(calc_path_length(px_d, py_d), 2)},
        "astar": {"expanded": exp_a, "time_ms": round(t_a, 2), "path_len": round(calc_path_length(px_a, py_a), 2)},
        "adaptive_astar": {"expanded": exp_ad, "time_ms": round(t_ad, 2), "path_len": round(calc_path_length(px_ad, py_ad), 2)},
    }


def test_t2_sparse_obstacles():
    """Test global planners on a 100x100 grid with 15% obstacles."""
    grid = generate_random_map(100, 100, 0.15, seed=42)
    start, goal = (0, 0), (99, 99)

    px_d, py_d, exp_d, t_d = dijkstra(grid, start, goal)
    px_a, py_a, exp_a, t_a = astar(grid, start, goal)
    px_ad, py_ad, exp_ad, t_ad = adaptive_astar(grid, start, goal)

    assert len(px_d) > 0, "[T2] Dijkstra: no path found"
    assert len(px_a) > 0, "[T2] A*: no path found"
    assert len(px_ad) > 0, "[T2] AdaptiveA*: no path found"
    assert check_path_no_collision(px_d, py_d, grid), "[T2] Dijkstra: collision"
    assert check_path_no_collision(px_a, py_a, grid), "[T2] A*: collision"
    assert check_path_no_collision(px_ad, py_ad, grid), "[T2] AdaptiveA*: collision"
    assert t_d < 500, f"[T2] Dijkstra: too slow ({t_d:.1f}ms)"
    assert t_a < 500, f"[T2] A*: too slow ({t_a:.1f}ms)"
    assert t_ad < 500, f"[T2] AdaptiveA*: too slow ({t_ad:.1f}ms)"

    print("[T2] sparse_obstacles PASS")
    return {
        "test": "t2_sparse_obstacles", "status": "PASS",
        "dijkstra": {"expanded": exp_d, "time_ms": round(t_d, 2), "path_len": round(calc_path_length(px_d, py_d), 2)},
        "astar": {"expanded": exp_a, "time_ms": round(t_a, 2), "path_len": round(calc_path_length(px_a, py_a), 2)},
        "adaptive_astar": {"expanded": exp_ad, "time_ms": round(t_ad, 2), "path_len": round(calc_path_length(px_ad, py_ad), 2)},
    }


def test_t3_dense_obstacles():
    """Test global planners on a 200x200 grid with 30% obstacles and verify path optimality."""
    grid = generate_random_map(200, 200, 0.30, seed=7)
    start, goal = (0, 0), (199, 199)

    px_d, py_d, exp_d, t_d = dijkstra(grid, start, goal)
    px_a, py_a, exp_a, t_a = astar(grid, start, goal)
    px_ad, py_ad, exp_ad, t_ad = adaptive_astar(grid, start, goal)

    assert len(px_d) > 0, "[T3] Dijkstra: no path found"
    assert len(px_a) > 0, "[T3] A*: no path found"
    assert len(px_ad) > 0, "[T3] AdaptiveA*: no path found"

    len_d = calc_path_length(px_d, py_d)
    len_a = calc_path_length(px_a, py_a)
    len_ad = calc_path_length(px_ad, py_ad)
    tol = 1e-6
    assert abs(len_d - len_a) < tol, f"[T3] Dijkstra vs A* path length diff: {abs(len_d - len_a):.2e}"
    assert abs(len_a - len_ad) < tol, f"[T3] A* vs AdaptiveA* path length diff: {abs(len_a - len_ad):.2e}"

    print("[T3] dense_obstacles PASS")
    return {
        "test": "t3_dense_obstacles", "status": "PASS",
        "dijkstra": {"expanded": exp_d, "time_ms": round(t_d, 2), "path_len": round(len_d, 2)},
        "astar": {"expanded": exp_a, "time_ms": round(t_a, 2), "path_len": round(len_a, 2)},
        "adaptive_astar": {"expanded": exp_ad, "time_ms": round(t_ad, 2), "path_len": round(len_ad, 2)},
    }


def test_t4_maze():
    """Test global planners on a 201x201 maze map and verify path optimality."""
    grid = generate_maze_map(201, 201, seed=42)
    start, goal = (1, 1), (199, 199)

    px_d, py_d, exp_d, t_d = dijkstra(grid, start, goal)
    px_a, py_a, exp_a, t_a = astar(grid, start, goal)
    px_ad, py_ad, exp_ad, t_ad = adaptive_astar(grid, start, goal)

    assert len(px_d) > 0, "[T4] Dijkstra: no path found"
    assert len(px_a) > 0, "[T4] A*: no path found"
    assert len(px_ad) > 0, "[T4] AdaptiveA*: no path found"

    len_d = calc_path_length(px_d, py_d)
    len_a = calc_path_length(px_a, py_a)
    len_ad = calc_path_length(px_ad, py_ad)
    tol = 1e-6
    assert abs(len_d - len_a) < tol, f"[T4] Dijkstra vs A* path length diff: {abs(len_d - len_a):.2e}"
    assert abs(len_a - len_ad) < tol, f"[T4] A* vs AdaptiveA* path length diff: {abs(len_a - len_ad):.2e}"

    print("[T4] maze PASS")
    return {
        "test": "t4_maze", "status": "PASS",
        "dijkstra": {"expanded": exp_d, "time_ms": round(t_d, 2), "path_len": round(len_d, 2)},
        "astar": {"expanded": exp_a, "time_ms": round(t_a, 2), "path_len": round(len_a, 2)},
        "adaptive_astar": {"expanded": exp_ad, "time_ms": round(t_ad, 2), "path_len": round(len_ad, 2)},
    }


def test_t5_dynamic_obstacles():
    """Test local replanning with dynamic obstacles added to a 100x100 grid."""
    grid = generate_random_map(100, 100, 0.20, seed=99)
    start, goal = (0, 0), (99, 99)

    px_a, py_a, exp_a, t_a = astar(grid, start, goal)
    assert len(px_a) > 0, "[T5] A*: no global path found"

    n_dynamic = 30
    rng = np.random.RandomState(seed=100)
    dyn_obs = []
    for _ in range(n_dynamic):
        ox = rng.randint(5, 95)
        oy = rng.randint(5, 95)
        dyn_obs.append((oy, ox))

    grid_dyn = grid.copy()
    for oy, ox in dyn_obs:
        grid_dyn[oy, ox] = 1

    inflated = inflate_obstacles(grid_dyn, radius=1)

    n_wp = max(1, len(px_a) // 10)
    wp_indices = list(range(0, len(px_a), n_wp))
    if wp_indices[-1] != len(px_a) - 1:
        wp_indices.append(len(px_a) - 1)

    local_path_x, local_path_y = [px_a[0]], [py_a[0]]
    collision_free = True

    for wi in range(len(wp_indices) - 1):
        seg_start = (py_a[wp_indices[wi]], px_a[wp_indices[wi]])
        seg_goal = (py_a[wp_indices[wi + 1]], px_a[wp_indices[wi + 1]])
        seg_px, seg_py, _, _ = astar(inflated, seg_start, seg_goal)
        if len(seg_px) == 0:
            seg_px, seg_py, _, _ = astar(grid_dyn, seg_start, seg_goal)
        if len(seg_px) > 0:
            local_path_x.extend(seg_px[1:])
            local_path_y.extend(seg_py[1:])
        else:
            local_path_x.extend(px_a[wp_indices[wi] + 1:wp_indices[wi + 1] + 1])
            local_path_y.extend(py_a[wp_indices[wi] + 1:wp_indices[wi + 1] + 1])

    for px, py in zip(local_path_x, local_path_y):
        if grid_dyn[py, px] == 1:
            collision_free = False
            break

    assert collision_free, "[T5] local trajectory collides with obstacles"

    print("[T5] dynamic_obstacles PASS")
    return {
        "test": "t5_dynamic_obstacles", "status": "PASS",
        "global_astar": {"expanded": exp_a, "time_ms": round(t_a, 2), "path_len": round(calc_path_length(px_a, py_a), 2)},
        "local_path_len": round(calc_path_length(local_path_x, local_path_y), 2),
        "dynamic_obstacles": n_dynamic,
        "collision_free": collision_free,
    }


def test_t6_large_scale():
    """Test Adaptive A* on a large 500x500 grid with 40% obstacles."""
    grid = generate_random_map(500, 500, 0.40, seed=42)
    start, goal = (0, 0), (499, 499)

    px_ad, py_ad, exp_ad, t_ad = adaptive_astar(grid, start, goal)

    assert len(px_ad) > 0, "[T6] AdaptiveA*: no path found"
    assert t_ad < 2000, f"[T6] AdaptiveA*: too slow ({t_ad:.1f}ms)"

    print("[T6] large_scale PASS")
    return {
        "test": "t6_large_scale", "status": "PASS",
        "adaptive_astar": {"expanded": exp_ad, "time_ms": round(t_ad, 2), "path_len": round(calc_path_length(px_ad, py_ad), 2)},
    }


# === Phase 4: Runner & Report ===

def run_all_tests():
    """Execute all test scenarios and collect results."""
    results = []
    tests = [
        test_t1_simple_no_obstacle,
        test_t2_sparse_obstacles,
        test_t3_dense_obstacles,
        test_t4_maze,
        test_t5_dynamic_obstacles,
        test_t6_large_scale,
    ]

    for test_fn in tests:
        name = test_fn.__name__
        try:
            result = test_fn()
            results.append(result)
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            results.append({"test": name, "status": "FAIL", "error": str(e)})
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append({"test": name, "status": "ERROR", "error": str(e)})

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_err = sum(1 for r in results if r["status"] == "ERROR")

    print("\n=== Test Summary ===")
    print(f"{'Test':<30} {'Status':<8}")
    print("-" * 38)
    for r in results:
        print(f"{r['test']:<30} {r['status']:<8}")
    print("-" * 38)
    print(f"PASS={n_pass}  FAIL={n_fail}  ERROR={n_err}  TOTAL={len(results)}")

    return results


def main():
    """Run all test scenarios and save report to JSON."""
    results = run_all_tests()
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_report.json"
    save_metrics_json({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, out_path)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
