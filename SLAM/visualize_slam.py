"""
SLAM dynamic visualization with matplotlib animation

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Global configuration and utility functions ===
# === Phase 2: Trajectory animation ===
# === Phase 3: Map building animation ===
# === Phase 4: Particle filter animation ===
# === Phase 5: Pose graph animation ===
# === Phase 6: Comprehensive comparison and robustness ===

show_animation = True

import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from SLAM.config import (VIS_FIGSIZE, VIS_DPI, VIS_TRAIL_ALPHA,
    VIS_PARTICLE_ALPHA, VIS_COV_SCALE, VIS_ANIM_INTERVAL, VIS_COLORS)


# === Phase 1: Global configuration and utility functions ===

def setup_rcparams():
    """
    Set matplotlib rcParams for consistent styling across all modules.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_robot(pose, ax, size=0.3, color='blue'):
    """
    Draw robot as triangle.

    :param pose: (ndarray) Robot pose [x, y, theta]
    :param ax: (matplotlib.axes.Axes) Plot axes
    :param size: (float) Robot marker size
    :param color: (str) Robot color
    :return: (matplotlib.patches.Polygon) Robot triangle patch
    """
    x, y, theta = pose[0], pose[1], pose[2]
    tri_local = np.array([
        [size, 0.0],
        [-size * 0.5, size * 0.5],
        [-size * 0.5, -size * 0.5],
    ])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    tri_world = tri_local @ rot.T + np.array([x, y])
    from matplotlib.patches import Polygon
    tri_patch = Polygon(tri_world, closed=True, facecolor=color,
                        edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_patch(tri_patch)
    return tri_patch


def plot_cov_ellipse(pose, cov, ax, n_std=3.0, color='blue', alpha=0.3):
    """
    Draw covariance ellipse.

    :param pose: (ndarray) Center pose [x, y]
    :param cov: (ndarray) 2x2 covariance matrix
    :param ax: (matplotlib.axes.Axes) Plot axes
    :param n_std: (float) Number of standard deviations
    :param color: (str) Ellipse color
    :param alpha: (float) Transparency
    :return: (Ellipse) Covariance ellipse patch
    """
    cov_2d = cov[:2, :2] if cov.shape[0] > 2 else cov
    eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
    eigenvals = np.maximum(eigenvals, 1e-12)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2.0 * n_std * np.sqrt(eigenvals[1])
    height = 2.0 * n_std * np.sqrt(eigenvals[0])
    ell = Ellipse(xy=pose[:2], width=width, height=height, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.0)
    ax.add_patch(ell)
    return ell


def _ensure_figs_dir(save_path):
    if save_path is not None:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)


def _auto_limits(ax, all_x, all_y, margin_frac=0.1, min_span=1.0):
    x_span = all_x.max() - all_x.min()
    y_span = all_y.max() - all_y.min()
    mx = max(x_span * margin_frac, min_span * 0.5)
    my = max(y_span * margin_frac, min_span * 0.5)
    ax.set_xlim(all_x.min() - mx, all_x.max() + mx)
    ax.set_ylim(all_y.min() - my, all_y.max() + my)


# === Phase 2: Trajectory animation ===

def animate_trajectory_comparison(gt_traj, algo_trajs=None, algo_labels=None,
                                   save_path=None):
    """
    Animated trajectory comparison.

    :param gt_traj: (ndarray) Ground truth trajectory, shape (N, 3)
    :param algo_trajs: (dict) Dict of {name: trajectory_array} for each algorithm
    :param algo_labels: (dict) Dict of {name: display_label}
    :param save_path: (str or None) Path to save animation
    :return: (animation.FuncAnimation) Animation object
    """
    if not show_animation:
        return None

    setup_rcparams()
    fig, ax = plt.subplots(figsize=VIS_FIGSIZE)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('SLAM Trajectory Comparison')

    algo_trajs = algo_trajs or {}
    algo_labels = algo_labels or {}

    gt_line, = ax.plot([], [], '-k', linewidth=2, label='Ground Truth')
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], 'o', color='#2ca02c', markersize=8, zorder=5)
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'x', color='#d62728', markersize=10, markeredgewidth=2, zorder=5)
    algo_lines = {}
    for name in algo_trajs:
        label = algo_labels.get(name, name)
        color = VIS_COLORS.get(name, None)
        ln, = ax.plot([], [], linewidth=1.5, label=label, alpha=VIS_TRAIL_ALPHA,
                       color=color)
        algo_lines[name] = ln

    robot_dot, = ax.plot([], [], 'o', color='black', markersize=5, zorder=6)
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    n_frames = len(gt_traj)

    all_x = gt_traj[:, 0].copy()
    all_y = gt_traj[:, 1].copy()
    for traj in algo_trajs.values():
        all_x = np.concatenate([all_x, traj[:, 0]])
        all_y = np.concatenate([all_y, traj[:, 1]])
    _auto_limits(ax, all_x, all_y)
    ax.legend(loc='upper right', frameon=True, fancybox=True)

    all_artists = [gt_line, robot_dot, step_text, ax.title] + list(algo_lines.values())

    def init():
        gt_line.set_data([], [])
        robot_dot.set_data([], [])
        step_text.set_text('')
        for ln in algo_lines.values():
            ln.set_data([], [])
        return all_artists

    def update(frame):
        step = frame + 1
        gt_line.set_data(gt_traj[:step, 0], gt_traj[:step, 1])
        robot_dot.set_data([gt_traj[frame, 0]], [gt_traj[frame, 1]])
        step_text.set_text(f'Step: {step}/{n_frames}')
        ax.set_title(f'SLAM Trajectory Comparison  [{step}/{n_frames}]')

        for name, traj in algo_trajs.items():
            s = min(step, len(traj))
            algo_lines[name].set_data(traj[:s, 0], traj[:s, 1])

        return all_artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, interval=VIS_ANIM_INTERVAL,
                                   blit=True, repeat=False)
    _ensure_figs_dir(save_path)
    if save_path:
        plt.tight_layout()
        ani.save(str(save_path), writer='pillow', fps=20)
        print(f"[save] trajectory animation -> {save_path}")
    plt.tight_layout()
    plt.show()
    return ani


# === Phase 3: Map building animation ===

def animate_map_building(gt_traj, landmarks, est_landmarks_history, save_path=None):
    """
    Animated incremental map building.

    :param gt_traj: (ndarray) Ground truth trajectory
    :param landmarks: (ndarray) True landmark positions
    :param est_landmarks_history: (list) List of estimated landmarks at each step
    :param save_path: (str or None) Save path
    :return: (animation.FuncAnimation) Animation object
    """
    if not show_animation:
        return None

    setup_rcparams()
    fig, ax = plt.subplots(figsize=VIS_FIGSIZE)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Incremental Map Building')

    gt_line, = ax.plot([], [], '-k', linewidth=1.5, alpha=VIS_TRAIL_ALPHA,
                        label='Ground Truth')
    robot_dot, = ax.plot([], [], 'o', color='black', markersize=6, zorder=6)
    true_lm_scatter = ax.scatter(landmarks[:, 0], landmarks[:, 1],
                                  c='gray', marker='x', s=30, alpha=0.4,
                                  label='True Landmarks', zorder=2)
    est_lm_scatter = ax.scatter([], [], c=VIS_COLORS['ekf'], marker='o',
                                 s=20, alpha=0.7, label='Estimated Landmarks',
                                 zorder=3)
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    n_frames = len(gt_traj)
    all_x = np.concatenate([gt_traj[:, 0], landmarks[:, 0]])
    all_y = np.concatenate([gt_traj[:, 1], landmarks[:, 1]])
    _auto_limits(ax, all_x, all_y)
    ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)

    all_artists = [gt_line, robot_dot, est_lm_scatter, step_text, ax.title]

    def init():
        gt_line.set_data([], [])
        robot_dot.set_data([], [])
        step_text.set_text('')
        return all_artists

    def update(frame):
        step = frame + 1
        gt_line.set_data(gt_traj[:step, 0], gt_traj[:step, 1])
        robot_dot.set_data([gt_traj[frame, 0]], [gt_traj[frame, 1]])
        step_text.set_text(f'Step: {step}/{n_frames}')
        ax.set_title(f'Incremental Map Building  [{step}/{n_frames}]')

        if frame < len(est_landmarks_history) and est_landmarks_history[frame] is not None:
            est_lm = est_landmarks_history[frame]
            if len(est_lm) > 0:
                est_lm_scatter.set_offsets(est_lm[:, :2])
            else:
                est_lm_scatter.set_offsets(np.empty((0, 2)))

        return all_artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, interval=VIS_ANIM_INTERVAL,
                                   blit=True, repeat=False)
    _ensure_figs_dir(save_path)
    if save_path:
        plt.tight_layout()
        ani.save(str(save_path), writer='pillow', fps=20)
        print(f"[save] map building animation -> {save_path}")
    plt.tight_layout()
    plt.show()
    return ani


# === Phase 4: Particle filter animation ===

def animate_particles(particle_history, gt_traj, save_path=None):
    """
    Animated FastSLAM particle visualization.

    :param particle_history: (list) List of particle position arrays at each step
    :param gt_traj: (ndarray) Ground truth trajectory
    :param save_path: (str or None) Save path
    :return: (animation.FuncAnimation) Animation object
    """
    if not show_animation:
        return None

    setup_rcparams()
    fig, ax = plt.subplots(figsize=VIS_FIGSIZE)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('FastSLAM Particle Evolution')

    gt_line, = ax.plot([], [], '-k', linewidth=2, alpha=VIS_TRAIL_ALPHA,
                        label='Ground Truth')
    particle_scatter = ax.scatter([], [], c=VIS_COLORS['fastslam'],
                                   s=8, alpha=VIS_PARTICLE_ALPHA,
                                   label='Particles', zorder=3)
    robot_dot, = ax.plot([], [], 'o', color='red', markersize=6, zorder=6,
                          label='Robot')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    n_frames = min(len(gt_traj), len(particle_history))
    all_x = gt_traj[:n_frames, 0].copy()
    all_y = gt_traj[:n_frames, 1].copy()
    for ph in particle_history[:n_frames]:
        if ph is not None and len(ph) > 0:
            all_x = np.concatenate([all_x, ph[:, 0]])
            all_y = np.concatenate([all_y, ph[:, 1]])
    _auto_limits(ax, all_x, all_y)
    ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)

    all_artists = [gt_line, particle_scatter, robot_dot, step_text, ax.title]

    def init():
        gt_line.set_data([], [])
        robot_dot.set_data([], [])
        step_text.set_text('')
        return all_artists

    def update(frame):
        step = frame + 1
        gt_line.set_data(gt_traj[:step, 0], gt_traj[:step, 1])
        robot_dot.set_data([gt_traj[frame, 0]], [gt_traj[frame, 1]])
        step_text.set_text(f'Step: {step}/{n_frames}  Particles: {len(particle_history[frame])}')
        ax.set_title(f'FastSLAM Particle Evolution  [{step}/{n_frames}]')

        if particle_history[frame] is not None and len(particle_history[frame]) > 0:
            particle_scatter.set_offsets(particle_history[frame][:, :2])
        else:
            particle_scatter.set_offsets(np.empty((0, 2)))

        return all_artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, interval=VIS_ANIM_INTERVAL,
                                   blit=True, repeat=False)
    _ensure_figs_dir(save_path)
    if save_path:
        plt.tight_layout()
        ani.save(str(save_path), writer='pillow', fps=20)
        print(f"[save] particle animation -> {save_path}")
    plt.tight_layout()
    plt.show()
    return ani


# === Phase 5: Pose graph animation ===

def animate_pose_graph(nodes_history, edges, loop_edges=None, save_path=None):
    """
    Animated pose graph optimization.

    :param nodes_history: (list) List of node arrays at each optimization iteration
    :param edges: (list) Odometry edges
    :param loop_edges: (list or None) Loop closure edges
    :param save_path: (str or None) Save path
    :return: (animation.FuncAnimation) Animation object
    """
    if not show_animation:
        return None

    setup_rcparams()
    fig, ax = plt.subplots(figsize=VIS_FIGSIZE)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Pose Graph Optimization')

    loop_edges = loop_edges or []
    n_iters = len(nodes_history)

    all_nodes = np.vstack(nodes_history)
    _auto_limits(ax, all_nodes[:, 0], all_nodes[:, 1])

    def _draw_edges(nodes, edge_list, color, linewidth, alpha):
        segments = []
        for e in edge_list:
            i, j = int(e[0]), int(e[1])
            if i < len(nodes) and j < len(nodes):
                segments.append([nodes[i, :2], nodes[j, :2]])
        if segments:
            lc = LineCollection(segments, colors=color, linewidths=linewidth,
                                alpha=alpha, zorder=1)
            ax.add_collection(lc)
            return lc
        return None

    node_scatter = ax.scatter([], [], c=VIS_COLORS['graphslam'], s=15,
                               zorder=4, label='Nodes')
    iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        node_scatter.set_offsets(np.empty((0, 2)))
        iter_text.set_text('')
        return node_scatter, iter_text

    def update(frame):
        ax.collections.clear()
        nodes = nodes_history[frame]
        node_scatter_new = ax.scatter(nodes[:, 0], nodes[:, 1],
                                       c=VIS_COLORS['graphslam'], s=15, zorder=4)
        _draw_edges(nodes, edges, VIS_COLORS['odom_edge'], 0.5, 0.4)
        if loop_edges:
            _draw_edges(nodes, loop_edges, VIS_COLORS['loop_edge'], 1.5, 0.8)
        iter_text.set_text(f'Iteration: {frame + 1}/{n_iters}')

        ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)
        return [node_scatter_new, iter_text]

    ani = animation.FuncAnimation(fig, update, frames=n_iters,
                                   init_func=init, interval=200,
                                   blit=False, repeat=False)
    _ensure_figs_dir(save_path)
    if save_path:
        plt.tight_layout()
        ani.save(str(save_path), writer='pillow', fps=5)
        print(f"[save] pose graph animation -> {save_path}")
    plt.tight_layout()
    plt.show()
    return ani


# === Phase 6: Comprehensive comparison and robustness ===

def visualize_comprehensive_comparison(results_dict, save_dir=None):
    """
    Multi-panel comprehensive comparison.

    :param results_dict: (dict) Results from compare_slam.run_all_algorithms()
    :param save_dir: (str or None) Directory to save figures
    :return: (None)
    """
    setup_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=VIS_FIGSIZE,
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.3})

    algo_names = [k for k in results_dict if k != 'ground_truth']
    gt_traj = np.array(results_dict.get('ground_truth', {}).get('trajectory', []))

    # left -> angle trajectory overlay
    ax_traj = axes[0, 0]
    if len(gt_traj) > 0:
        ax_traj.plot(gt_traj[:, 0], gt_traj[:, 1], '-k', linewidth=2,
                      label='Ground Truth', alpha=0.9)
        ax_traj.plot(gt_traj[0, 0], gt_traj[0, 1], 'o', color='#2ca02c', markersize=8, zorder=5)
        ax_traj.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'x', color='#d62728', markersize=10, markeredgewidth=2, zorder=5)
    for name in algo_names:
        traj = np.array(results_dict[name].get('trajectory', []))
        if len(traj) > 0:
            n = min(len(traj), len(gt_traj)) if len(gt_traj) > 0 else len(traj)
            color = VIS_COLORS.get(name, None)
            ax_traj.plot(traj[:n, 0], traj[:n, 1], linewidth=1.5,
                          label=name, alpha=VIS_TRAIL_ALPHA, color=color)
    ax_traj.set_xlabel('x [m]')
    ax_traj.set_ylabel('y [m]')
    ax_traj.set_title('Trajectory Overlay')
    ax_traj.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)
    ax_traj.axis('equal')

    # right -> angle position error time series
    ax_err = axes[0, 1]
    for name in algo_names:
        traj = np.array(results_dict[name].get('trajectory', []))
        if len(traj) > 0 and len(gt_traj) > 0:
            n = min(len(traj), len(gt_traj))
            pos_err = np.sqrt((traj[:n, 0] - gt_traj[:n, 0])**2 +
                               (traj[:n, 1] - gt_traj[:n, 1])**2)
            color = VIS_COLORS.get(name, None)
            ax_err.plot(np.arange(n), pos_err, linewidth=1, label=name,
                         alpha=VIS_TRAIL_ALPHA, color=color)
    for name in algo_names:
        traj = np.array(results_dict[name].get('trajectory', []))
        if len(traj) > 0 and len(gt_traj) > 0:
            n = min(len(traj), len(gt_traj))
            pos_err = np.sqrt((traj[:n, 0] - gt_traj[:n, 0])**2 + (traj[:n, 1] - gt_traj[:n, 1])**2)
            mean_err = np.mean(pos_err)
            color = VIS_COLORS.get(name, None)
            ax_err.axhline(y=mean_err, color=color, linestyle=':', linewidth=0.8, alpha=0.5)
    ax_err.set_xlabel('Step')
    ax_err.set_ylabel('Position Error [m]')
    ax_err.set_title('Position Error')
    ax_err.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)

    # left -> angle RMSE bar chart
    ax_rmse = axes[1, 0]
    rmse_vals = []
    rmse_names = []
    for name in algo_names:
        metrics = results_dict[name].get('metrics', {})
        if 'pos_rmse' in metrics:
            rmse_vals.append(metrics['pos_rmse'])
            rmse_names.append(name)
    if rmse_vals:
        bar_colors = [VIS_COLORS.get(n, '#333333') for n in rmse_names]
        bars = ax_rmse.bar(rmse_names, rmse_vals, color=bar_colors, alpha=0.8)
        for bar, val in zip(bars, rmse_vals):
            ax_rmse.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax_rmse.set_ylabel('RMSE [m]')
    ax_rmse.set_title('RMSE Comparison')
    ax_rmse.tick_params(axis='x', rotation=15)

    # right -> angle computation time bar chart
    ax_time = axes[1, 1]
    time_vals = []
    time_names = []
    for name in algo_names:
        metrics = results_dict[name].get('metrics', {})
        if 'step_time_ms' in metrics:
            time_vals.append(metrics['step_time_ms'])
            time_names.append(name)
    if time_vals:
        bar_colors = [VIS_COLORS.get(n, '#333333') for n in time_names]
        bars = ax_time.bar(time_names, time_vals, color=bar_colors, alpha=0.8)
        for bar, val in zip(bars, time_vals):
            ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax_time.set_ylabel('Avg Step Time [ms]')
    ax_time.set_title('Computation Time')
    ax_time.tick_params(axis='x', rotation=15)

    if save_dir is None:
        save_dir = pathlib.Path(__file__).parent / 'figs'
    else:
        save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'slam_comprehensive.png'
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=VIS_DPI, bbox_inches='tight')
    plt.savefig(str(pathlib.Path(__file__).parent / 'results' / 'slam_comprehensive.png'), dpi=VIS_DPI, bbox_inches='tight')
    print(f"[save] comprehensive comparison -> {save_path}")
    plt.show()


def visualize_robustness_analysis(noise_levels, rmse_per_algo, save_dir=None):
    """
    Noise robustness plot.

    :param noise_levels: (list) Noise multiplier values
    :param rmse_per_algo: (dict) Dict of {algo_name: rmse_array}
    :param save_dir: (str or None) Save directory
    :return: (None)
    """
    setup_rcparams()
    fig, ax = plt.subplots(figsize=VIS_FIGSIZE)

    for name, rmse_arr in rmse_per_algo.items():
        color = VIS_COLORS.get(name, None)
        ax.plot(noise_levels, rmse_arr, 'o-', linewidth=1.5, markersize=5,
                 label=name, color=color)

    ax.set_xlabel('Noise Scale Factor')
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('Robustness Analysis: RMSE vs Noise Level')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    if save_dir is None:
        save_dir = pathlib.Path(__file__).parent / 'figs'
    else:
        save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'slam_robustness.png'
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=VIS_DPI, bbox_inches='tight')
    plt.savefig(str(pathlib.Path(__file__).parent / 'results' / 'slam_robustness.png'), dpi=VIS_DPI, bbox_inches='tight')
    print(f"[save] robustness analysis -> {save_path}")
    plt.show()
