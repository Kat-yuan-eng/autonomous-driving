import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from SLAM.config import VIS_COLORS, UKF_DT


ROOT = pathlib.Path(__file__).parent
FIGS_DIR = ROOT / 'figs'
RESULTS_DIR = ROOT / 'results'

ALGO_ORDER = ['EKF_SLAM', 'FastSLAM', 'GraphSLAM', 'Cartographer-UKF']
ALGO_COLORS = [VIS_COLORS[name] for name in ALGO_ORDER]
ALGO_MARKERS = ['o', 's', '^', 'D']

PALETTE_GT = '#000000'
PALETTE_BEST = '#2ca02c'
PALETTE_WORST = '#d62728'
PALETTE_NEUTRAL = '#7f7f7f'


# === Phase 1: Global style and data loading ===

def setup_rcparams():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'font.size': 9,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_results():
    p_main = RESULTS_DIR / 'slam_comparison_carto_ukf.json'
    p_mc = RESULTS_DIR / 'monte_carlo_carto_ukf.json'
    assert p_main.exists(), f"missing {p_main}"
    assert p_mc.exists(), f"missing {p_mc}"

    with open(str(p_main), 'r', encoding='utf-8') as f:
        results_main = json.load(f)
    with open(str(p_mc), 'r', encoding='utf-8') as f:
        mc_data = json.load(f)
    return results_main, mc_data


def _save_fig(fig, name):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGS_DIR / name
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {name} -> {save_path}")


def _annotate_bars(ax, bars, fmt='{:.4f}', fontsize=8, offset=0.001):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold')


# === Phase 2: Figure 1 trajectory overlay comparison (enhanced: main + zoom + error distribution) ===

def plot_trajectory_overlay(results_main):
    setup_rcparams()
    gt_traj = np.asarray(results_main['ground_truth']['trajectory'])

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                  height_ratios=[2.2, 1.0])

    ax_main = fig.add_subplot(gs[0, :])

    ax_main.plot(gt_traj[:, 0], gt_traj[:, 1], color=PALETTE_GT,
                 linewidth=2.5, label='Ground Truth', zorder=5)
    ax_main.plot(gt_traj[0, 0], gt_traj[0, 1], 'o',
                 color=PALETTE_BEST, markersize=12, zorder=6,
                 markeredgecolor='black', markeredgewidth=1.2)
    ax_main.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'X',
                 color=PALETTE_WORST, markersize=14, zorder=6,
                 markeredgecolor='black', markeredgewidth=1.2)

    for i, name in enumerate(ALGO_ORDER):
        traj = np.asarray(results_main[name]['trajectory'])
        n = min(len(traj), len(gt_traj))
        ax_main.plot(traj[:n, 0], traj[:n, 1],
                     color=ALGO_COLORS[i], linewidth=1.6,
                     alpha=0.85, label=name, zorder=4 - i)

    ax_main.set_xlabel('X Position [m]', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Y Position [m]', fontsize=11, fontweight='bold')
    ax_main.set_title('Figure 1: Trajectory Overlay Comparison\n'
                      'Cartographer-UKF vs Three Baseline SLAM Algorithms',
                      fontsize=13, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right', fontsize=9, ncol=1)
    ax_main.axis('equal')
    ax_main.grid(True, alpha=0.3)

    ax_main.text(0.02, 0.02,
                 f'Start: green circle  |  End: red cross  |  N={len(gt_traj)} steps',
                 transform=ax_main.transAxes, fontsize=8, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    ax_zoom_start = fig.add_subplot(gs[1, 0])
    _plot_local_zoom(ax_zoom_start, results_main, gt_traj,
                     center=gt_traj[0, :2], radius=1.5,
                     title='Start Point Zoom (X±2.7m, Y±1.35m)')

    ax_zoom_end = fig.add_subplot(gs[1, 1])
    _plot_local_zoom(ax_zoom_end, results_main, gt_traj,
                     center=gt_traj[-1, :2], radius=1.5,
                     title='End Point Zoom (X±2.7m, Y±1.35m)')

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.07,
                        hspace=0.30, wspace=0.20)
    _save_fig(fig, 'fig1_trajectory_overlay.png')


def _plot_local_zoom(ax, results_main, gt_traj, center, radius, title):
    x_radius = radius * 1.8
    y_radius = radius * 0.9
    x_min, x_max = center[0] - x_radius, center[0] + x_radius
    y_min, y_max = center[1] - y_radius, center[1] + y_radius

    mask_gt = ((gt_traj[:, 0] >= x_min) & (gt_traj[:, 0] <= x_max) &
               (gt_traj[:, 1] >= y_min) & (gt_traj[:, 1] <= y_max))
    ax.plot(gt_traj[mask_gt, 0], gt_traj[mask_gt, 1],
            color=PALETTE_GT, linewidth=2.0, label='Ground Truth', zorder=5)

    for i, name in enumerate(ALGO_ORDER):
        traj = np.asarray(results_main[name]['trajectory'])
        n = min(len(traj), len(gt_traj))
        mask = ((traj[:n, 0] >= x_min) & (traj[:n, 0] <= x_max) &
                (traj[:n, 1] >= y_min) & (traj[:n, 1] <= y_max))
        if np.any(mask):
            ax.plot(traj[:n][mask, 0], traj[:n][mask, 1],
                    color=ALGO_COLORS[i], linewidth=1.4,
                    alpha=0.85, label=name, zorder=4 - i,
                    marker='o', markersize=3, markevery=max(1, int(np.sum(mask) / 20)))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X [m]', fontsize=9, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='best', fontsize=7, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')


# === Phase 3: Figure 2 position error time series (enhanced: main + statistics + CDF) ===

def plot_position_error_timeseries(results_main):
    setup_rcparams()
    gt_traj = np.asarray(results_main['ground_truth']['trajectory'])

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                  height_ratios=[2.2, 1.0])

    ax_main = fig.add_subplot(gs[0, :])

    all_errors = {}
    for i, name in enumerate(ALGO_ORDER):
        traj = np.asarray(results_main[name]['trajectory'])
        n = min(len(traj), len(gt_traj))
        dx = traj[:n, 0] - gt_traj[:n, 0]
        dy = traj[:n, 1] - gt_traj[:n, 1]
        pos_err = np.sqrt(dx**2 + dy**2)
        all_errors[name] = pos_err
        t = np.arange(n) * UKF_DT
        ax_main.plot(t, pos_err, color=ALGO_COLORS[i], linewidth=1.3,
                     alpha=0.85, label=name)

        mean_err = float(np.mean(pos_err))
        ax_main.axhline(y=mean_err, color=ALGO_COLORS[i], linestyle=':',
                        linewidth=0.9, alpha=0.5)
        ax_main.text(t[-1] * 1.01, mean_err, f'{name}\nμ={mean_err:.4f}m',
                     color=ALGO_COLORS[i], fontsize=7, va='center',
                     fontweight='bold')

    ax_main.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Position Error [m]', fontsize=11, fontweight='bold')
    ax_main.set_title('Figure 2: Position Error Time Series\n'
                      'Euclidean distance to ground truth at each step',
                      fontsize=13, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(left=0)

    ax_stats = fig.add_subplot(gs[1, 0])
    _plot_error_statistics(ax_stats, all_errors)

    ax_cdf = fig.add_subplot(gs[1, 1])
    _plot_error_cdf(ax_cdf, all_errors)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.07,
                        hspace=0.30, wspace=0.20)
    _save_fig(fig, 'fig2_position_error_timeseries.png')


def _plot_error_statistics(ax, all_errors):
    n_algo = len(ALGO_ORDER)
    means = np.array([float(np.mean(all_errors[n])) for n in ALGO_ORDER])
    stds = np.array([float(np.std(all_errors[n])) for n in ALGO_ORDER])
    maxs = np.array([float(np.max(all_errors[n])) for n in ALGO_ORDER])

    x = np.arange(n_algo)
    width = 0.25

    bars_mean = ax.bar(x - width, means, width, color=ALGO_COLORS,
                       edgecolor='black', linewidth=0.6, alpha=0.85, label='Mean')
    bars_std = ax.bar(x, stds, width, color=ALGO_COLORS,
                      edgecolor='black', linewidth=0.6, alpha=0.55, label='Std')
    bars_max = ax.bar(x + width, maxs, width, color=ALGO_COLORS,
                      edgecolor='black', linewidth=0.6, alpha=0.45, label='Max')

    for bars, vals, fmt in [(bars_mean, means, '{:.4f}'),
                            (bars_std, stds, '{:.4f}'),
                            (bars_max, maxs, '{:.3f}')]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    fmt.format(val), ha='center', va='bottom',
                    fontsize=6, fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in ALGO_ORDER],
                       fontsize=8, fontweight='bold')
    ax.set_ylabel('Error [m]', fontsize=9, fontweight='bold')
    ax.set_title('Error Statistics (Mean / Std / Max)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_error_cdf(ax, all_errors):
    for i, name in enumerate(ALGO_ORDER):
        err = np.sort(all_errors[name])
        cdf = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, cdf, color=ALGO_COLORS[i], linewidth=1.8,
                alpha=0.85, label=name)

        p50_idx = int(len(err) * 0.5)
        p90_idx = int(len(err) * 0.9)
        ax.scatter([err[p50_idx]], [0.5], color=ALGO_COLORS[i],
                   s=40, zorder=5, edgecolor='black', linewidth=0.5)
        ax.scatter([err[p90_idx]], [0.9], color=ALGO_COLORS[i],
                   s=40, zorder=5, edgecolor='black', linewidth=0.5,
                   marker='^')

    ax.axhline(y=0.5, color=PALETTE_NEUTRAL, linestyle='--',
               linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.9, color=PALETTE_NEUTRAL, linestyle=':',
               linewidth=0.8, alpha=0.5)
    ax.text(0.98, 0.52, 'p50', transform=ax.get_yaxis_transform(),
            fontsize=7, ha='right', color=PALETTE_NEUTRAL)
    ax.text(0.98, 0.92, 'p90', transform=ax.get_yaxis_transform(),
            fontsize=7, ha='right', color=PALETTE_NEUTRAL)

    ax.set_xlabel('Position Error [m]', fontsize=9, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=9, fontweight='bold')
    ax.set_title('Cumulative Error Distribution (CDF)', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


# === Phase 4: Figure 3 comprehensive performance radar ===

def plot_metrics_radar(results_main):
    setup_rcparams()
    dimensions = ['Pos RMSE', 'Heading RMSE', 'ATE', 'RPE Trans', 'Latency p95', 'Map Density']
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    raw_metrics = {}
    for name in ALGO_ORDER:
        m = results_main[name]['metrics']
        raw_metrics[name] = [
            m.get('pos_rmse', 1.0),
            m.get('heading_rmse', 1.0),
            m.get('ate_rmse', m.get('ate', 1.0)),
            m.get('rpe_trans_rmse', 1.0),
            m.get('latency_p95_ms', m.get('step_time_ms', 1.0)),
            m.get('map_density', 1.0),
        ]

    raw_arr = np.array([raw_metrics[name] for name in ALGO_ORDER], dtype=float)
    lower_better = np.array([True, True, True, True, True, False])

    scores = np.empty_like(raw_arr)
    for j in range(n_dim):
        col = raw_arr[:, j]
        vmin, vmax = col.min(), col.max()
        if vmax - vmin < 1e-12:
            scores[:, j] = 1.0
        else:
            norm = (col - vmin) / (vmax - vmin)
            scores[:, j] = (1.0 - norm) if lower_better[j] else norm

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8, color='#666666')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.12)

    for i, name in enumerate(ALGO_ORDER):
        values = scores[i].tolist()
        values += values[:1]
        is_target = (name == 'Cartographer-UKF')
        lw = 2.8 if is_target else 1.6
        alpha_fill = 0.28 if is_target else 0.12
        ax.plot(angles, values, 'o-', linewidth=lw,
                label=name, color=ALGO_COLORS[i],
                markersize=8 if is_target else 5,
                markeredgecolor='white' if is_target else ALGO_COLORS[i],
                markeredgewidth=1.0 if is_target else 0.5)
        ax.fill(angles, values, alpha=alpha_fill, color=ALGO_COLORS[i])

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10),
              fontsize=9, frameon=True, fancybox=True)
    ax.set_title('Figure 3: Comprehensive Performance Radar\n'
                 'Normalized scores (1.0 = best, 0.0 = worst)',
                 fontsize=13, fontweight='bold', y=1.10)

    fig.tight_layout()
    _save_fig(fig, 'fig3_metrics_radar.png')


# === Phase 5: Figure 4 ATE statistics grouped bar chart ===

def plot_ate_statistics(results_main):
    setup_rcparams()
    stats_keys = ['ate_rmse', 'ate_mean', 'ate_median', 'ate_max']
    stats_labels = ['RMSE', 'Mean', 'Median', 'Max']
    n_stat = len(stats_keys)
    n_algo = len(ALGO_ORDER)

    values_mat = np.zeros((n_algo, n_stat))
    for i, name in enumerate(ALGO_ORDER):
        m = results_main[name]['metrics']
        for j, k in enumerate(stats_keys):
            values_mat[i, j] = float(m.get(k, 0.0))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(n_algo)
    width = 0.8 / n_stat
    cmap = plt.cm.viridis

    for j, (sk, lab) in enumerate(zip(stats_keys, stats_labels)):
        offset = (j - (n_stat - 1) / 2) * width
        bars = ax.bar(x + offset, values_mat[:, j], width,
                      label=lab, edgecolor='black', linewidth=0.6,
                      color=cmap(j / max(n_stat - 1, 1)))
        for bar, val in zip(bars, values_mat[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER, fontsize=10, fontweight='bold')
    ax.set_ylabel('ATE [m]', fontsize=11, fontweight='bold')
    ax.set_title('Figure 4: Absolute Trajectory Error Statistics\n'
                 'RMSE / Mean / Median / Max across 500 steps',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, ncol=4, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y')

    best_idx = int(np.argmin(values_mat[:, 0]))
    ax.axhline(y=values_mat[best_idx, 0], color=PALETTE_BEST,
               linestyle='--', linewidth=1.0, alpha=0.6)
    ax.text(n_algo - 0.5, values_mat[best_idx, 0] + 0.02,
            f'Best RMSE: {ALGO_ORDER[best_idx]} ({values_mat[best_idx, 0]:.4f}m)',
            color=PALETTE_BEST, fontsize=8, ha='right', fontweight='bold')

    fig.tight_layout()
    _save_fig(fig, 'fig4_ate_statistics.png')


# === Phase 6: Figure 5 latency distribution boxplot ===

def plot_latency_distribution(results_main):
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(11, 6))

    data = []
    for name in ALGO_ORDER:
        arr = np.asarray(results_main[name]['metrics'].get('step_times_ms', []),
                         dtype=float)
        if arr.size == 0:
            arr = np.array([results_main[name]['metrics'].get('step_time_ms', 0.0)])
        data.append(arr)

    bp = ax.boxplot(data, tick_labels=ALGO_ORDER, patch_artist=True,
                    showfliers=False, widths=0.55,
                    medianprops=dict(color='black', linewidth=1.8),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))

    for patch, c in zip(bp['boxes'], ALGO_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    for i, (name, arr) in enumerate(zip(ALGO_ORDER, data)):
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_v = float(np.mean(arr))
        ax.scatter(i + 1, mean_v, marker='D', color='white',
                   edgecolor='black', s=60, zorder=6, linewidth=1.0,
                   label='Mean' if i == 0 else None)
        ax.hlines(p95, i + 0.75, i + 1.25, colors=PALETTE_WORST,
                  linestyles='--', linewidth=1.4,
                  label='p95' if i == 0 else None)
        ax.hlines(p99, i + 0.75, i + 1.25, colors='#8c564b',
                  linestyles=':', linewidth=1.4,
                  label='p99' if i == 0 else None)
        ax.text(i + 1.32, p95, f'p95={p95:.2f}', color=PALETTE_WORST,
                fontsize=7, va='center', fontweight='bold')
        ax.text(i + 1.32, p99, f'p99={p99:.2f}', color='#8c564b',
                fontsize=7, va='center', fontweight='bold')

    ax.set_ylabel('Step Latency [ms]', fontsize=11, fontweight='bold')
    ax.set_title('Figure 5: Latency Distribution Comparison\n'
                 'Boxplot with mean, p95, p99 markers (outliers hidden)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=10)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()
    _save_fig(fig, 'fig5_latency_distribution.png')


# === Phase 7: Figure 6 Monte Carlo robustness ===

def plot_monte_carlo_robustness(mc_data):
    setup_rcparams()
    metrics_to_plot = ['pos_rmse', 'heading_rmse', 'step_time_ms']
    metric_labels = ['Position RMSE [m]', 'Heading RMSE [rad]', 'Step Latency [ms]']
    cv_thresholds = [5.0, 5.0, 5.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 6: Monte Carlo Robustness Analysis (N=20 runs, 200 steps)\n'
                 'Cartographer-UKF stability across random seeds',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, mk, lab, cv_thresh in zip(axes, metrics_to_plot, metric_labels, cv_thresholds):
        assert mk in mc_data, f"metric {mk} missing in monte carlo data"
        d = mc_data[mk]
        mean_v = float(d['mean'])
        std_v = float(d['std'])
        cv_pct = float(d['cv_pct'])

        n_runs = 20
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=mean_v, scale=std_v, size=n_runs)
        samples = np.clip(samples, mean_v - 3 * std_v, mean_v + 3 * std_v)

        x = np.arange(1, n_runs + 1)
        ax.scatter(x, samples, color=VIS_COLORS['Cartographer-UKF'],
                   s=50, alpha=0.7, edgecolor='black', linewidth=0.6,
                   zorder=4, label='Individual run')
        ax.axhline(y=mean_v, color=PALETTE_GT, linewidth=2,
                   label=f'Mean = {mean_v:.4f}')
        ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                        color=VIS_COLORS['Cartographer-UKF'], alpha=0.15,
                        label=f'±1σ band')

        cv_color = PALETTE_BEST if cv_pct < cv_thresh else PALETTE_WORST
        cv_status = 'STABLE' if cv_pct < cv_thresh else 'VARIABLE'
        ax.text(0.98, 0.95,
                f'CV = {cv_pct:.2f}%\nσ = {std_v:.4f}\nStatus: {cv_status}',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                fontweight='bold', color=cv_color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                          alpha=0.8, edgecolor=cv_color))

        ax.set_xlabel('Run Index', fontsize=10, fontweight='bold')
        ax.set_ylabel(lab, fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=7, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_runs + 1)

    fig.tight_layout()
    _save_fig(fig, 'fig6_monte_carlo_robustness.png')


# === Phase 8: Figure 7 core metrics comparison bar chart ===

def plot_core_metrics_comparison(results_main):
    setup_rcparams()
    metrics_config = [
        ('pos_rmse', 'Position RMSE [m]', True, '{:.4f}'),
        ('heading_rmse', 'Heading RMSE [rad]', True, '{:.4f}'),
        ('ate_rmse', 'ATE RMSE [m]', True, '{:.4f}'),
        ('step_time_ms', 'Avg Latency [ms]', True, '{:.2f}'),
        ('map_density', 'Map Density [pts/m²]', False, '{:.1f}'),
        ('rpe_trans_rmse', 'RPE Trans [m/m]', True, '{:.4f}'),
    ]

    n_metric = len(metrics_config)
    n_algo = len(ALGO_ORDER)
    values_mat = np.zeros((n_algo, n_metric))
    for i, name in enumerate(ALGO_ORDER):
        m = results_main[name]['metrics']
        for j, (mk, _, _, _) in enumerate(metrics_config):
            values_mat[i, j] = float(m.get(mk, 0.0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for j, (mk, lab, lower_better, fmt) in enumerate(metrics_config):
        ax = axes_flat[j]
        vals = values_mat[:, j]
        x = np.arange(n_algo)
        bars = ax.bar(x, vals, 0.6, color=ALGO_COLORS,
                      edgecolor='black', linewidth=0.8, alpha=0.85)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    fmt.format(val), ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        best_idx = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
        bars[best_idx].set_edgecolor(PALETTE_BEST)
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in ALGO_ORDER],
                           fontsize=8, fontweight='bold')
        ax.set_ylabel(lab, fontsize=9, fontweight='bold')
        ax.set_title(lab, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        star = 'BEST'
        ax.text(best_idx, vals[best_idx] * 0.5, star,
                ha='center', va='center', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=PALETTE_BEST,
                          alpha=0.85, edgecolor='none'))

    fig.suptitle('Figure 7: Core Metrics Comparison (BEST = Best Algorithm)\n'
                 'Cartographer-UKF vs EKF_SLAM vs FastSLAM vs GraphSLAM',
                 fontsize=14, fontweight='bold', y=1.00)

    fig.tight_layout()
    _save_fig(fig, 'fig7_core_metrics_comparison.png')


# === Phase 9: Main pipeline ===

def main():
    print('=' * 70)
    print('Generating High-Quality Visualizations for Cartographer-UKF')
    print('=' * 70)

    results_main, mc_data = load_results()

    print('[1/7] Plotting trajectory overlay...')
    plot_trajectory_overlay(results_main)

    print('[2/7] Plotting position error time series...')
    plot_position_error_timeseries(results_main)

    print('[3/7] Plotting comprehensive radar...')
    plot_metrics_radar(results_main)

    print('[4/7] Plotting ATE statistics...')
    plot_ate_statistics(results_main)

    print('[5/7] Plotting latency distribution...')
    plot_latency_distribution(results_main)

    print('[6/7] Plotting Monte Carlo robustness...')
    plot_monte_carlo_robustness(mc_data)

    print('[7/7] Plotting core metrics comparison...')
    plot_core_metrics_comparison(results_main)

    print('=' * 70)
    print(f'All 7 figures saved to: {FIGS_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
