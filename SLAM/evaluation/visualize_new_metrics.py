import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from SLAM.config import VIS_COLORS, VIS_DPI


ROOT = pathlib.Path(__file__).parent.parent
FIGS_DIR = ROOT / 'figs'
RESULTS_DIR = ROOT / 'results'


# === Phase 1: Global style ===

def setup_rcparams():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'legend.frameon': True,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def _algo_names(results_dict):
    return [k for k in results_dict if k != 'ground_truth']


def _colors(algo_names):
    return [VIS_COLORS.get(k, '#333333') for k in algo_names]


def _save_fig(fig, name):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGS_DIR / name), dpi=VIS_DPI)
    fig.savefig(str(RESULTS_DIR / name), dpi=VIS_DPI)
    plt.close(fig)
    print(f"[save] {name} -> {FIGS_DIR / name}")


# === Phase 2: RPE comparison bar chart ===

def plot_rpe_comparison(results_dict):
    setup_rcparams()
    algo_names = _algo_names(results_dict)
    colors = _colors(algo_names)
    trans_rmse = [results_dict[n]['metrics'].get('rpe_trans_rmse', 0.0) for n in algo_names]
    rot_rmse = [results_dict[n]['metrics'].get('rpe_rot_rmse', 0.0) for n in algo_names]

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(12, 5), dpi=VIS_DPI)
    x = np.arange(len(algo_names))
    width = 0.6

    bars_t = ax_t.bar(x, trans_rmse, width, color=colors, edgecolor='black', linewidth=0.8)
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(algo_names, rotation=15, ha='right')
    ax_t.set_ylabel('RPE Translation RMSE [m/m]')
    ax_t.set_title('RPE Translation (per meter)')
    for bar, val in zip(bars_t, trans_rmse):
        ax_t.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax_t.grid(True, alpha=0.3, axis='y')

    bars_r = ax_r.bar(x, rot_rmse, width, color=colors, edgecolor='black', linewidth=0.8)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(algo_names, rotation=15, ha='right')
    ax_r.set_ylabel('RPE Rotation RMSE [rad/m]')
    ax_r.set_title('RPE Rotation (per meter)')
    for bar, val in zip(bars_r, rot_rmse):
        ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax_r.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _save_fig(fig, 'slam_compare_rpe.png')


# === Phase 3: Map density comparison ===

def plot_map_density_comparison(results_dict):
    setup_rcparams()
    algo_names = _algo_names(results_dict)
    colors = _colors(algo_names)
    densities = [results_dict[n]['metrics'].get('map_density', 0.0) for n in algo_names]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=VIS_DPI)
    x = np.arange(len(algo_names))
    bars = ax.bar(x, densities, 0.6, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=15, ha='right')
    ax.set_ylabel('Map Density [pts/m^2]')
    ax.set_title('Map Point Cloud Density')
    for bar, val in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_fig(fig, 'slam_compare_map_density.png')


# === Phase 4: Latency distribution comparison ===

def plot_latency_profile(results_dict):
    setup_rcparams()
    algo_names = _algo_names(results_dict)
    colors = _colors(algo_names)
    data = []
    for n in algo_names:
        arr = np.asarray(results_dict[n]['metrics'].get('latency_samples_ms', []), dtype=float)
        if arr.size == 0:
            arr = np.array([results_dict[n]['metrics'].get('step_time_ms', 0.0)])
        data.append(arr)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=VIS_DPI)
    bp = ax.boxplot(data, tick_labels=algo_names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for line, c in zip(bp['medians'], colors):
        line.set_color('black')

    for i, (n, arr) in enumerate(zip(algo_names, data)):
        p95 = float(np.percentile(arr, 95)) if arr.size > 0 else 0.0
        p99 = float(np.percentile(arr, 99)) if arr.size > 0 else 0.0
        ax.hlines(p95, i - 0.3, i + 0.3, colors='red', linestyles='--', linewidth=1.2,
                  label='p95' if i == 0 else None)
        ax.hlines(p99, i - 0.3, i + 0.3, colors='darkred', linestyles=':', linewidth=1.2,
                  label='p99' if i == 0 else None)
        ax.text(i + 0.35, p95, f'p95={p95:.2f}', color='red', fontsize=7, va='center')
        ax.text(i + 0.35, p99, f'p99={p99:.2f}', color='darkred', fontsize=7, va='center')

    ax.set_ylabel('Step Latency [ms]')
    ax.set_title('Latency Distribution (with p95/p99 markers)')
    ax.legend(frameon=True, fancybox=True, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_fig(fig, 'slam_compare_latency.png')


# === Phase 5: ATE full statistics comparison ===

def plot_ate_statistics(results_dict):
    setup_rcparams()
    algo_names = _algo_names(results_dict)
    colors = _colors(algo_names)
    stats_keys = ['ate_rmse', 'ate_mean', 'ate_median', 'ate_max']
    stats_labels = ['RMSE', 'Mean', 'Median', 'Max']

    fig, ax = plt.subplots(figsize=(12, 5), dpi=VIS_DPI)
    n_algo = len(algo_names)
    n_stat = len(stats_keys)
    x = np.arange(n_algo)
    width = 0.8 / n_stat

    for k, (sk, lab) in enumerate(zip(stats_keys, stats_labels)):
        vals = [results_dict[n]['metrics'].get(sk, 0.0) for n in algo_names]
        offset = (k - (n_stat - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=lab,
                      edgecolor='black', linewidth=0.6,
                      color=plt.cm.viridis(k / max(n_stat - 1, 1)))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=15, ha='right')
    ax.set_ylabel('ATE [m]')
    ax.set_title('Absolute Trajectory Error Statistics (Umeyama SE(2) aligned)')
    ax.legend(frameon=True, fancybox=True, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_fig(fig, 'slam_compare_ate_stats.png')


# === Phase 6: Comprehensive radar chart ===

def plot_new_metrics_radar(results_dict):
    setup_rcparams()
    algo_names = _algo_names(results_dict)
    colors = _colors(algo_names)

    dimensions = ['ATE', 'RPE Trans', 'RPE Rot', 'Map Density', 'Latency p95', 'Loop Recall']
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    ate_raw = np.array([results_dict[n]['metrics'].get('ate_rmse',
                                                       results_dict[n]['metrics'].get('ate', 1.0))
                        for n in algo_names], dtype=float)
    rpe_t_raw = np.array([results_dict[n]['metrics'].get('rpe_trans_rmse', 1.0) for n in algo_names],
                         dtype=float)
    rpe_r_raw = np.array([results_dict[n]['metrics'].get('rpe_rot_rmse', 1.0) for n in algo_names],
                         dtype=float)
    dens_raw = np.array([results_dict[n]['metrics'].get('map_density', 1.0) for n in algo_names],
                        dtype=float)
    lat_raw = np.array([results_dict[n]['metrics'].get('latency_p95_ms',
                                                       results_dict[n]['metrics'].get('step_time_ms',
                                                                                       1.0))
                        for n in algo_names], dtype=float)
    loop_raw = np.array([results_dict[n]['metrics'].get('loop_recall', 0.0) for n in algo_names],
                        dtype=float)

    def _norm_lower(v):
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.ones_like(v)
        return 1.0 - (v - vmin) / (vmax - vmin)

    def _norm_higher(v):
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.ones_like(v)
        return (v - vmin) / (vmax - vmin)

    scores = np.column_stack([
        _norm_lower(ate_raw),
        _norm_lower(rpe_t_raw),
        _norm_lower(rpe_r_raw),
        _norm_higher(dens_raw),
        _norm_lower(lat_raw),
        _norm_higher(loop_raw),
    ])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=VIS_DPI)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 1.1)

    for i, name in enumerate(algo_names):
        values = scores[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              frameon=True, fancybox=True, fontsize=9)
    ax.set_title('New Metrics Comprehensive Radar (v3.0)', y=1.08, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, 'slam_compare_new_radar.png')
