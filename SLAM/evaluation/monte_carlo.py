import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from SLAM.evaluation.sensitivity import run_single_eval

ROOT = pathlib.Path(__file__).parent.parent
FIGS_DIR = ROOT / 'figs'
RESULTS_DIR = ROOT / 'results'


# === Phase 1: Monte Carlo robustness test ===

def _stats_arr(arr):
    arr_f = np.asarray(arr, dtype=float)
    mean = round(float(np.mean(arr_f)), 6)
    std = round(float(np.std(arr_f)), 6)
    fluct = round(float((np.max(arr_f) - np.min(arr_f))
                         / max(mean, 1e-9) * 100), 6)
    return mean, std, fluct


def run_monte_carlo(n_runs=20, seed_base=42, params_dict=None, n_steps=100,
                    noise_scale=1.0):
    assert n_runs >= 5, f"n_runs too small: {n_runs}, need >=5"
    assert seed_base >= 0, f"seed_base must be >=0, got {seed_base}"
    base_params = dict(params_dict) if params_dict else {}

    ate_runs, rpe_t_runs, rpe_r_runs = [], [], []
    pos_runs, head_runs, lat_runs = [], [], []

    for i in range(n_runs):
        seed = seed_base + i
        metrics = run_single_eval(base_params, seed=seed, n_steps=n_steps,
                                   noise_scale=noise_scale)
        ate_runs.append(metrics['ate_rmse'])
        rpe_t_runs.append(metrics['rpe_trans_rmse'])
        rpe_r_runs.append(metrics['rpe_rot_rmse'])
        pos_runs.append(metrics['pos_rmse'])
        head_runs.append(metrics['heading_rmse'])
        lat_runs.append(metrics['latency_p95_ms'])
        print(f"[mc] run={i+1}/{n_runs} seed={seed} "
              f"ate={metrics['ate_rmse']:.6f} pos={metrics['pos_rmse']:.6f}")

    ate_arr = np.round(np.array(ate_runs), 6)
    rpe_t_arr = np.round(np.array(rpe_t_runs), 6)
    rpe_r_arr = np.round(np.array(rpe_r_runs), 6)
    pos_arr = np.round(np.array(pos_runs), 6)
    head_arr = np.round(np.array(head_runs), 6)
    lat_arr = np.round(np.array(lat_runs), 6)

    ate_mean, ate_std, ate_fluct = _stats_arr(ate_arr)
    rpe_t_mean, rpe_t_std, rpe_t_fluct = _stats_arr(rpe_t_arr)
    rpe_r_mean, rpe_r_std, rpe_r_fluct = _stats_arr(rpe_r_arr)
    pos_mean, pos_std, pos_fluct = _stats_arr(pos_arr)
    head_mean, head_std, head_fluct = _stats_arr(head_arr)
    lat_mean, lat_std, lat_fluct = _stats_arr(lat_arr)

    assert ate_fluct <= 5.0, \
        f"ATE 波动幅度 {ate_fluct:.6f}% 超过 5%, 需检查系统鲁棒性"

    return {
        'n_runs': int(n_runs),
        'seed_base': int(seed_base),
        'ate_runs': ate_arr,
        'rpe_trans_runs': rpe_t_arr,
        'rpe_rot_runs': rpe_r_arr,
        'pos_rmse_runs': pos_arr,
        'heading_rmse_runs': head_arr,
        'latency_runs': lat_arr,
        'ate_mean': ate_mean, 'ate_std': ate_std,
        'ate_fluctuation_pct': ate_fluct,
        'rpe_trans_mean': rpe_t_mean, 'rpe_trans_std': rpe_t_std,
        'rpe_trans_fluctuation_pct': rpe_t_fluct,
        'rpe_rot_mean': rpe_r_mean, 'rpe_rot_std': rpe_r_std,
        'rpe_rot_fluctuation_pct': rpe_r_fluct,
        'pos_rmse_mean': pos_mean, 'pos_rmse_std': pos_std,
        'pos_rmse_fluctuation_pct': pos_fluct,
        'heading_rmse_mean': head_mean, 'heading_rmse_std': head_std,
        'heading_rmse_fluctuation_pct': head_fluct,
        'latency_mean': lat_mean, 'latency_std': lat_std,
        'latency_fluctuation_pct': lat_fluct,
    }


# === Phase 2: Monte Carlo boxplot ===

def plot_monte_carlo_boxplot(mc_results):
    assert isinstance(mc_results, dict), \
        f"mc_results must be dict, got {type(mc_results)}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panels = [
        (mc_results['ate_runs'], 'ATE RMSE [m]', mc_results['ate_mean'],
         mc_results['ate_std'], '#1f77b4'),
        (mc_results['rpe_trans_runs'], 'RPE Trans RMSE [m/m]',
         mc_results['rpe_trans_mean'], mc_results['rpe_trans_std'], '#ff7f0e'),
        (mc_results['pos_rmse_runs'], 'Pos RMSE [m]',
         mc_results['pos_rmse_mean'], mc_results['pos_rmse_std'], '#2ca02c'),
        (mc_results['heading_rmse_runs'], 'Heading RMSE [rad]',
         mc_results['heading_rmse_mean'], mc_results['heading_rmse_std'],
         '#d62728'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    for ax, (arr, ylabel, mean, std, color) in zip(axes.flat, panels):
        arr_f = np.asarray(arr, dtype=float)
        bp = ax.boxplot(arr_f, patch_artist=True, widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red',
                                        markersize=7, markeredgecolor='red'))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.axhline(y=mean, color='red', linestyle='--', linewidth=1.0,
                    label=f'mean={mean:.6f}')
        ax.axhline(y=mean + std, color='blue', linestyle=':', linewidth=0.8,
                    label=f'+1sigma={mean+std:.6f}')
        ax.axhline(y=mean - std, color='blue', linestyle=':', linewidth=0.8,
                    label=f'-1sigma={mean-std:.6f}')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} (Monte Carlo n={len(arr_f)})')
        ax.legend(frameon=True, fancybox=True, fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Monte Carlo Robustness Test', fontsize=13)
    fig.tight_layout()
    fname = 'monte_carlo_boxplot.png'
    fig.savefig(str(FIGS_DIR / fname), dpi=100)
    fig.savefig(str(RESULTS_DIR / fname), dpi=100)
    plt.close(fig)
    print(f"[save] monte carlo boxplot -> {FIGS_DIR / fname}")


# === Phase 3: Monte Carlo full orchestration ===

def run_all_monte_carlo(n_runs=20, seed_base=42, n_steps=100, params_dict=None,
                        noise_scale=1.0):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mc = run_monte_carlo(n_runs=n_runs, seed_base=seed_base,
                          params_dict=params_dict, n_steps=n_steps,
                          noise_scale=noise_scale)
    plot_monte_carlo_boxplot(mc)

    summary = {
        'n_runs': mc['n_runs'],
        'seed_base': mc['seed_base'],
        'ate_mean': mc['ate_mean'], 'ate_std': mc['ate_std'],
        'ate_fluctuation_pct': mc['ate_fluctuation_pct'],
        'rpe_trans_mean': mc['rpe_trans_mean'],
        'rpe_trans_std': mc['rpe_trans_std'],
        'rpe_trans_fluctuation_pct': mc['rpe_trans_fluctuation_pct'],
        'rpe_rot_mean': mc['rpe_rot_mean'],
        'rpe_rot_std': mc['rpe_rot_std'],
        'rpe_rot_fluctuation_pct': mc['rpe_rot_fluctuation_pct'],
        'pos_rmse_mean': mc['pos_rmse_mean'],
        'pos_rmse_std': mc['pos_rmse_std'],
        'pos_rmse_fluctuation_pct': mc['pos_rmse_fluctuation_pct'],
        'heading_rmse_mean': mc['heading_rmse_mean'],
        'heading_rmse_std': mc['heading_rmse_std'],
        'heading_rmse_fluctuation_pct': mc['heading_rmse_fluctuation_pct'],
        'latency_mean': mc['latency_mean'],
        'latency_std': mc['latency_std'],
        'latency_fluctuation_pct': mc['latency_fluctuation_pct'],
    }
    out_path = RESULTS_DIR / 'monte_carlo_summary.json'
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[save] monte carlo summary -> {out_path}")
    print(f"[mc] ATE mean={mc['ate_mean']} std={mc['ate_std']} "
          f"fluct={mc['ate_fluctuation_pct']}%")
    return mc


if __name__ == '__main__':
    run_all_monte_carlo(n_runs=20, seed_base=42, n_steps=500, noise_scale=0.01)
