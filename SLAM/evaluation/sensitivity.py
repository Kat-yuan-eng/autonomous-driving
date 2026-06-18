import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from SLAM import config
from SLAM.config import (UKF_DT, IMU_DT, LIDAR_N_BEAMS, LIDAR_RANGE_MAX,
    WHEEL_SIGMA_V, WHEEL_SIGMA_W)
from SLAM.slam_sim import (generate_reference_trajectory, generate_landmarks,
    generate_lidar_scan, generate_imu_batch, generate_wheel_odom, angle_mod)
from SLAM.compare_slam import run_cartographer_ukf
from SLAM.evaluation.metrics import compute_rpe, compute_ate_tum, compute_latency_profile

ROOT = pathlib.Path(__file__).parent.parent
FIGS_DIR = ROOT / 'figs'
RESULTS_DIR = ROOT / 'results'

_UKF_Q_BASE = config.UKF_Q.copy()
_SEARCH_WIN_LIN_BASE = float(config.SEARCH_WIN_LIN)
_SCORE_HEALTHY_BASE = 0.6

PARAM_BASELINE = {
    'UKF_Q': 1.0,
    'SEARCH_WIN_LIN': _SEARCH_WIN_LIN_BASE,
    'SCORE_HEALTHY': _SCORE_HEALTHY_BASE,
}

NOISE_PARAMS = ['ACCEL_SIGMA', 'GYRO_SIGMA', 'WHEEL_SIGMA_V',
                'WHEEL_SIGMA_W', 'LIDAR_SIGMA_RANGE', 'LIDAR_SIGMA_BEARING']


# === Phase 1: Parameter monkey patch tool ===

def _param_transform(name, value):
    if name == 'UKF_Q':
        return float(value) * _UKF_Q_BASE
    return float(value)


def _capture_originals(param_names):
    originals = {}
    for name in param_names:
        store = {}
        if hasattr(config, name):
            store['config'] = getattr(config, name)
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith('SLAM'):
                continue
            if hasattr(mod, name):
                store[mod_name] = getattr(mod, name)
        originals[name] = store
    return originals


def _apply_param(name, value):
    setattr(config, name, value)
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith('SLAM'):
            continue
        if hasattr(mod, name):
            setattr(mod, name, value)


def _restore_originals(originals):
    for name, store in originals.items():
        for loc, val in store.items():
            if loc == 'config':
                setattr(config, name, val)
            else:
                mod = sys.modules.get(loc)
                if mod is not None:
                    setattr(mod, name, val)


# === Phase 2: Single evaluation ===

def run_single_eval(params_dict, seed=42, n_steps=200, noise_scale=1.0):
    assert isinstance(params_dict, dict), f"params_dict must be dict, got {type(params_dict)}"
    assert n_steps >= 20, f"n_steps too small: {n_steps}, need >=20 for stable metrics"
    assert noise_scale > 0, f"noise_scale must be >0, got {noise_scale}"
    for name in params_dict:
        assert name in PARAM_BASELINE, f"unknown param {name}, supported: {list(PARAM_BASELINE)}"

    np.random.seed(seed)
    patch_names = list(params_dict.keys()) + NOISE_PARAMS
    originals = _capture_originals(patch_names)
    for name, v in params_dict.items():
        _apply_param(name, _param_transform(name, v))
    for name in NOISE_PARAMS:
        base_val = float(getattr(config, name))
        _apply_param(name, base_val * noise_scale)

    ref_traj_full = generate_reference_trajectory('figure8', UKF_DT)
    n_use = min(n_steps, len(ref_traj_full))
    ref_traj = ref_traj_full[:n_use].copy()
    landmarks = generate_landmarks(50, 10.0)
    n_imu_sub = max(1, int(UKF_DT / IMU_DT))

    lidar_scans = np.zeros((n_use, LIDAR_N_BEAMS))
    for i in range(n_use):
        lidar_scans[i] = generate_lidar_scan(ref_traj[i, 0], ref_traj[i, 1],
                                              ref_traj[i, 2], landmarks)

    ddx = np.diff(ref_traj[:, 0])
    ddy = np.diff(ref_traj[:, 1])
    ddtheta = angle_mod(np.diff(ref_traj[:, 2]))
    v_true_arr = np.concatenate([[0.0], np.sqrt(ddx**2 + ddy**2) / UKF_DT])
    omega_true_arr = np.concatenate([[0.0], ddtheta / UKF_DT])
    v_noise = np.random.randn(n_use) * float(getattr(config, 'WHEEL_SIGMA_V'))
    w_noise = np.random.randn(n_use) * float(getattr(config, 'WHEEL_SIGMA_W'))
    wheel_scans = np.column_stack([v_true_arr + v_noise, omega_true_arr + w_noise])

    imu_scans = generate_imu_batch(n_use, n_imu_sub, IMU_DT, ref_traj=ref_traj)

    traj_est, m_basic = run_cartographer_ukf(
        ref_traj, landmarks, lidar_scans, imu_scans, wheel_scans)
    _restore_originals(originals)

    n = min(len(traj_est), len(ref_traj))
    rpe = compute_rpe(traj_est[:n], ref_traj[:n], delta_m=1.0)
    ate = compute_ate_tum(traj_est[:n], ref_traj[:n], align='se3')
    step_times_ms = m_basic.get('step_times_ms', [m_basic.get('step_time_ms', 0.0)])
    lat = compute_latency_profile(step_times_ms, window=10)

    metrics = {
        'ate_rmse': round(float(ate['ate_rmse']), 6),
        'rpe_trans_rmse': round(float(rpe['rpe_trans_rmse']), 6),
        'rpe_rot_rmse': round(float(rpe['rpe_rot_rmse']), 6),
        'latency_p95_ms': round(float(lat['latency_p95_ms']), 6),
        'pos_rmse': round(float(m_basic['pos_rmse']), 6),
        'heading_rmse': round(float(m_basic['heading_rmse']), 6),
    }
    return metrics


# === Phase 3: Grid scan ===

def grid_scan_param(param_name, base_value, run_func, scan_range=0.5, n_points=11):
    assert isinstance(param_name, str), f"param_name must be str, got {type(param_name)}"
    assert 0 < scan_range <= 1.0, f"scan_range must be in (0,1], got {scan_range}"
    assert n_points >= 5, f"n_points must be >=5, got {n_points}"
    assert callable(run_func), f"run_func must be callable, got {type(run_func)}"

    values = np.linspace(base_value * (1 - scan_range),
                          base_value * (1 + scan_range), n_points)
    values = np.round(values, 6)

    ate_arr, rpe_t_arr, rpe_r_arr, lat_arr = [], [], [], []
    for v in values:
        metrics = run_func(float(v))
        ate_arr.append(metrics['ate_rmse'])
        rpe_t_arr.append(metrics['rpe_trans_rmse'])
        rpe_r_arr.append(metrics['rpe_rot_rmse'])
        lat_arr.append(metrics['latency_p95_ms'])

    return {
        'param_name': param_name,
        'values': np.round(np.array(values), 6),
        'ate': np.round(np.array(ate_arr), 6),
        'rpe_trans': np.round(np.array(rpe_t_arr), 6),
        'rpe_rot': np.round(np.array(rpe_r_arr), 6),
        'latency_p95': np.round(np.array(lat_arr), 6),
    }


# === Phase 4: Bayesian optimization ===

def _score_metrics(metrics):
    return (float(metrics['ate_rmse'])
            + float(metrics['rpe_trans_rmse'])
            + 0.1 * float(metrics['latency_p95_ms']))


def bayesian_optimize(param_bounds, run_func, n_iter=30, n_mc=20):
    assert isinstance(param_bounds, dict) and len(param_bounds) >= 1, \
        f"param_bounds must be non-empty dict, got {param_bounds}"
    assert callable(run_func), f"run_func must be callable, got {type(run_func)}"
    assert n_iter >= 5, f"n_iter too small: {n_iter}, need >=5"
    assert n_mc >= 1, f"n_mc must be >=1, got {n_mc}"

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from scipy.stats import norm

    param_names = list(param_bounds.keys())
    n_params = len(param_names)
    bounds = np.array([[float(lo), float(hi)] for lo, hi in param_bounds.values()])
    assert np.all(bounds[:, 1] > bounds[:, 0]), f"invalid bounds: {bounds}"

    def _eval_mean(params_vec):
        params_dict = {name: round(float(v), 6)
                       for name, v in zip(param_names, params_vec)}
        scores = []
        for _ in range(n_mc):
            metrics = run_func(params_dict)
            scores.append(_score_metrics(metrics))
        return float(np.mean(scores))

    rng = np.random.RandomState(42)
    n_init = max(5, 2 * n_params)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, n_params))
    y = np.array([_eval_mean(x) for x in X])

    history = []
    best_idx = int(np.argmin(y))
    best_params = {name: round(float(v), 6)
                   for name, v in zip(param_names, X[best_idx])}
    best_score = round(float(y[best_idx]), 6)
    history.append({'params': dict(best_params), 'score': best_score})

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_params))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                   random_state=42)

    for _ in range(n_iter):
        gp.fit(X, y)
        n_candidates = 1000
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1],
                              size=(n_candidates, n_params))
        mu, sigma = gp.predict(X_cand, return_std=True)
        y_best = float(np.min(y))
        improvement = y_best - mu
        sigma_safe = np.maximum(sigma, 1e-9)
        Z = improvement / sigma_safe
        ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        next_idx = int(np.argmax(ei))
        x_next = X_cand[next_idx]
        y_next = _eval_mean(x_next)
        X = np.vstack([X, x_next])
        y = np.concatenate([y, [y_next]])

        cur_best_idx = int(np.argmin(y))
        cur_params = {name: round(float(v), 6)
                      for name, v in zip(param_names, X[cur_best_idx])}
        cur_score = round(float(y[cur_best_idx]), 6)
        history.append({'params': dict(cur_params), 'score': cur_score})
        if cur_score < best_score:
            best_score = cur_score
            best_params = dict(cur_params)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'history': history,
    }


# === Phase 5: Sensitivity visualization ===

def plot_sensitivity_heatmap(sensitivity_data, param_name):
    assert isinstance(sensitivity_data, dict), \
        f"sensitivity_data must be dict, got {type(sensitivity_data)}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    values = np.asarray(sensitivity_data['values'])
    ate = np.asarray(sensitivity_data['ate'])
    rpe_t = np.asarray(sensitivity_data['rpe_trans'])
    rpe_r = np.asarray(sensitivity_data['rpe_rot'])
    lat = np.asarray(sensitivity_data['latency_p95'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    panels = [
        (axes[0, 0], ate, 'ATE RMSE [m]', '#1f77b4'),
        (axes[0, 1], rpe_t, 'RPE Trans RMSE [m/m]', '#ff7f0e'),
        (axes[1, 0], rpe_r, 'RPE Rot RMSE [rad/m]', '#2ca02c'),
        (axes[1, 1], lat, 'Latency p95 [ms]', '#d62728'),
    ]
    for ax, y, ylabel, color in panels:
        ax.plot(values, y, 'o-', color=color, linewidth=1.5, markersize=5)
        i_min = int(np.argmin(y))
        ax.plot(values[i_min], y[i_min], '*', color='red', markersize=16,
                zorder=5, label=f'min={y[i_min]:.6f}@{values[i_min]:.6f}')
        ax.axhline(y=float(np.mean(y)), color=color, linestyle=':', alpha=0.5)
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs {param_name}')
        ax.legend(frameon=True, fancybox=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=13)
    fig.tight_layout()
    fname = f'sensitivity_{param_name}.png'
    fig.savefig(str(FIGS_DIR / fname), dpi=100)
    fig.savefig(str(RESULTS_DIR / fname), dpi=100)
    plt.close(fig)
    print(f"[save] sensitivity {param_name} -> {FIGS_DIR / fname}")


# === Phase 6: 5-parameter full sensitivity orchestration ===

def run_all_sensitivity(n_steps=100, noise_scale=0.3):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scan_specs = [
        ('UKF_Q', 1.0, 0.5),
        ('SEARCH_WIN_LIN', _SEARCH_WIN_LIN_BASE, 0.5),
        ('SCORE_HEALTHY', _SCORE_HEALTHY_BASE, 0.5),
    ]

    summary = {}
    for name, base, scan_range in scan_specs:
        print(f"[scan] {name} base={base} scan_range={scan_range}")

        def _run_func(v, _name=name, _n_steps=n_steps, _ns=noise_scale):
            return run_single_eval({_name: v}, seed=42, n_steps=_n_steps,
                                    noise_scale=_ns)

        sens = grid_scan_param(name, base, _run_func,
                                scan_range=scan_range, n_points=11)
        plot_sensitivity_heatmap(sens, name)

        ate_arr = np.asarray(sens['ate'])
        i_best = int(np.argmin(ate_arr))
        summary[name] = {
            'base_value': round(float(base), 6),
            'best_value': round(float(sens['values'][i_best]), 6),
            'best_ate': round(float(ate_arr[i_best]), 6),
            'ate_min': round(float(np.min(ate_arr)), 6),
            'ate_max': round(float(np.max(ate_arr)), 6),
            'ate_fluct_pct': round(
                float((np.max(ate_arr) - np.min(ate_arr))
                      / max(np.mean(ate_arr), 1e-9) * 100), 6),
        }
        print(f"[scan] {name} best={summary[name]['best_value']} "
              f"ate={summary[name]['best_ate']} "
              f"fluct={summary[name]['ate_fluct_pct']}%")

    out_path = RESULTS_DIR / 'sensitivity_summary.json'
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[save] sensitivity summary -> {out_path}")
    return summary


if __name__ == '__main__':
    run_all_sensitivity(n_steps=100, noise_scale=0.3)
