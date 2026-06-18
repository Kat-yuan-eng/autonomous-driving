"""
Parameter sweep and optimization for LQR-Stanley controller

author: Kat-yuan-eng (RuiWen Liao)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import time
import numpy as np
import importlib
import json
import Control.config as cfg
import Control.lqr.lqr_controller as lqr_mod
import Control.stanley.stanley_controller as stan_mod
import Control.speed_control.speed_controller as spd_mod
import Control.compare_controllers as ctrl_mod
from Control.config import METRIC_WEIGHTS

PP_BASELINE = {
    'straight':  {'rmse_lat':0.052375,'max_lat':0.200000,'smoothness':0.00000280,'step_time_ms':0.022,'rmse_v':0.432072,'rmse_theta':0.053098},
    's_curve':   {'rmse_lat':0.021742,'max_lat':0.142729,'smoothness':0.00000460,'step_time_ms':0.023,'rmse_v':0.241596,'rmse_theta':0.117732},
    'sharp_turn':{'rmse_lat':0.061924,'max_lat':0.200000,'smoothness':0.00000729,'step_time_ms':0.022,'rmse_v':0.563415,'rmse_theta':0.063451},
    'low_speed': {'rmse_lat':0.070987,'max_lat':0.200000,'smoothness':0.00000158,'step_time_ms':0.022,'rmse_v':0.029303,'rmse_theta':0.090957},
    'combined':  {'rmse_lat':0.044438,'max_lat':0.200000,'smoothness':0.00000670,'step_time_ms':0.022,'rmse_v':0.492971,'rmse_theta':0.059657},
}
STANLEY_BASELINE = {
    'straight':  {'rmse_lat':0.860360,'max_lat':1.249670,'smoothness':0.00007333,'step_time_ms':0.023,'rmse_v':0.218283,'rmse_theta':2.737374},
    's_curve':   {'rmse_lat':0.617218,'max_lat':0.782548,'smoothness':0.00007352,'step_time_ms':0.023,'rmse_v':0.147405,'rmse_theta':2.833549},
    'sharp_turn':{'rmse_lat':0.860360,'max_lat':1.249670,'smoothness':0.00007333,'step_time_ms':0.023,'rmse_v':0.218283,'rmse_theta':2.737374},
    'low_speed': {'rmse_lat':1.127872,'max_lat':1.241470,'smoothness':0.00005526,'step_time_ms':0.023,'rmse_v':0.018290,'rmse_theta':2.788261},
    'combined':  {'rmse_lat':0.860360,'max_lat':1.249670,'smoothness':0.00007333,'step_time_ms':0.023,'rmse_v':0.218283,'rmse_theta':2.737374},
}
SMC_BASELINE = {
    'straight':  {'rmse_lat':0.566907,'max_lat':0.883982,'smoothness':0.00004152,'step_time_ms':0.025,'rmse_v':0.264818,'rmse_theta':1.087058},
    's_curve':   {'rmse_lat':0.446628,'max_lat':0.775831,'smoothness':0.00003654,'step_time_ms':0.025,'rmse_v':0.198614,'rmse_theta':0.954094},
    'sharp_turn':{'rmse_lat':0.512004,'max_lat':0.883982,'smoothness':0.00003888,'step_time_ms':0.025,'rmse_v':0.339505,'rmse_theta':1.033622},
    'low_speed': {'rmse_lat':0.094917,'max_lat':0.200000,'smoothness':0.00001537,'step_time_ms':0.024,'rmse_v':0.020385,'rmse_theta':0.380865},
    'combined':  {'rmse_lat':0.554423,'max_lat':0.903265,'smoothness':0.00003748,'step_time_ms':0.025,'rmse_v':0.421201,'rmse_theta':1.068690},
}

ALGO_ORDER = ['LQR-Stanley', 'PurePursuit', 'Stanley-only', 'SMC']
COURSE_TYPES = ["straight", "s_curve", "sharp_turn", "low_speed", "combined"]
METRIC_KEYS = ['rmse_lat', 'max_lat', 'smoothness', 'step_time_ms', 'rmse_v', 'rmse_theta']


def compute_weighted_score(lqr_metrics):
    """
    Compute weighted performance score for LQR-Stanley against baselines.

    :param lqr_metrics: (dict) Course type to LQR-Stanley metrics
    :return: (float) Weighted score (lower is better)
    """
    raw = np.zeros((4, 5, 6))
    baselines = {'LQR-Stanley': lqr_metrics, 'PurePursuit': PP_BASELINE,
                 'Stanley-only': STANLEY_BASELINE, 'SMC': SMC_BASELINE}
    for a_idx, name in enumerate(ALGO_ORDER):
        for c_idx, ct in enumerate(COURSE_TYPES):
            for m_idx, k in enumerate(METRIC_KEYS):
                raw[a_idx, c_idx, m_idx] = baselines[name][ct][k]
    normed = np.zeros_like(raw)
    for m_idx, k in enumerate(METRIC_KEYS):
        if k == 'step_time_ms':
            normed[:, :, m_idx] = np.clip(raw[:, :, m_idx] / 1.0, 0.0, 1.0)
        else:
            r_min = raw[:, :, m_idx].min()
            r_max = raw[:, :, m_idx].max()
            normed[:, :, m_idx] = (raw[:, :, m_idx] - r_min) / (r_max - r_min + 1e-12)
    s = 0.0
    for c_idx in range(5):
        for m_idx in range(6):
            s += METRIC_WEIGHTS[m_idx] * normed[0, c_idx, m_idx]
    return s / 5.0


def patch_config(params):
    """
    Patch global config parameters and reload dependent modules.

    :param params: (dict) Parameter name to value mapping
    """
    for key, val in params.items():
        setattr(cfg, key, val)
    if any(k.startswith('Q_') or k.startswith('R_') or k.startswith('KAPPA_') for k in params):
        cfg.Q_LQR = np.diag([cfg.Q_LAT_MIN, 1.0, cfg.Q_THETA_MIN, 0.5, 3.0])
        cfg.R_LQR = np.diag([cfg.R_DELTA_MAX, 0.5])
    importlib.reload(lqr_mod)
    importlib.reload(stan_mod)
    importlib.reload(spd_mod)
    importlib.reload(ctrl_mod)


def run_lqr_stanley_only(params):
    """
    Run LQR-Stanley simulation across all course types with given parameters.

    :param params: (dict) Parameter name to value mapping
    :return: (dict) Course type to metrics dict
    """
    patch_config(params)
    gains_path = pathlib.Path(__file__).parent / 'lqr_gains_adaptive.npz'
    if gains_path.exists():
        gains_path.unlink()
    metrics = {}
    for ct in COURSE_TYPES:
        ref_dict = ctrl_mod.generate_reference_course(ct)
        sim = ctrl_mod.run_simulation(
            ctrl_mod.controller_lqr_stanley, ref_dict, max_time=30.0, y0=0.2,
            ctrl_type='lqr_stanley', force_recompute_gains=True)
        met = ctrl_mod.compute_metrics(sim)
        met['step_time_ms'] = float(np.mean(sim['step_times']))
        met['n_degrade'] = int(sim['n_degrade'])
        metrics[ct] = met
    return metrics


# === Phase 1: Sensitivity Analysis ===

SENSITIVITY_PARAMS = {
    'K_STANLEY':    (2.0,   1.6,   2.4,   5),
    'K_SW':         (30.0,  24.0,  36.0,  5),
    'V_SW':         (0.3,   0.24,  0.36,  5),
    'KAPPA_SW':     (1.5,   1.2,   1.8,   5),
    'E_LAT_DEGRADE':(0.10,  0.08,  0.12,  5),
    'DE_LAT_DEGRADE':(0.5,  0.4,   0.6,   5),
    'K_PP_LOW':     (0.1,   0.08,  0.12,  5),
    'LFC_LOW':      (0.5,   0.4,   0.6,   5),
    'KP_V':         (2.0,   1.6,   2.4,   5),
    'KI_V':         (0.1,   0.08,  0.12,  5),
    'KD_V':         (0.3,   0.24,  0.36,  5),
    'TAU_FF':       (0.5,   0.4,   0.6,   5),
    'BETA_SAFE':    (2.0,   1.6,   2.4,   5),
    'T_REACT':      (0.2,   0.16,  0.24,  5),
    'ALPHA_F':      (0.3,   0.24,  0.36,  5),
    'T_LA_FF':      (0.6,   0.48,  0.72,  5),
    'L_LA_MIN':     (0.3,   0.24,  0.36,  5),
    'T_ERR_BASE':   (0.2,   0.16,  0.24,  5),
    'T_ERR_KAPPA':  (0.25,  0.20,  0.30,  5),
    'W_ERR_BASE':   (0.3,   0.24,  0.36,  5),
    'W_ERR_KAPPA':  (0.6,   0.48,  0.72,  5),
    'Q_LAT_MIN':    (20.0,  16.0,  24.0,  5),
    'Q_LAT_MAX':    (60.0,  48.0,  72.0,  5),
    'Q_THETA_MIN':  (5.0,   4.0,   6.0,   5),
    'Q_THETA_MAX':  (15.0,  12.0,  18.0,  5),
    'R_DELTA_MIN':  (0.08,  0.064, 0.096, 5),
    'R_DELTA_MAX':  (0.15,  0.12,  0.18,  5),
    'KAPPA_LOW':    (0.5,   0.4,   0.6,   5),
    'KAPPA_HIGH':   (3.0,   2.4,   3.6,   5),
    'LAM_SMC':      (3.0,   2.4,   3.6,   5),
    'ETA_SMC':      (0.8,   0.64,  0.96,  5),
    'PHI_SMC':      (0.05,  0.04,  0.06,  5),
    'E_LAT_TH':     (0.15,  0.12,  0.18,  5),
    'INTEGRAL_LIMIT':(1.0,  0.8,   1.2,   5),
    'A_LAT_MAX':    (1.5,   1.2,   1.8,   5),
}


def phase1_sensitivity():
    """
    Run one-at-a-time sensitivity analysis on all tunable parameters.

    :return: (list) Ranked (param_name, result_dict) sorted by sensitivity descending
    """
    print("=" * 70)
    print("Phase 1: Sensitivity Analysis (one-at-a-time ±20%)")
    print("=" * 70)
    baseline_score = compute_weighted_score(run_lqr_stanley_only({}))
    print(f"Baseline score: {baseline_score:.4f}\n")
    results = {}
    for pname, (nominal, lo, hi, n_steps) in SENSITIVITY_PARAMS.items():
        values = np.linspace(lo, hi, n_steps)
        scores = []
        for val in values:
            m = run_lqr_stanley_only({pname: val})
            s = compute_weighted_score(m)
            scores.append(s)
        sensitivity = max(scores) - min(scores)
        best_val = values[np.argmin(scores)]
        best_score = min(scores)
        improvement = baseline_score - best_score
        results[pname] = {
            'nominal': nominal, 'best_val': best_val, 'best_score': best_score,
            'sensitivity': sensitivity, 'improvement': improvement,
            'values': values.tolist(), 'scores': scores,
        }
        print(f"  {pname:20s}  nominal={nominal:8.3f}  best={best_val:8.4f}  "
              f"sensitivity={sensitivity:.4f}  improvement={improvement:+.4f}")
    ranked = sorted(results.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
    print(f"\n{'='*70}")
    print("Ranked by sensitivity (most → least):")
    for i, (pname, r) in enumerate(ranked[:15]):
        print(f"  {i+1:2d}. {pname:20s}  sensitivity={r['sensitivity']:.4f}  "
              f"best={r['best_val']:.4f}  improvement={r['improvement']:+.4f}")
    with open(pathlib.Path(__file__).parent / 'results' / 'sensitivity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return ranked


# === Phase 2: Coordinate Descent ===

def phase2_coordinate_descent(ranked_params, top_n=10, n_rounds=4, n_levels=9):
    """
    Coordinate descent optimization on top-N most sensitive parameters.

    :param ranked_params: (list) Ranked (param_name, result_dict) from sensitivity analysis
    :param top_n: (int) Number of top parameters to optimize
    :param n_rounds: (int) Number of coordinate descent rounds
    :param n_levels: (int) Number of levels per parameter sweep
    :return: (tuple) (best_params, best_score) optimized parameters and score
    """
    print(f"\n{'='*70}")
    print(f"Phase 2: Coordinate Descent on top {top_n} parameters")
    print(f"  {n_rounds} rounds × {n_levels} levels per param = ~{top_n * n_levels * n_rounds} evals")
    print("=" * 70)
    top_names = [pname for pname, _ in ranked_params[:top_n]]
    for pname in top_names:
        r = dict(ranked_params)[pname]
        print(f"  {pname}: best={r['best_val']:.4f} (nominal={r['nominal']:.3f})")
    current_params = {pname: dict(ranked_params)[pname]['best_val'] for pname in top_names}
    current_score = compute_weighted_score(run_lqr_stanley_only(current_params))
    print(f"\n  Initial score: {current_score:.4f}")
    for rnd in range(n_rounds):
        print(f"\n  Round {rnd+1}/{n_rounds}:")
        improved = False
        for pname in top_names:
            center = current_params[pname]
            nominal = SENSITIVITY_PARAMS[pname][0]
            shrink = 0.5 ** rnd
            span = abs(nominal) * 0.25 * shrink
            lo = center - span
            hi = center + span
            values = np.linspace(lo, hi, n_levels)
            best_val = center
            best_s = current_score
            for val in values:
                test_params = current_params.copy()
                test_params[pname] = val
                metrics = run_lqr_stanley_only(test_params)
                score = compute_weighted_score(metrics)
                if score < best_s:
                    best_s = score
                    best_val = val
            if best_val != center:
                current_params[pname] = best_val
                current_score = best_s
                improved = True
                print(f"    {pname}: {center:.6f} → {best_val:.6f}  score={best_s:.4f}")
        if not improved:
            print(f"    No improvement in round {rnd+1}, stopping.")
            break
    print(f"\n  Phase 2 final score: {current_score:.4f}")
    print(f"  Optimized parameters:")
    for k, v in current_params.items():
        print(f"    {k} = {v:.6f}")
    return current_params, current_score


# === Phase 3: High-Precision Fine-Tuning ===

def phase3_finetune(best_params, best_score, n_rounds=3, n_levels=11):
    """
    High-precision fine-tuning around the best parameter values.

    :param best_params: (dict) Best parameters from coordinate descent
    :param best_score: (float) Best score from coordinate descent
    :param n_rounds: (int) Number of fine-tuning rounds
    :param n_levels: (int) Number of levels per parameter sweep
    :return: (tuple) (final_params, final_score) fine-tuned parameters and score
    """
    print(f"\n{'='*70}")
    print(f"Phase 3: High-precision fine-tuning ({n_rounds} rounds, {n_levels} levels)")
    print("=" * 70)
    current_params = best_params.copy()
    current_score = best_score
    for rnd in range(n_rounds):
        print(f"\n  Round {rnd+1}/{n_rounds}:")
        improved = False
        for pname in list(current_params.keys()):
            center = current_params[pname]
            step = center * 0.005
            lo = center - step * (n_levels // 2)
            hi = center + step * (n_levels // 2)
            values = np.linspace(lo, hi, n_levels)
            best_val = center
            best_s = current_score
            for val in values:
                test_params = current_params.copy()
                test_params[pname] = val
                metrics = run_lqr_stanley_only(test_params)
                score = compute_weighted_score(metrics)
                if score < best_s:
                    best_s = score
                    best_val = val
            if best_val != center:
                current_params[pname] = best_val
                current_score = best_s
                improved = True
                print(f"    {pname}: {center:.6f} → {best_val:.6f}  score={best_s:.4f}")
        if not improved:
            print(f"    No improvement in round {rnd+1}, stopping.")
            break
    print(f"\n  Phase 3 final score: {current_score:.4f}")
    print(f"  Final parameters (high precision):")
    for k, v in current_params.items():
        print(f"    {k} = {v:.6f}")
    return current_params, current_score


def main():
    """
    Run full three-phase parameter optimization pipeline.
    """
    results_dir = pathlib.Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    sens_path = results_dir / 'sensitivity_results.json'
    if sens_path.exists():
        with open(sens_path) as f:
            sens_data = json.load(f)
        ranked = sorted(sens_data.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
        print("Phase 1: Loaded cached sensitivity results")
        for i, (pname, r) in enumerate(ranked[:10]):
            print(f"  {i+1}. {pname:20s}  sensitivity={r['sensitivity']:.4f}  best={r['best_val']:.4f}")
    else:
        ranked = phase1_sensitivity()
    top_n = min(10, len(ranked))
    best_params, best_score = phase2_coordinate_descent(ranked, top_n=top_n, n_rounds=4, n_levels=9)
    final_params, final_score = phase3_finetune(best_params, best_score, n_rounds=3, n_levels=11)
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Baseline score: {compute_weighted_score(run_lqr_stanley_only({})):.4f}")
    print(f"  Optimized score: {final_score:.4f}")
    print(f"  Improvement: {compute_weighted_score(run_lqr_stanley_only({})) - final_score:.4f}")
    print(f"\n  Optimized parameters (high precision):")
    for k, v in final_params.items():
        nominal = SENSITIVITY_PARAMS.get(k, (0,))[0]
        print(f"    {k} = {v:.6f}  (nominal: {nominal})")
    with open(results_dir / 'optimized_params.json', 'w') as f:
        json.dump({'params': {k: float(v) for k, v in final_params.items()},
                   'score': float(final_score)}, f, indent=2)


if __name__ == '__main__':
    main()
