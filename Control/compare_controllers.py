"""
Controller comparison: LQR-Stanley vs PurePursuit vs Stanley vs SMC

author: Kat-yuan-eng (RuiWen Liao)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Control.lqr.lqr_controller import (solve_dare, build_state_matrix, precompute_lqr_gains, precompute_adaptive_gains,
    save_gains, load_gains, save_adaptive_gains, load_adaptive_gains,
    find_nearest_point, compute_lateral_error, compute_heading_error, compute_error_state,
    interpolate_gain, interpolate_adaptive_gain, feedforward_compensate, lookahead_curvature,
    lqr_control_adaptive,
    WHEELBASE, V_MAX, DELTA_MAX, A_MAX, DELTA_DOT_MAX, DT, DV_TABLE)
from Control.stanley.stanley_controller import (stanley_control, stanley_speed_control, degradation_check, progressive_blend,
    K_STANLEY, V_MIN_STANLEY, K_SW, V_SW, E_LAT_DEGRADE, KAPPA_SW, DE_LAT_DEGRADE)
from Control.speed_control.speed_controller import (curvature_speed_limit, lookahead_speed_limit, dynamic_safety_margin, pid_speed_control,
    A_LAT_MAX, TAU_FF, KP_V, KI_V, KD_V, E_LAT_TH, BETA_SAFE, T_REACT, INTEGRAL_LIMIT, ALPHA_F)
from Control.pure_pursuit.pure_pursuit import pure_pursuit_steer, controller_pure_pursuit
from Control.smc.smc_controller import controller_smc
from Control.reference_extractor import compute_curvature
from Control.config import (T_LA_FF, L_LA_MIN, T_ERR_BASE, T_ERR_KAPPA, W_ERR_BASE, W_ERR_KAPPA,
    LAM_SMC, ETA_SMC, PHI_SMC, K_PP_LOW, LFC_LOW, METRIC_WEIGHTS,
    COLORS, COLOR_LQR, COLOR_PP, COLOR_STANLEY, COLOR_SMC)
from Control.control_output import clip_control, publish_cmd

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

COLORS = {'LQR-Stanley': COLOR_LQR, 'PurePursuit': COLOR_PP, 'Stanley-only': COLOR_STANLEY, 'SMC': COLOR_SMC}
ALGO_ORDER = ['LQR-Stanley', 'PurePursuit', 'Stanley-only', 'SMC']
LINESTYLE = {'LQR-Stanley': '-', 'PurePursuit': '--', 'Stanley-only': ':', 'SMC': '-.'}
COURSE_DISPLAY_NAMES = {
    'straight': 'Straight', 's_curve': 'S-curve', 'sharp_turn': 'Sharp turn',
    'low_speed': 'Low speed', 'combined': 'Combined',
}

# === Phase 1: Reference Trajectory Generation ===

def smooth_yaw(yaw_arr):
    """
    Unwrap yaw angles to remove discontinuities at +/- pi.

    :param yaw_arr: (ndarray) Array of yaw angles [rad]
    :return: (ndarray) Unwrapped yaw angles [rad]
    """
    return np.unwrap(yaw_arr, discont=np.pi)

def generate_reference_course(course_type, ds=0.1):
    """
    Generate a reference trajectory for a given course type.

    :param course_type: (str) One of 'straight', 's_curve', 'sharp_turn', 'low_speed', 'combined'
    :param ds: (float) Path discretization step [m]
    :return: (dict) Reference dict with keys 'x_ref', 'y_ref', 'theta_ref', 'kappa_ref', 'v_ref', 's_arr', 'v_limit_full'
    """
    if course_type == "straight":
        x_wp = np.arange(0, 20.01, 1.0)
        y_wp = np.zeros_like(x_wp)
        v_base = 2.0
    elif course_type == "s_curve":
        x_wp = np.arange(0, 30.01, 1.0)
        y_wp = np.sin(x_wp / 3.0) * 3.0
        v_base = 1.5
    elif course_type == "sharp_turn":
        x_wp_straight = np.arange(0, 10.01, 0.5)
        y_wp_straight = np.zeros_like(x_wp_straight)
        r_turn = 2.0
        angles = np.linspace(0, np.pi / 2, 20)
        x_wp_turn = x_wp_straight[-1] + r_turn * np.sin(angles)
        y_wp_turn = r_turn * (1 - np.cos(angles))
        x_wp = np.concatenate([x_wp_straight, x_wp_turn[1:]])
        y_wp = np.concatenate([y_wp_straight, y_wp_turn[1:]])
        v_base = 1.0
    elif course_type == "low_speed":
        x_wp = np.arange(0, 5.01, 0.5)
        y_wp = np.zeros_like(x_wp)
        v_base = 0.2
    elif course_type == "combined":
        x_s1 = np.arange(0, 8.01, 0.5)
        y_s1 = np.zeros_like(x_s1)
        x_sc = np.arange(8, 22.01, 0.5)
        y_sc = np.sin((x_sc - 8) / 3.0) * 2.5
        r_c = 2.0
        angles_c = np.linspace(0, np.pi / 2, 15)
        x_turn = x_sc[-1] + r_c * np.sin(angles_c)
        y_turn = y_sc[-1] + r_c * (1 - np.cos(angles_c))
        x_wp = np.concatenate([x_s1, x_sc[1:], x_turn[1:]])
        y_wp = np.concatenate([y_s1, y_sc[1:], y_turn[1:]])
        v_base = 1.5
    else:
        assert False, f"unknown course_type: {course_type}"

    dx_wp = np.diff(x_wp)
    dy_wp = np.diff(y_wp)
    ds_wp = np.sqrt(dx_wp**2 + dy_wp**2)
    s_wp = np.concatenate([[0], np.cumsum(ds_wp)])
    s_dense = np.arange(0, s_wp[-1], ds)
    x_ref = np.interp(s_dense, s_wp, x_wp)
    y_ref = np.interp(s_dense, s_wp, y_wp)

    dx_ref = np.gradient(x_ref, s_dense)
    dy_ref = np.gradient(y_ref, s_dense)
    theta_ref = np.arctan2(dy_ref, dx_ref)
    theta_ref = smooth_yaw(theta_ref)

    n_pts = len(x_ref)
    kappa_ref = compute_curvature(x_ref, y_ref)

    s_arr = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_ref)**2 + np.diff(y_ref)**2))])
    v_limit_full = curvature_speed_limit(kappa_ref, A_LAT_MAX, V_MAX)

    if course_type == "combined":
        v_ref = np.where(np.abs(kappa_ref) < 0.05, 2.0,
                np.where(np.abs(kappa_ref) < 0.2, 1.5, 1.0))
        v_ref = np.minimum(v_ref, v_limit_full)
    elif course_type == "sharp_turn":
        v_ref = np.where(np.abs(kappa_ref) < 0.05, 2.0, v_base)
        v_ref = np.minimum(v_ref, v_limit_full)
    else:
        v_ref = np.minimum(v_limit_full, v_base * np.ones(n_pts))

    return {'x_ref': x_ref, 'y_ref': y_ref, 'theta_ref': theta_ref,
            'kappa_ref': kappa_ref, 'v_ref': v_ref, 's_arr': s_arr,
            'v_limit_full': v_limit_full}

# === Phase 2: Vehicle Simulation Model ===

def bicycle_sim(state, delta, accel, dt=0.005, wheelbase=0.3):
    """
    Bicycle model simulation step.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param delta: (float) Steering angle [rad]
    :param accel: (float) Acceleration [m/s^2]
    :param dt: (float) Time step [s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :return: (ndarray) Updated state [x, y, theta, v]
    """
    x, y, theta, v = state
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(delta) / (wheelbase + 1e-18) * dt
    v_new = np.clip(v + accel * dt, 0.0, V_MAX)
    return np.array([x_new, y_new, theta_new, v_new])

# === Phase 3: Controller Implementation ===

def controller_lqr_stanley(state, ref_dict, ctrl_state, dt=0.005):
    """
    LQR-Stanley hybrid controller with adaptive gain and degradation fallback.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param ref_dict: (dict) Reference trajectory dictionary
    :param ctrl_state: (dict) Controller internal state
    :param dt: (float) Time step [s]
    :return: (tuple) (delta_clipped, accel_clipped, ctrl_state_new)
    """
    x, y, theta, v = state
    x_ref = ref_dict['x_ref']
    y_ref = ref_dict['y_ref']
    theta_ref = ref_dict['theta_ref']
    kappa_ref = ref_dict['kappa_ref']
    v_ref = ref_dict['v_ref']
    s_arr = ref_dict['s_arr']

    idx_prev = ctrl_state['idx_prev']
    e_lat_prev = ctrl_state['e_lat_prev']
    e_theta_prev = ctrl_state['e_theta_prev']
    de_lat_prev = ctrl_state.get('de_lat_prev', 0.0)
    de_theta_prev = ctrl_state.get('de_theta_prev', 0.0)
    integral_e_v = ctrl_state['integral_e_v']
    e_v_prev = ctrl_state['e_v_prev']
    delta_prev = ctrl_state['delta_prev']
    n_sat_count = ctrl_state['n_sat_count']
    gains_straight = ctrl_state['gains_straight']
    gains_curve = ctrl_state['gains_curve']

    idx = find_nearest_point(x, y, x_ref, y_ref, idx_prev)
    idx = min(idx, len(x_ref) - 1)

    e_lat = compute_lateral_error(x, y, theta_ref[idx], x_ref[idx], y_ref[idx])
    e_theta = compute_heading_error(theta, theta_ref[idx])

    kappa_now = kappa_ref[idx]
    t_err_preview = T_ERR_BASE + T_ERR_KAPPA * np.abs(kappa_now)
    w_err_preview = W_ERR_BASE + W_ERR_KAPPA * np.abs(kappa_now)

    s_target = s_arr[idx] + v * t_err_preview
    idx_la = min(np.searchsorted(s_arr, s_target), len(x_ref) - 1)
    e_lat_la = compute_lateral_error(x, y, theta_ref[idx_la], x_ref[idx_la], y_ref[idx_la])
    e_theta_la = compute_heading_error(theta, theta_ref[idx_la])

    e_lat_blend = (1 - w_err_preview) * e_lat + w_err_preview * e_lat_la
    e_theta_blend = (1 - w_err_preview) * e_theta + w_err_preview * e_theta_la

    e, _, _, de_lat, de_theta = compute_error_state(
        x, y, theta, v, x_ref[idx], y_ref[idx], theta_ref[idx], v_ref[idx],
        e_lat_prev, e_theta_prev, de_lat_prev, de_theta_prev, dt=DT, alpha_f=ALPHA_F)
    e[0] = e_lat_blend
    e[2] = e_theta_blend

    v_limit_full = ref_dict['v_limit_full']
    s_from_idx = s_arr - s_arr[idx]
    v_pre = lookahead_speed_limit(v_limit_full, v, s_from_idx, A_MAX, T_REACT)
    v_safe = dynamic_safety_margin(v_pre, e_lat, E_LAT_TH, BETA_SAFE)
    v_target = min(v_safe, v_ref[idx])

    kappa_la = lookahead_curvature(kappa_ref, idx, v, s_arr, T_LA_FF, L_LA_MIN)
    delta_ff, a_ff = feedforward_compensate(kappa_la, v_ref[idx], v, TAU_FF, WHEELBASE)
    delta_lqr, accel_lqr = lqr_control_adaptive(e, v, kappa_la, gains_straight, gains_curve, DV_TABLE, delta_ff, a_ff)

    should_degrade, n_sat_count_new = degradation_check(v, delta_lqr, DELTA_MAX, n_sat_count, e_lat=e_lat, de_lat=de_lat)

    if should_degrade:
        if v < V_SW:
            delta_fallback = pure_pursuit_steer(x, y, theta, v, x_ref, y_ref, idx, K_PP_LOW, LFC_LOW)
        else:
            delta_fallback = stanley_control(e_theta, e_lat, v, K_STANLEY, V_MIN_STANLEY, WHEELBASE,
                                             theta, x, y, x_ref[idx], y_ref[idx], theta_ref[idx])
        delta_cmd = progressive_blend(delta_lqr, delta_fallback, v, kappa=kappa_la, k_sw=K_SW, v_sw=V_SW, kappa_sw=KAPPA_SW)
    else:
        delta_cmd = delta_lqr

    accel_cmd, integral_e_v_new, e_v_new = pid_speed_control(
        v_target, v, e_v_prev, integral_e_v, DT, TAU_FF, KP_V, KI_V, KD_V, INTEGRAL_LIMIT)

    delta_clipped, accel_clipped = clip_control(delta_cmd, accel_cmd, delta_prev, DT, DELTA_MAX, A_MAX, DELTA_DOT_MAX)

    ctrl_state_new = {
        'idx_prev': idx, 'e_lat_prev': e_lat, 'e_theta_prev': e_theta,
        'de_lat_prev': de_lat, 'de_theta_prev': de_theta,
        'integral_e_v': integral_e_v_new, 'e_v_prev': e_v_new,
        'delta_prev': delta_clipped, 'n_sat_count': n_sat_count_new,
        'gains_straight': gains_straight, 'gains_curve': gains_curve,
        'n_degrade': int(should_degrade), 'v_target': v_target,
    }
    return delta_clipped, accel_clipped, ctrl_state_new



def controller_stanley_only(state, ref_dict, ctrl_state, dt=0.005):
    """
    Stanley-only lateral controller with PI speed control.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param ref_dict: (dict) Reference trajectory dictionary
    :param ctrl_state: (dict) Controller internal state
    :param dt: (float) Time step [s]
    :return: (tuple) (delta_clipped, accel_clipped, ctrl_state_new)
    """
    x, y, theta, v = state
    x_ref = ref_dict['x_ref']
    y_ref = ref_dict['y_ref']
    theta_ref = ref_dict['theta_ref']
    v_ref = ref_dict['v_ref']

    idx_prev = ctrl_state['idx_prev']
    integral_e_v = ctrl_state['integral_e_v']
    delta_prev = ctrl_state['delta_prev']

    idx = find_nearest_point(x, y, x_ref, y_ref, idx_prev)
    idx = min(idx, len(x_ref) - 1)

    e_theta = compute_heading_error(theta, theta_ref[idx])
    e_lat = compute_lateral_error(x, y, theta_ref[idx], x_ref[idx], y_ref[idx])

    delta = stanley_control(e_theta, e_lat, v, K_STANLEY, V_MIN_STANLEY, WHEELBASE,
                            theta, x, y, x_ref[idx], y_ref[idx], theta_ref[idx])

    e_v = v_ref[idx] - v
    accel, integral_e_v_new = stanley_speed_control(e_v, integral_e_v, DT, KP_V, KI_V, INTEGRAL_LIMIT)

    delta_clipped, accel_clipped = clip_control(delta, accel, delta_prev, DT, DELTA_MAX, A_MAX, DELTA_DOT_MAX)

    ctrl_state_new = {'idx_prev': idx, 'integral_e_v': integral_e_v_new, 'delta_prev': delta_clipped}
    return delta_clipped, accel_clipped, ctrl_state_new


# === Phase 4: Simulation Run ===

def run_simulation(controller_fn, ref_dict, dt=0.005, max_time=30.0,
                   x0=0.0, y0=0.2, theta0=0.0, v0=0.0, ctrl_type='default',
                   force_recompute_gains=False):
    """
    Run closed-loop simulation with a given controller.

    :param controller_fn: (callable) Controller function (state, ref_dict, ctrl_state, dt) -> (delta, accel, ctrl_state)
    :param ref_dict: (dict) Reference trajectory dictionary
    :param dt: (float) Simulation time step [s]
    :param max_time: (float) Maximum simulation time [s]
    :param x0: (float) Initial x position [m]
    :param y0: (float) Initial y position [m]
    :param theta0: (float) Initial heading [rad]
    :param v0: (float) Initial speed [m/s]
    :param ctrl_type: (str) Controller type identifier
    :param force_recompute_gains: (bool) Force recomputation of LQR gains
    :return: (dict) Simulation results with time histories and metrics
    """
    n_ref = len(ref_dict['x_ref'])
    n_max = int(max_time / dt) + 1

    t_list, x_list, y_list, theta_list, v_list = [], [], [], [], []
    delta_list, accel_list = [], []
    e_lat_list, e_theta_list, e_v_list = [], [], []
    step_times = []
    n_degrade_total = 0

    state = np.array([x0, y0, theta0, v0])

    if ctrl_type == 'lqr_stanley':
        gains_path = pathlib.Path(__file__).parent / 'lqr_gains_adaptive.npz'
        if force_recompute_gains and gains_path.exists():
            gains_path.unlink()
        if gains_path.exists():
            gains_straight, gains_curve = load_adaptive_gains(str(gains_path))
        else:
            gains_straight, gains_curve = precompute_adaptive_gains()
            save_adaptive_gains(gains_straight, gains_curve, str(gains_path))
        ctrl_state = {
            'idx_prev': 0, 'e_lat_prev': 0.0, 'e_theta_prev': 0.0,
            'de_lat_prev': 0.0, 'de_theta_prev': 0.0,
            'integral_e_v': 0.0, 'e_v_prev': 0.0, 'delta_prev': 0.0,
            'n_sat_count': 0, 'gains_straight': gains_straight, 'gains_curve': gains_curve,
            'n_degrade': 0,
        }
    elif ctrl_type == 'pure_pursuit':
        ctrl_state = {'idx_prev': 0, 'delta_prev': 0.0}
    elif ctrl_type == 'stanley_only':
        ctrl_state = {'idx_prev': 0, 'integral_e_v': 0.0, 'delta_prev': 0.0}
    elif ctrl_type == 'smc':
        ctrl_state = {'idx_prev': 0, 'e_lat_prev': 0.0, 'de_lat_prev': 0.0, 'integral_e_v': 0.0, 'delta_prev': 0.0}
    else:
        ctrl_state = {}

    for step in range(n_max):
        t_now = step * dt
        idx_now = ctrl_state.get('idx_prev', 0)
        n_margin = max(3, int(n_ref * 0.01))
        if idx_now >= n_ref - n_margin:
            break

        t_start = time.perf_counter()
        delta, accel, ctrl_state = controller_fn(state, ref_dict, ctrl_state, dt)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        step_times.append(t_elapsed)

        n_degrade_total += ctrl_state.get('n_degrade', 0)

        x_ref_now = ref_dict['x_ref'][idx_now]
        y_ref_now = ref_dict['y_ref'][idx_now]
        theta_ref_now = ref_dict['theta_ref'][idx_now]
        v_ref_now = ref_dict['v_ref'][idx_now]

        e_lat = compute_lateral_error(state[0], state[1], theta_ref_now, x_ref_now, y_ref_now)
        e_theta = compute_heading_error(state[2], theta_ref_now)
        v_eval = ctrl_state.get('v_target', v_ref_now)
        e_v = state[3] - v_eval

        t_list.append(t_now)
        x_list.append(state[0])
        y_list.append(state[1])
        theta_list.append(state[2])
        v_list.append(state[3])
        delta_list.append(delta)
        accel_list.append(accel)
        e_lat_list.append(e_lat)
        e_theta_list.append(e_theta)
        e_v_list.append(e_v)

        state = bicycle_sim(state, delta, accel, dt, WHEELBASE)

    return {
        't_arr': np.array(t_list), 'x_arr': np.array(x_list), 'y_arr': np.array(y_list),
        'theta_arr': np.array(theta_list), 'v_arr': np.array(v_list),
        'delta_arr': np.array(delta_list), 'accel_arr': np.array(accel_list),
        'e_lat_arr': np.array(e_lat_list), 'e_theta_arr': np.array(e_theta_list),
        'e_v_arr': np.array(e_v_list), 'n_degrade': n_degrade_total,
        'step_times': np.array(step_times),
    }

# === Phase 5: Evaluation Metrics ===

def compute_metrics(sim_result):
    """
    Compute tracking performance metrics from simulation results.

    :param sim_result: (dict) Simulation output from run_simulation
    :return: (dict) Metrics dict with keys 'rmse_lat', 'rmse_theta', 'rmse_v', 'max_lat', 'smoothness', 'step_time_ms', 'n_degrade'
    """
    e_lat = sim_result['e_lat_arr']
    e_theta = sim_result['e_theta_arr']
    e_v = sim_result['e_v_arr']
    delta = sim_result['delta_arr']
    step_times = sim_result['step_times']

    rmse_lat = np.sqrt(np.mean(e_lat**2))
    rmse_theta = np.sqrt(np.mean(e_theta**2))
    rmse_v = np.sqrt(np.mean(e_v**2))
    max_lat = np.max(np.abs(e_lat))
    mean_step_time = np.mean(step_times)
    smoothness = np.mean(np.diff(delta)**2) if len(delta) > 1 else 0.0

    return {
        'rmse_lat': rmse_lat, 'rmse_theta': rmse_theta, 'rmse_v': rmse_v,
        'max_lat': max_lat, 'smoothness': smoothness,
        'step_time_ms': mean_step_time, 'n_degrade': sim_result['n_degrade'],
    }

# === Phase 6: Visualization ===

MM_TO_INCH = 1.0 / 25.4

def fig1_trajectory_combined(ref_dict, results_dict, metrics_dict, figs_dir):
    """
    Plot combined trajectory comparison with curvature peak inset.

    :param ref_dict: (dict) Reference trajectory
    :param results_dict: (dict) Algorithm name to simulation results
    :param metrics_dict: (dict) Algorithm name to metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 183 * MM_TO_INCH, 80 * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(ref_dict['x_ref'], ref_dict['y_ref'], 'k--', linewidth=0.8, label='Reference', zorder=1)
    for name in ALGO_ORDER:
        res = results_dict[name]
        rmse = metrics_dict[name]['rmse_lat']
        ax.plot(res['x_arr'], res['y_arr'], color=COLORS[name], linestyle=LINESTYLE[name],
                linewidth=0.9, label=f'{name} ({rmse:.4f})', zorder=2)
    ax.plot(results_dict[ALGO_ORDER[0]]['x_arr'][0], results_dict[ALGO_ORDER[0]]['y_arr'][0],
            'k^', markersize=4, zorder=3)
    x_s_curve = ref_dict['x_ref']
    y_s_curve = ref_dict['y_ref']
    kappa = ref_dict['kappa_ref']
    peak_idx = np.argmax(np.abs(kappa))
    cx, cy = x_s_curve[peak_idx], y_s_curve[peak_idx]
    ax_in = inset_axes(ax, width="35%", height="35%", loc='upper right')
    margin = 1.5
    mask = (x_s_curve > cx - margin) & (x_s_curve < cx + margin)
    ax_in.plot(x_s_curve[mask], y_s_curve[mask], 'k--', linewidth=0.6)
    for name in ALGO_ORDER:
        res = results_dict[name]
        r_mask = (res['x_arr'] > cx - margin) & (res['x_arr'] < cx + margin)
        ax_in.plot(res['x_arr'][r_mask], res['y_arr'][r_mask], color=COLORS[name],
                   linestyle=LINESTYLE[name], linewidth=0.7)
    ax_in.set_xlim(cx - margin, cx + margin)
    y_range = y_s_curve[mask]
    y_center = 0.5 * (y_range.min() + y_range.max())
    ax_in.set_ylim(y_center - margin, y_center + margin)
    ax_in.tick_params(labelsize=5)
    ax_in.set_title('Curvature peak', fontsize=5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(fontsize=5, loc='lower right')
    ax.set_aspect('equal')
    fig.savefig(figs_dir / 'fig1_trajectory_combined.png', bbox_inches='tight')
    plt.close(fig)


def fig2_lateral_error_combined(results_dict, metrics_dict, figs_dir):
    """
    Plot lateral error time series for all algorithms.

    :param results_dict: (dict) Algorithm name to simulation results
    :param metrics_dict: (dict) Algorithm name to metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 89 * MM_TO_INCH, 55 * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, h))
    for name in ALGO_ORDER:
        res = results_dict[name]
        rmse = metrics_dict[name]['rmse_lat']
        ax.plot(res['t_arr'], res['e_lat_arr'], color=COLORS[name], linestyle=LINESTYLE[name],
                linewidth=0.6, label=f'{name} ({rmse:.4f})')
    ax.axhline(0, color='k', linewidth=0.4)
    ax.axhline(0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax.axhline(-0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$e_{\mathrm{lat}}$ (m)')
    ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig2_lateral_error_combined.png')
    plt.close(fig)


def fig3_rmse_bar_comparison(all_metrics, figs_dir):
    """
    Plot RMSE bar chart comparison across course types.

    :param all_metrics: (dict) Course type to algorithm metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 183 * MM_TO_INCH, 60 * MM_TO_INCH
    course_types = list(all_metrics.keys())
    n_courses = len(course_types)
    n_algos = len(ALGO_ORDER)
    x_pos = np.arange(n_courses)
    bar_w = 0.8 / n_algos
    fig, ax = plt.subplots(figsize=(w, h))
    for a_idx, name in enumerate(ALGO_ORDER):
        vals = [all_metrics[ct][name]['rmse_lat'] for ct in course_types]
        offset = (a_idx - n_algos / 2 + 0.5) * bar_w
        bars = ax.bar(x_pos + offset, vals, bar_w, color=COLORS[name], label=name, edgecolor='white', linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=4.5, rotation=45)
    ax.set_xticks(x_pos)
    ct_labels = [COURSE_DISPLAY_NAMES.get(ct, ct) for ct in course_types]
    ax.set_xticklabels(ct_labels, fontsize=6)
    ax.set_ylabel(r'RMSE$_{\mathrm{lat}}$ (m)')
    ax.legend(fontsize=5, ncol=n_algos, loc='upper left')
    ax.grid(axis='y', alpha=0.15)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig3_rmse_bar_comparison.png')
    plt.close(fig)


def fig4_control_input_combined(results_dict, metrics_dict, figs_dir):
    """
    Plot steering angle time series for all algorithms.

    :param results_dict: (dict) Algorithm name to simulation results
    :param metrics_dict: (dict) Algorithm name to metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 89 * MM_TO_INCH, 55 * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, h))
    for name in ALGO_ORDER:
        res = results_dict[name]
        smooth = metrics_dict[name]['smoothness']
        ax.plot(res['t_arr'], res['delta_arr'], color=COLORS[name], linestyle=LINESTYLE[name],
                linewidth=0.5, label=f'{name} ({smooth:.1e})')
    ax.axhline(DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax.axhline(-DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\delta$ (rad)')
    ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig4_control_input_combined.png')
    plt.close(fig)


def fig5_comprehensive_evaluation(all_metrics, figs_dir):
    """
    Plot comprehensive evaluation heatmap and weighted score ranking.

    :param all_metrics: (dict) Course type to algorithm metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 183 * MM_TO_INCH, 70 * MM_TO_INCH
    metric_keys = ['rmse_lat', 'max_lat', 'smoothness', 'step_time_ms', 'rmse_v', 'rmse_theta']
    metric_labels = ['RMSE_lat\n(m)', 'max_lat\n(m)', 'Smoothness\n(rad²)', 'Step time\n(ms)', 'RMSE_v\n(m/s)', 'RMSE_θ\n(rad)']
    course_types = list(all_metrics.keys())

    raw = np.zeros((len(ALGO_ORDER), len(course_types), len(metric_keys)))
    for a_idx, name in enumerate(ALGO_ORDER):
        for c_idx, ct in enumerate(course_types):
            for m_idx, k in enumerate(metric_keys):
                raw[a_idx, c_idx, m_idx] = all_metrics[ct][name][k]

    normed = np.zeros_like(raw)
    for m_idx, k in enumerate(metric_keys):
        if k == 'step_time_ms':
            normed[:, :, m_idx] = np.clip(raw[:, :, m_idx] / 1.0, 0.0, 1.0)
        else:
            r_min = raw[:, :, m_idx].min()
            r_max = raw[:, :, m_idx].max()
            normed[:, :, m_idx] = (raw[:, :, m_idx] - r_min) / (r_max - r_min + 1e-12)

    fig, (ax_heat, ax_score) = plt.subplots(1, 2, figsize=(w, h),
                                             gridspec_kw={'width_ratios': [3, 1.2], 'wspace': 0.35})

    heat_data = normed.mean(axis=2)
    im = ax_heat.imshow(heat_data.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(ALGO_ORDER)))
    ax_heat.set_xticklabels(ALGO_ORDER, fontsize=6, rotation=15, ha='right')
    ax_heat.set_yticks(range(len(course_types)))
    ax_heat.set_yticklabels([COURSE_DISPLAY_NAMES.get(ct, ct) for ct in course_types], fontsize=6)
    for a_idx in range(len(ALGO_ORDER)):
        for c_idx in range(len(course_types)):
            val = heat_data[a_idx, c_idx]
            txt_color = 'white' if val > 0.65 else 'black'
            ax_heat.text(a_idx, c_idx, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color=txt_color)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cbar.set_label('Normalized score (0=best, 1=worst)', fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)
    ax_heat.set_title('Multi-metric performance heatmap', fontsize=7, fontweight='bold', pad=6)

    scores = {}
    for a_idx, name in enumerate(ALGO_ORDER):
        s = 0.0
        for c_idx, ct in enumerate(course_types):
            for m_idx in range(len(metric_keys)):
                s += METRIC_WEIGHTS[m_idx] * normed[a_idx, c_idx, m_idx]
        scores[name] = s / len(course_types)

    sorted_names = sorted(ALGO_ORDER, key=lambda n: scores[n])
    sorted_scores = [scores[n] for n in sorted_names]
    bar_colors = [COLORS[n] for n in sorted_names]

    bars = ax_score.barh(range(len(sorted_names)), sorted_scores, color=bar_colors,
                         edgecolor='white', linewidth=0.3, height=0.6)
    for i, (bar, sc) in enumerate(zip(bars, sorted_scores)):
        ax_score.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                      f'{sc:.3f}', va='center', fontsize=5.5)
    ax_score.set_yticks(range(len(sorted_names)))
    ax_score.set_yticklabels(sorted_names, fontsize=6)
    ax_score.set_xlabel('Weighted score (lower=better)', fontsize=6)
    ax_score.set_title('Comprehensive\nranking', fontsize=7, fontweight='bold', pad=6)
    ax_score.set_xlim(0, max(sorted_scores) * 1.25)
    ax_score.invert_yaxis()

    fig.savefig(figs_dir / 'fig5_comprehensive_evaluation.png', bbox_inches='tight')
    plt.close(fig)


def fig6_computational_efficiency(all_metrics, figs_dir):
    """
    Plot computational efficiency bar chart for all algorithms.

    :param all_metrics: (dict) Course type to algorithm metrics
    :param figs_dir: (Path) Output directory for figures
    """
    w, h = 89 * MM_TO_INCH, 50 * MM_TO_INCH
    course_types = list(all_metrics.keys())
    avg_times = {}
    for name in ALGO_ORDER:
        avg_times[name] = np.mean([all_metrics[ct][name]['step_time_ms'] for ct in course_types])
    fig, ax = plt.subplots(figsize=(w, h))
    vals = [avg_times[name] for name in ALGO_ORDER]
    bars = ax.bar(ALGO_ORDER, vals, color=[COLORS[name] for name in ALGO_ORDER], edgecolor='white', linewidth=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    ax.axhline(1.0, color='#666666', linewidth=0.5, linestyle='--')
    ax.text(len(ALGO_ORDER) - 0.5, 1.05, '1 ms threshold', fontsize=5, color='#666666', ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Step time (ms)')
    ax.set_ylim(0.01, max(vals) * 3)
    ax.tick_params(axis='x', labelsize=6)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig6_computational_efficiency.png')
    plt.close(fig)

# === Phase 7: Main Function ===

def main():
    """
    Run full controller comparison across all course types and generate figures.
    """
    figs_dir = pathlib.Path(__file__).parent / 'figs'
    results_dir = pathlib.Path(__file__).parent / 'results'
    figs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    gains_path = pathlib.Path(__file__).parent / 'lqr_gains_adaptive.npz'
    if not gains_path.exists():
        gains_straight, gains_curve = precompute_adaptive_gains()
        save_adaptive_gains(gains_straight, gains_curve, str(gains_path))
    else:
        gains_straight, gains_curve = load_adaptive_gains(str(gains_path))

    course_types = ["straight", "s_curve", "sharp_turn", "low_speed", "combined"]

    controllers = {
        'LQR-Stanley': (controller_lqr_stanley, 'lqr_stanley'),
        'PurePursuit': (controller_pure_pursuit, 'pure_pursuit'),
        'Stanley-only': (controller_stanley_only, 'stanley_only'),
        'SMC': (controller_smc, 'smc'),
    }

    all_metrics = {}
    all_results = {}
    all_refs = {}

    for ct in course_types:
        print(f"\n{'='*50}")
        print(f"[{ct}] generating reference course")
        ref_dict = generate_reference_course(ct)
        all_refs[ct] = ref_dict

        results_dict = {}
        metrics_dict = {}

        for name, (ctrl_fn, ctrl_type) in controllers.items():
            print(f"  [{ct}] running {name} ...", end=' ')
            sim = run_simulation(ctrl_fn, ref_dict, max_time=30.0, y0=0.2, ctrl_type=ctrl_type)
            met = compute_metrics(sim)
            results_dict[name] = sim
            metrics_dict[name] = met
            print(f"rmse_lat={met['rmse_lat']:.4f}  max_lat={met['max_lat']:.4f}  "
                  f"smooth={met['smoothness']:.6f}  step={met['step_time_ms']:.2f}ms")

        all_results[ct] = results_dict
        all_metrics[ct] = metrics_dict

    fig1_trajectory_combined(all_refs['combined'], all_results['combined'], all_metrics['combined'], figs_dir)
    fig2_lateral_error_combined(all_results['combined'], all_metrics['combined'], figs_dir)
    fig3_rmse_bar_comparison(all_metrics, figs_dir)
    fig4_control_input_combined(all_results['combined'], all_metrics['combined'], figs_dir)
    fig5_comprehensive_evaluation(all_metrics, figs_dir)
    fig6_computational_efficiency(all_metrics, figs_dir)

    rows = []
    for ct in course_types:
        for name in controllers:
            m = all_metrics[ct][name]
            rows.append([ct, name, f"{m['rmse_lat']:.6f}", f"{m['rmse_theta']:.6f}",
                         f"{m['rmse_v']:.6f}", f"{m['max_lat']:.6f}",
                         f"{m['smoothness']:.8f}", f"{m['step_time_ms']:.3f}",
                         str(m['n_degrade'])])
    header = 'course,algorithm,rmse_lat,rmse_theta,rmse_v,max_lat,smoothness,step_time_ms,n_degrade'
    csv_lines = [header] + [','.join(r) for r in rows]
    with open(results_dir / 'metrics.csv', 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Course':<14} {'Algorithm':<14} {'RMSE_lat':>10} {'max_lat':>10} {'smooth':>12} {'step_ms':>10}")
    print('-' * 70)
    for ct in course_types:
        for name in controllers:
            m = all_metrics[ct][name]
            print(f"{ct:<14} {name:<14} {m['rmse_lat']:10.4f} {m['max_lat']:10.4f} "
                  f"{m['smoothness']:12.6f} {m['step_time_ms']:10.2f}")
        print('-' * 70)

    print(f"\n6 figures saved to {figs_dir}")
    print(f"Metrics saved to {results_dir / 'metrics.csv'}")


if __name__ == '__main__':
    main()
