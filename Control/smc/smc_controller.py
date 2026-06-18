"""
Sliding Mode Control (SMC) for lateral vehicle control

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - Sliding mode control for vehicle path tracking
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from Control.lqr.lqr_controller import (find_nearest_point, compute_lateral_error,
    compute_heading_error, WHEELBASE, DELTA_MAX, A_MAX, DELTA_DOT_MAX)
from Control.config import (LAM_SMC, ETA_SMC, PHI_SMC, ALPHA_F,
    KP_V, KI_V, INTEGRAL_LIMIT, V_MAX, DT)
from Control.control_output import clip_control


def controller_smc(state, ref_dict, ctrl_state, dt=0.005, lam_smc=LAM_SMC,
                   eta_smc=ETA_SMC, phi_smc=PHI_SMC, alpha_f=ALPHA_F):
    """
    Sliding Mode Controller for lateral and longitudinal vehicle control.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param ref_dict: (dict) Reference trajectory dictionary
    :param ctrl_state: (dict) Controller internal state
    :param dt: (float) Time step [s]
    :param lam_smc: (float) SMC sliding surface slope
    :param eta_smc: (float) SMC switching gain
    :param phi_smc: (float) SMC boundary layer thickness
    :param alpha_f: (float) Low-pass filter coefficient for derivative
    :return: (tuple) (delta_clipped, accel_clipped, ctrl_state_new)
    """
    x, y, theta, v = state
    x_ref = ref_dict['x_ref']
    y_ref = ref_dict['y_ref']
    theta_ref = ref_dict['theta_ref']
    v_ref = ref_dict['v_ref']

    idx_prev = ctrl_state['idx_prev']
    e_lat_prev = ctrl_state['e_lat_prev']
    de_lat_prev = ctrl_state.get('de_lat_prev', 0.0)
    integral_e_v = ctrl_state['integral_e_v']
    delta_prev = ctrl_state['delta_prev']

    idx = find_nearest_point(x, y, x_ref, y_ref, idx_prev)
    idx = min(idx, len(x_ref) - 1)

    e_lat = compute_lateral_error(x, y, theta_ref[idx], x_ref[idx], y_ref[idx])
    e_theta = compute_heading_error(theta, theta_ref[idx])
    e_v = v_ref[idx] - v
    de_lat_raw = (e_lat - e_lat_prev) / (dt + 1e-12)
    de_lat = alpha_f * de_lat_raw + (1 - alpha_f) * de_lat_prev

    s_smc = de_lat + lam_smc * e_lat
    v_safe = max(v, 0.1)
    delta_eq = -WHEELBASE / v_safe * (lam_smc * de_lat + v_safe * np.sin(e_theta))
    delta_sw = -eta_smc * np.clip(s_smc / phi_smc, -1.0, 1.0)
    delta = delta_eq + delta_sw + e_theta
    delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)

    integral_e_v_new = np.clip(integral_e_v + e_v * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
    accel = KP_V * e_v + KI_V * integral_e_v_new
    accel = np.clip(accel, -A_MAX, A_MAX)

    delta_clipped, accel_clipped = clip_control(delta, accel, delta_prev, dt,
                                                DELTA_MAX, A_MAX, DELTA_DOT_MAX)

    ctrl_state_new = {'idx_prev': idx, 'e_lat_prev': e_lat, 'de_lat_prev': de_lat,
                      'integral_e_v': integral_e_v_new, 'delta_prev': delta_clipped}
    return delta_clipped, accel_clipped, ctrl_state_new


def bicycle_sim(state, delta, accel, dt=0.005, wheelbase=0.3):
    """
    Bicycle model simulation step.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param delta: (float) Steering angle [rad]
    :param accel: (float) Longitudinal acceleration [m/s^2]
    :param dt: (float) Time step [s]
    :param wheelbase: (float) Wheelbase [m]
    :return: (ndarray) Updated state [x, y, theta, v]
    """
    x, y, theta, v = state
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(delta) / (wheelbase + 1e-18) * dt
    v_new = np.clip(v + accel * dt, 0.0, V_MAX)
    return np.array([x_new, y_new, theta_new, v_new])


def generate_s_curve_ref(ds=0.1):
    """
    Generate S-curve reference trajectory.

    :param ds: (float) Path discretization step [m]
    :return: (dict) Reference trajectory dictionary
    """
    x_wp = np.arange(0, 30.01, 1.0)
    y_wp = np.sin(x_wp / 3.0) * 3.0
    v_base = 1.5

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
    theta_ref = np.unwrap(theta_ref, discont=np.pi)

    n_pts = len(x_ref)
    v_ref = np.minimum(v_base * np.ones(n_pts), V_MAX)

    return {'x_ref': x_ref, 'y_ref': y_ref, 'theta_ref': theta_ref, 'v_ref': v_ref}


def main():
    """
    Standalone demo: run SMC controller on S-curve trajectory and plot results.
    """
    ref_dict = generate_s_curve_ref()

    dt = DT
    max_time = 30.0
    n_max = int(max_time / dt) + 1
    n_ref = len(ref_dict['x_ref'])

    state = np.array([0.0, 0.2, 0.0, 0.0])
    ctrl_state = {'idx_prev': 0, 'e_lat_prev': 0.0, 'de_lat_prev': 0.0,
                  'integral_e_v': 0.0, 'delta_prev': 0.0}

    t_list, x_list, y_list = [], [], []
    e_lat_list, delta_list = [], []

    for step in range(n_max):
        idx_now = ctrl_state['idx_prev']
        n_margin = max(3, int(n_ref * 0.01))
        if idx_now >= n_ref - n_margin:
            break

        delta, accel, ctrl_state = controller_smc(state, ref_dict, ctrl_state, dt)

        t_list.append(step * dt)
        x_list.append(state[0])
        y_list.append(state[1])
        e_lat_list.append(ctrl_state['e_lat_prev'])
        delta_list.append(delta)

        state = bicycle_sim(state, delta, accel, dt, WHEELBASE)

    t_arr = np.array(t_list)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    e_lat_arr = np.array(e_lat_list)
    delta_arr = np.array(delta_list)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.plot(ref_dict['x_ref'], ref_dict['y_ref'], 'k--', linewidth=0.8, label='Reference')
    ax1.plot(x_arr, y_arr, '#4DAF4A', linewidth=0.9, label='SMC')
    ax1.plot(x_arr[0], y_arr[0], 'k^', markersize=5)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('SMC Controller - S-curve Tracking')
    ax1.legend(fontsize=8, frameon=True, fancybox=True)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_arr, e_lat_arr, '#4DAF4A', linewidth=0.6)
    ax2.axhline(0, color='k', linewidth=0.4)
    ax2.axhline(0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax2.axhline(-0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel(r'$e_{\mathrm{lat}}$ (m)')
    ax2.set_title('Lateral Error')
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_arr, delta_arr, '#4DAF4A', linewidth=0.6)
    ax3.axhline(DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax3.axhline(-DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('t (s)')
    ax3.set_ylabel(r'$\delta$ (rad)')
    ax3.set_title('Steering Angle')
    ax3.grid(True, alpha=0.3)

    rmse_lat = np.sqrt(np.mean(e_lat_arr**2))
    print(f"[SMC] rmse_lat={rmse_lat:.4f} m")

    fig.tight_layout()
    fig_dir = pathlib.Path(__file__).parent.parent / 'figs'
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / 'smc_demo_s_curve.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
