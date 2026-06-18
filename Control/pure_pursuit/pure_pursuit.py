"""
Pure Pursuit steering controller for path tracking

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [Pure Pursuit Algorithm](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from Control.lqr.lqr_controller import find_nearest_point, WHEELBASE, V_MAX, DELTA_MAX, A_MAX, DELTA_DOT_MAX
from Control.config import K_PP_LOW, LFC_LOW
from Control.control_output import clip_control

# === Phase 1: Pure Pursuit Steering Computation ===

def pure_pursuit_steer(x, y, theta, v, x_ref, y_ref, idx, k_pp, Lfc, wheelbase=WHEELBASE):
    """
    Compute Pure Pursuit steering angle.

    :param x: (float) Current x position
    :param y: (float) Current y position
    :param theta: (float) Current heading angle [rad]
    :param v: (float) Current speed [m/s]
    :param x_ref: (ndarray) Reference x positions
    :param y_ref: (ndarray) Reference y positions
    :param idx: (int) Current nearest point index
    :param k_pp: (float) Pure Pursuit gain
    :param Lfc: (float) Minimum lookahead distance [m]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :return: (float) Steering angle [rad]
    """
    Lf = k_pp * v + Lfc
    dists = np.sqrt((x_ref[idx:] - x)**2 + (y_ref[idx:] - y)**2)
    la_mask = dists >= Lf
    if np.any(la_mask):
        la_idx = idx + np.argmax(la_mask)
    else:
        la_idx = len(x_ref) - 1
    la_idx = min(la_idx, len(x_ref) - 1)
    dx_la = x_ref[la_idx] - x
    dy_la = y_ref[la_idx] - y
    alpha = np.arctan2(dy_la, dx_la) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    return np.arctan2(2.0 * wheelbase * np.sin(alpha), Lf + 1e-12)

# === Phase 2: Pure Pursuit Full Controller ===

def controller_pure_pursuit(state, ref_dict, ctrl_state, dt=0.005, k_pp=K_PP_LOW, Lfc=LFC_LOW, Kp_pp=1.0):
    """
    Pure Pursuit controller with speed regulation.

    :param state: (ndarray) Vehicle state [x, y, theta, v]
    :param ref_dict: (dict) Reference trajectory with keys 'x_ref', 'y_ref', 'v_ref'
    :param ctrl_state: (dict) Controller internal state with keys 'idx_prev', 'delta_prev'
    :param dt: (float) Time step [s]
    :param k_pp: (float) Pure Pursuit gain
    :param Lfc: (float) Minimum lookahead distance [m]
    :param Kp_pp: (float) Proportional speed gain
    :return: (tuple) (delta_clipped, accel_clipped, ctrl_state_new)
    """
    x, y, theta, v = state
    x_ref = ref_dict['x_ref']
    y_ref = ref_dict['y_ref']
    v_ref = ref_dict['v_ref']

    idx_prev = ctrl_state['idx_prev']
    delta_prev = ctrl_state['delta_prev']

    idx = find_nearest_point(x, y, x_ref, y_ref, idx_prev)
    idx = min(idx, len(x_ref) - 1)

    delta = pure_pursuit_steer(x, y, theta, v, x_ref, y_ref, idx, k_pp, Lfc)
    accel = Kp_pp * (v_ref[idx] - v)

    delta_clipped, accel_clipped = clip_control(delta, accel, delta_prev, dt, DELTA_MAX, A_MAX, DELTA_DOT_MAX)

    ctrl_state_new = {'idx_prev': idx, 'delta_prev': delta_clipped}
    return delta_clipped, accel_clipped, ctrl_state_new

# === Phase 3: Bicycle Model Simulation ===

def bicycle_sim(state, delta, accel, dt=0.005, wheelbase=WHEELBASE):
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

# === Phase 4: Standalone Demo ===

def main():
    """
    Standalone demo: generate S-curve reference and run Pure Pursuit controller.
    """
    ds = 0.1
    x_wp = np.arange(0, 30.01, 1.0)
    y_wp = np.sin(x_wp / 3.0) * 3.0

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

    v_ref = 1.5 * np.ones(len(x_ref))

    ref_dict = {'x_ref': x_ref, 'y_ref': y_ref, 'theta_ref': theta_ref, 'v_ref': v_ref}

    dt = 0.005
    max_time = 30.0
    n_max = int(max_time / dt) + 1
    n_ref = len(x_ref)

    state = np.array([0.0, 0.2, 0.0, 0.0])
    ctrl_state = {'idx_prev': 0, 'delta_prev': 0.0}

    x_arr, y_arr, t_arr = [], [], []
    delta_arr, v_arr, e_lat_arr = [], [], []

    for step in range(n_max):
        idx_now = ctrl_state['idx_prev']
        n_margin = max(3, int(n_ref * 0.01))
        if idx_now >= n_ref - n_margin:
            break

        delta, accel, ctrl_state = controller_pure_pursuit(state, ref_dict, ctrl_state, dt)

        idx_now = ctrl_state['idx_prev']
        dx = state[0] - x_ref[idx_now]
        dy = state[1] - y_ref[idx_now]
        nx = -np.sin(theta_ref[idx_now])
        ny = np.cos(theta_ref[idx_now])
        e_lat = dx * nx + dy * ny

        t_arr.append(step * dt)
        x_arr.append(state[0])
        y_arr.append(state[1])
        delta_arr.append(delta)
        v_arr.append(state[3])
        e_lat_arr.append(e_lat)

        state = bicycle_sim(state, delta, accel, dt)

    t_arr = np.array(t_arr)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    delta_arr = np.array(delta_arr)
    v_arr = np.array(v_arr)
    e_lat_arr = np.array(e_lat_arr)

    rmse_lat = np.sqrt(np.mean(e_lat_arr**2))
    max_lat = np.max(np.abs(e_lat_arr))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    ax1.plot(x_ref, y_ref, 'k--', linewidth=1.0, label='Reference')
    ax1.plot(x_arr, y_arr, '#D95F02', linewidth=0.8, label='Pure Pursuit')
    ax1.plot(x_arr[0], y_arr[0], 'k^', markersize=6, label='Start')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title(f'Pure Pursuit Tracking  (RMSE_lat={rmse_lat:.4f} m, max_lat={max_lat:.4f} m)')
    ax1.legend(frameon=True, fancybox=True)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_arr, e_lat_arr, '#D95F02', linewidth=0.6)
    ax2.axhline(0, color='k', linewidth=0.4)
    ax2.axhline(0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax2.axhline(-0.1, color='#666666', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('e_lat [m]')
    ax2.set_title('Lateral Error')
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_arr, delta_arr, '#2166AC', linewidth=0.6)
    ax3.axhline(DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax3.axhline(-DELTA_MAX, color='#666666', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('delta [rad]')
    ax3.set_title('Steering Angle')
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    figs_dir = pathlib.Path(__file__).parent.parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    fig.savefig(figs_dir / 'pure_pursuit_demo.png', dpi=150)
    plt.show()

    print(f"[PurePursuit] rmse_lat={rmse_lat:.4f} m  max_lat={max_lat:.4f} m  "
          f"steps={len(t_arr)}  final_v={v_arr[-1]:.3f} m/s")


if __name__ == '__main__':
    main()
