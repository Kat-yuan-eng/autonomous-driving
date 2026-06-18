"""
Longitudinal speed controller with curvature limiting and PID control

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np
from Control.config import (A_LAT_MAX, V_MAX, A_MAX, TAU_FF, KP_V, KI_V, KD_V,
    E_LAT_TH, BETA_SAFE, T_REACT, INTEGRAL_LIMIT, ALPHA_F)

# === Phase 5: Longitudinal Speed Control ===

def curvature_speed_limit(kappa_arr, a_lat_max=1.5, v_max=2.5):
    """
    Compute curvature-limited speed profile from lateral acceleration constraint.

    :param kappa_arr: (ndarray) Curvature array [1/m]
    :param a_lat_max: (float) Maximum allowable lateral acceleration [m/s^2]
    :param v_max: (float) Maximum speed [m/s]
    :return: (ndarray) Speed limit array [m/s]
    """
    v_limit_arr = np.minimum(v_max, np.sqrt(a_lat_max / (np.abs(kappa_arr) + 1e-9)))
    return v_limit_arr

def lookahead_speed_limit(v_limit_arr, v, s_arr, a_max=2.0, t_react=0.2):
    """
    Apply lookahead braking constraint based on stopping distance.

    :param v_limit_arr: (ndarray) Curvature-limited speed profile [m/s]
    :param v: (float) Current speed [m/s]
    :param s_arr: (ndarray) Cumulative arc-length array [m]
    :param a_max: (float) Maximum deceleration [m/s^2]
    :param t_react: (float) Reaction time [s]
    :return: (float) Lookahead-constrained speed [m/s]
    """
    assert len(v_limit_arr) == len(s_arr), "v_limit_arr and s_arr must have same length"
    v = max(v, 0.0)
    d_brake = v**2 / (2 * a_max) + v * t_react
    mask = s_arr <= d_brake
    if not np.any(mask):
        v_pre = v_limit_arr[-1]
    else:
        v_pre = np.min(v_limit_arr[mask])
    return v_pre

def dynamic_safety_margin(v_pre, e_lat, e_lat_th=0.1, beta_safe=3.0):
    """
    Apply dynamic safety margin reducing speed when lateral error exceeds threshold.

    :param v_pre: (float) Preview speed limit [m/s]
    :param e_lat: (float) Lateral error [m]
    :param e_lat_th: (float) Lateral error threshold [m]
    :param beta_safe: (float) Exponential decay rate
    :return: (float) Safety-adjusted speed [m/s]
    """
    v_safe = v_pre * np.exp(-beta_safe * np.maximum(0, np.abs(e_lat) - e_lat_th))
    return v_safe

def pid_speed_control(v_target, v, e_v_prev, integral_e_v, dt=0.005, tau_ff=0.5, K_p=2.0, K_i=0.1, K_d=0.3, integral_limit=1.0):
    """
    PID longitudinal speed controller with feedforward term.

    :param v_target: (float) Target speed [m/s]
    :param v: (float) Current speed [m/s]
    :param e_v_prev: (float) Previous speed error [m/s]
    :param integral_e_v: (float) Accumulated integral of speed error [m]
    :param dt: (float) Time step [s]
    :param tau_ff: (float) Feedforward time constant [s]
    :param K_p: (float) Proportional gain
    :param K_i: (float) Integral gain
    :param K_d: (float) Derivative gain
    :param integral_limit: (float) Anti-windup integral clamp [m]
    :return: (tuple) (a_cmd, integral_e_v_new, e_v) acceleration [m/s^2], updated integral, current error
    """
    assert dt > 0, f"dt must be positive, got {dt}"
    assert tau_ff > 0, f"tau_ff must be positive, got {tau_ff}"
    e_v = v_target - v
    e_v_dot = (e_v - e_v_prev) / (dt + 1e-12)
    integral_e_v_new = np.clip(integral_e_v + e_v * dt, -integral_limit, integral_limit)
    a_ff = (v_target - v) / (tau_ff + 1e-12)
    a_cmd = a_ff + K_p * e_v + K_i * integral_e_v_new + K_d * e_v_dot
    return a_cmd, integral_e_v_new, e_v

def main():
    print("Speed Controller Demo")
    kappa_arr = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    v_limit = curvature_speed_limit(kappa_arr)
    print(f"Curvature: {kappa_arr}")
    print(f"Speed limit: {v_limit}")
    v_safe = dynamic_safety_margin(2.0, 0.05)
    print(f"Dynamic safety margin (v_pre=2.0, e_lat=0.05): {v_safe:.3f} m/s")
    accel, integral_new, e_v = pid_speed_control(1.5, 1.0, 0.5, 0.0)
    print(f"PID speed control (target=1.5, current=1.0): accel={accel:.3f} m/s²")

if __name__ == '__main__':
    main()
