"""
Stanley steering controller for lateral vehicle control with degradation fallback

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
"""

import numpy as np
from Control.config import (K_STANLEY, V_MIN_STANLEY, K_SW, V_SW, N_SAT_TRIGGER,
    E_LAT_DEGRADE, KAPPA_SW, DE_LAT_DEGRADE, K_KAPPA_SW, V_REF_STANLEY_GAIN,
    WHEELBASE, DELTA_MAX)

# === Phase 4: Stanley Degradation Control ===

def stanley_control(e_theta, e_lat, v, k_stanley=K_STANLEY, v_min=V_MIN_STANLEY, wheelbase=WHEELBASE, theta=0.0, x=0.0, y=0.0, x_ref=0.0, y_ref=0.0, theta_ref=0.0,
                    v_ref_gain=V_REF_STANLEY_GAIN):
    """
    Compute Stanley steering angle with front-axle lateral error.

    :param e_theta: (float) Heading error [rad]
    :param e_lat: (float) Lateral error at rear axle [m]
    :param v: (float) Current speed [m/s]
    :param k_stanley: (float) Stanley lateral gain
    :param v_min: (float) Minimum speed for normalization [m/s]
    :param wheelbase: (float) Vehicle wheelbase [m]
    :param theta: (float) Current heading [rad]
    :param x: (float) Current x position [m]
    :param y: (float) Current y position [m]
    :param x_ref: (float) Reference x position [m]
    :param y_ref: (float) Reference y position [m]
    :param theta_ref: (float) Reference heading [rad]
    :param v_ref_gain: (float) Reference speed gain for adaptive k
    :return: (float) Steering angle [rad]
    """
    assert v >= 0, f"v must be non-negative, got {v}"
    fx = x + wheelbase * np.cos(theta)
    fy = y + wheelbase * np.sin(theta)
    nx = -np.sin(theta_ref)
    ny = np.cos(theta_ref)
    e_fa = (fx - x_ref) * nx + (fy - y_ref) * ny
    k_eff = k_stanley * max(1.0, v_ref_gain / max(v, v_min))
    delta = e_theta + np.arctan(k_eff * e_fa / max(v, v_min))
    delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)
    return delta

def stanley_speed_control(e_v, integral_e_v, dt=0.005, K_p=2.0, K_i=0.1, integral_limit=1.0):
    """
    PI speed controller for Stanley fallback mode.

    :param e_v: (float) Speed error [m/s]
    :param integral_e_v: (float) Accumulated integral of speed error [m]
    :param dt: (float) Time step [s]
    :param K_p: (float) Proportional gain
    :param K_i: (float) Integral gain
    :param integral_limit: (float) Anti-windup integral clamp [m]
    :return: (tuple) (accel, integral_e_v_new) acceleration [m/s^2] and updated integral
    """
    assert dt > 0, f"dt must be positive, got {dt}"
    integral_e_v_new = np.clip(integral_e_v + e_v * dt, -integral_limit, integral_limit)
    a = K_p * e_v + K_i * integral_e_v_new
    return a, integral_e_v_new

def degradation_check(v, delta_cmd, delta_max, n_sat_count, e_lat=0.0, de_lat=0.0,
                      n_sat_trigger=N_SAT_TRIGGER,
                      v_sw=V_SW, e_lat_degrade=E_LAT_DEGRADE, de_lat_degrade=DE_LAT_DEGRADE):
    """
    Check if LQR should degrade to Stanley fallback.

    :param v: (float) Current speed [m/s]
    :param delta_cmd: (float) Commanded steering angle [rad]
    :param delta_max: (float) Maximum steering angle [rad]
    :param n_sat_count: (int) Consecutive steering saturation count
    :param e_lat: (float) Lateral error [m]
    :param de_lat: (float) Lateral error rate [m/s]
    :param n_sat_trigger: (int) Saturation count threshold for degradation
    :param v_sw: (float) Speed threshold for low-speed degradation [m/s]
    :param e_lat_degrade: (float) Lateral error degradation threshold [m]
    :param de_lat_degrade: (float) Lateral error rate degradation threshold [m/s]
    :return: (tuple) (should_degrade, n_sat_count_new)
    """
    assert delta_max > 0, f"delta_max must be positive, got {delta_max}"
    n_sat_count_new = n_sat_count + 1 if np.abs(delta_cmd) > 0.95 * delta_max else 0
    should_degrade = (v < v_sw) or (n_sat_count_new >= n_sat_trigger) or (np.abs(e_lat) > e_lat_degrade) or (np.abs(de_lat) > de_lat_degrade)
    return should_degrade, n_sat_count_new

def progressive_blend(delta_lqr, delta_stanley, v, kappa=0.0, k_sw=K_SW, v_sw=V_SW, kappa_sw=KAPPA_SW, k_kappa_sw=K_KAPPA_SW):
    """
    Blend LQR and Stanley outputs with sigmoid-based progressive weighting.

    :param delta_lqr: (float) LQR steering command [rad]
    :param delta_stanley: (float) Stanley steering command [rad]
    :param v: (float) Current speed [m/s]
    :param kappa: (float) Path curvature [1/m]
    :param k_sw: (float) Sigmoid steepness for speed blending
    :param v_sw: (float) Speed crossover point [m/s]
    :param kappa_sw: (float) Curvature crossover point [1/m]
    :param k_kappa_sw: (float) Sigmoid steepness for curvature blending
    :return: (float) Blended steering angle [rad]
    """
    w_lqr_v = 1.0 / (1.0 + np.exp(-k_sw * (v - v_sw)))
    w_lqr_k = 1.0 / (1.0 + np.exp(-k_kappa_sw * (kappa_sw - np.abs(kappa))))
    w_lqr = w_lqr_v * w_lqr_k
    if v < v_sw:
        w_lqr = min(w_lqr, 0.05)
    return w_lqr * delta_lqr + (1 - w_lqr) * delta_stanley

def main():
    print("Stanley Controller Demo")
    e_theta = 0.1
    e_lat = 0.05
    v = 1.0
    delta = stanley_control(e_theta, e_lat, v)
    print(f"e_theta={e_theta:.3f} rad, e_lat={e_lat:.3f} m, v={v:.1f} m/s -> delta={delta:.4f} rad")
    should_degrade, n_sat = degradation_check(v, delta, DELTA_MAX, 0)
    print(f"Degradation check: should_degrade={should_degrade}, n_sat={n_sat}")
    blended = progressive_blend(0.1, 0.2, v)
    print(f"Progressive blend (delta_lqr=0.1, delta_stanley=0.2, v={v}): {blended:.4f}")

if __name__ == '__main__':
    main()
