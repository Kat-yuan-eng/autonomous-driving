"""
Control output clipping and command publishing

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np

# === Phase 1: Output and Constraints ===

def clip_control(delta, accel, delta_prev, dt=0.005, delta_max=0.5236, a_max=2.0, delta_dot_max=1.7453):
    """
    Clip steering and acceleration commands within physical limits.

    :param delta: (float) Raw steering angle [rad]
    :param accel: (float) Raw acceleration [m/s²]
    :param delta_prev: (float) Previous steering angle [rad]
    :param dt: (float) Time step [s]
    :param delta_max: (float) Maximum steering angle [rad]
    :param a_max: (float) Maximum acceleration [m/s²]
    :param delta_dot_max: (float) Maximum steering rate [rad/s]
    :return: (tuple) (clipped_delta, clipped_accel)
    """
    assert dt > 0, f"dt must be positive, got {dt}"
    assert delta_max > 0, f"delta_max must be positive, got {delta_max}"
    assert a_max > 0, f"a_max must be positive, got {a_max}"
    assert delta_dot_max > 0, f"delta_dot_max must be positive, got {delta_dot_max}"
    delta = np.clip(delta, -delta_max, delta_max)
    delta = np.clip(delta, delta_prev - delta_dot_max * dt, delta_prev + delta_dot_max * dt)
    accel = np.clip(accel, -a_max, a_max)
    return delta, accel

def publish_cmd(delta, accel):
    """
    Package control commands into a dictionary.

    :param delta: (float) Steering angle command [rad]
    :param accel: (float) Acceleration command [m/s²]
    :return: (dict) Command dictionary with 'delta_cmd' and 'a_cmd'
    """
    return {"delta_cmd": delta, "a_cmd": accel}
