"""
Vehicle kinematic model utilities for bicycle model simulation

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np

WHEELBASE = 0.3


def bicycle_kinematics(state, v, delta, dt, wheelbase=WHEELBASE):
    """
    Compute next state using bicycle kinematic model.

    :param state: (numpy.ndarray) Current state [x, y, beta] where beta is heading angle.
    :param v: (float) Velocity.
    :param delta: (float) Steering angle in radians.
    :param dt: (float) Time step.
    :param wheelbase: (float) Wheelbase length. Default is WHEELBASE.
    :return: (numpy.ndarray) Next state [x_next, y_next, beta_next].
    """
    x, y, beta = state
    x_next = x + v * np.cos(beta) * dt
    y_next = y + v * np.sin(beta) * dt
    beta_next = beta + v * np.tan(delta) / wheelbase * dt
    return np.array([x_next, y_next, beta_next])


def compute_curvature(p_prev, p_curr, p_next):
    """
    Compute curvature from three consecutive points using the Menger curvature formula.

    :param p_prev: (numpy.ndarray) Previous point [x, y, ...].
    :param p_curr: (numpy.ndarray) Current point [x, y, ...].
    :param p_next: (numpy.ndarray) Next point [x, y, ...].
    :return: (float) Curvature kappa.
    """
    a = np.linalg.norm(p_curr[:2] - p_next[:2])
    b = np.linalg.norm(p_prev[:2] - p_next[:2])
    c = np.linalg.norm(p_prev[:2] - p_curr[:2])
    cross = abs(
        (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1])
        - (p_next[0] - p_prev[0]) * (p_curr[1] - p_prev[1])
    )
    area = 0.5 * cross
    kappa = 2.0 * area / max(a * b * c, 1e-12)
    return kappa


def compute_steering(kappa, wheelbase=WHEELBASE):
    """
    Compute steering angle from curvature using the bicycle model relation.

    :param kappa: (float) Curvature.
    :param wheelbase: (float) Wheelbase length. Default is WHEELBASE.
    :return: (float) Steering angle delta in radians.
    """
    return np.arctan(wheelbase * kappa)


if __name__ == '__main__':
    state = np.array([0.0, 0.0, 0.0])
    v, delta, dt = 1.0, np.deg2rad(10), 0.1
    for _ in range(50):
        state = bicycle_kinematics(state, v, delta, dt)
    print(f"[bicycle] final=({state[0]:.3f}, {state[1]:.3f}, {np.rad2deg(state[2]):.2f} deg)")

    p_prev = np.array([0.0, 0.0, 0.0])
    p_curr = np.array([1.0, 0.1, 0.1])
    p_next = np.array([2.0, 0.3, 0.1])
    kappa = compute_curvature(p_prev, p_curr, p_next)
    delta_steer = compute_steering(kappa)
    print(f"[curvature] kappa={kappa:.5f} 1/m, delta={np.rad2deg(delta_steer):.2f} deg")
