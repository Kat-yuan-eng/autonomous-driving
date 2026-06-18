"""
LQR controller module
"""
from .lqr_controller import (solve_dare, build_state_matrix, adaptive_Q, adaptive_R,
    precompute_lqr_gains, precompute_adaptive_gains, save_gains, load_gains,
    save_adaptive_gains, load_adaptive_gains, find_nearest_point, compute_lateral_error,
    compute_heading_error, compute_error_state, interpolate_gain, interpolate_adaptive_gain,
    feedforward_compensate, lookahead_curvature, lqr_control_adaptive,
    WHEELBASE, V_MAX, DELTA_MAX, A_MAX, DELTA_DOT_MAX, DT, DV_TABLE)
