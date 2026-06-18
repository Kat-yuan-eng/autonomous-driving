"""
Kalman filter tracker with Hungarian data association

author: Kat-yuan-eng (RuiWen Liao)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import SIGMA_POS, SIGMA_VEL, SIGMA_OBS, D_ASSOC, N_CONFIRM, N_DELETE, CHI2_NEW, CHI2_STABLE, V_DYNAMIC_THRESH, N_DYNAMIC_EXTRA, DT

# === Phase 3: Kalman tracking ===

_next_id = 0

def reset_tracker_id():
    """
    Reset the global tracker ID counter to zero.

    :return: None
    """
    global _next_id
    _next_id = 0

def create_tracker(x, y, vx=0.0, vy=0.0):
    """
    Create a new tracker dict with constant-velocity Kalman state.

    :param x: (float) Initial x position in meters
    :param y: (float) Initial y position in meters
    :param vx: (float) Initial x velocity in m/s
    :param vy: (float) Initial y velocity in m/s
    :return: (dict) Tracker dictionary with state, covariance, and metadata
    """
    global _next_id
    trk = {
        'track_id': _next_id,
        'x_trk': np.array([x, y, vx, vy], dtype=np.float64),
        'P_trk': np.diag([0.5, 0.5, 1.0, 1.0]),  # [m^2, m^2, (m/s)^2, (m/s)^2]
        'n_matched': 0,
        'confirmed': False,
        'n_lost': 0,
        'is_dynamic': False,
        'history': [(x, y)],
    }
    _next_id += 1
    return trk

def track_predict(trackers, dt=DT, sigma_pos=SIGMA_POS, sigma_vel=SIGMA_VEL):
    """
    Predict all trackers forward by one time step using constant-velocity model.

    :param trackers: (list) List of tracker dicts
    :param dt: (float) Time step in seconds
    :param sigma_pos: (float) Process noise standard deviation for position in m
    :param sigma_vel: (float) Process noise standard deviation for velocity in m/s
    :return: (list) Updated list of tracker dicts after prediction
    """
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    Q = np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2])
    for trk in trackers:
        trk['x_trk'] = F @ trk['x_trk']
        trk['P_trk'] = F @ trk['P_trk'] @ F.T + Q
    return trackers

def _mahal_distance(det_center, tracker, sigma_obs=SIGMA_OBS):
    """
    Compute Mahalanobis distance between a detection and a tracker.

    :param det_center: (np.ndarray) Length-2 detection center [x, y]
    :param tracker: (dict) Tracker dictionary with 'x_trk' and 'P_trk'
    :param sigma_obs: (float) Observation noise standard deviation in meters
    :return: (float) Mahalanobis distance
    """
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64)
    S = H @ tracker['P_trk'] @ H.T + np.diag([sigma_obs**2, sigma_obs**2])
    dz = det_center - H @ tracker['x_trk']
    S_reg = S + np.eye(2) * 1e-9
    d_mahal = np.sqrt(max(0.0, dz @ np.linalg.inv(S_reg) @ dz))
    return d_mahal

def hungarian_associate(detections, trackers, sigma_obs=SIGMA_OBS, n_confirm=N_CONFIRM,
                        chi2_new=CHI2_NEW, chi2_stable=CHI2_STABLE):
    """
    Associate detections to trackers using Hungarian algorithm with Mahalanobis gating.

    :param detections: (list) List of detection dicts with 'center' key
    :param trackers: (list) List of tracker dicts
    :param sigma_obs: (float) Observation noise standard deviation in meters
    :param n_confirm: (int) Number of matches needed to confirm a tracker
    :param chi2_new: (float) Chi-squared gate threshold for unconfirmed trackers
    :param chi2_stable: (float) Chi-squared gate threshold for confirmed trackers
    :return: (tuple) (associations, unmatched_det, unmatched_trk) — association pairs and unmatched indices
    """
    if len(detections) == 0 or len(trackers) == 0:
        return [], list(range(len(detections))), list(range(len(trackers)))
    det_centers = np.array([d["center"] for d in detections])
    n_det = len(detections)
    n_trk = len(trackers)
    cost = np.zeros((n_det, n_trk), dtype=np.float64)
    for j in range(n_trk):
        for i in range(n_det):
            cost[i, j] = _mahal_distance(det_centers[i], trackers[j], sigma_obs)
    cost_clipped = np.clip(cost, 0, 1e3)
    row_ind, col_ind = linear_sum_assignment(cost_clipped)
    associations = []
    unmatched_det = set(range(n_det))
    unmatched_trk = set(range(n_trk))
    for r, c in zip(row_ind, col_ind):
        chi2_th = chi2_new if trackers[c]['n_matched'] < n_confirm else chi2_stable
        if cost[r, c] <= np.sqrt(chi2_th):
            associations.append((r, c))
            unmatched_det.discard(r)
            unmatched_trk.discard(c)
    return associations, list(unmatched_det), list(unmatched_trk)

def track_update(trackers, associations, detections, sigma_obs=SIGMA_OBS):
    """
    Update tracker states with associated detections using Kalman update (Joseph form).

    :param trackers: (list) List of tracker dicts
    :param associations: (list) List of (det_idx, trk_idx) pairs
    :param detections: (list) List of detection dicts with 'center' key
    :param sigma_obs: (float) Observation noise standard deviation in meters
    :return: (list) Updated list of tracker dicts
    """
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64)
    R = np.diag([sigma_obs**2, sigma_obs**2])
    I4 = np.eye(4)
    for det_idx, trk_idx in associations:
        trk = trackers[trk_idx]
        z = detections[det_idx]["center"]
        z_pred = H @ trk['x_trk']
        S = H @ trk['P_trk'] @ H.T + R
        S_reg = S + np.eye(2) * 1e-9
        K = trk['P_trk'] @ H.T @ np.linalg.inv(S_reg)
        dz = z - z_pred
        trk['x_trk'] = trk['x_trk'] + K @ dz
        I_KH = I4 - K @ H
        trk['P_trk'] = I_KH @ trk['P_trk'] @ I_KH.T + K @ R @ K.T
        trk['n_matched'] += 1
        trk['n_lost'] = 0
        trk['history'].append((trk['x_trk'][0], trk['x_trk'][1]))
    return trackers

def track_manage(trackers, associations, detections, n_confirm=N_CONFIRM, n_delete=N_DELETE,
                 v_dynamic_thresh=V_DYNAMIC_THRESH, n_dynamic_extra=N_DYNAMIC_EXTRA):
    """
    Manage tracker lifecycle: confirm, delete lost, create new, classify dynamic.

    :param trackers: (list) List of tracker dicts
    :param associations: (list) List of (det_idx, trk_idx) pairs
    :param detections: (list) List of detection dicts with 'center' key
    :param n_confirm: (int) Matches needed for confirmation
    :param n_delete: (int) Lost frames before deletion
    :param v_dynamic_thresh: (float) Speed threshold for dynamic classification in m/s
    :param n_dynamic_extra: (int) Extra matches beyond confirm for dynamic check
    :return: (list) Updated list of tracker dicts
    """
    associated_trk = {c for _, c in associations}
    associated_det = {d for d, _ in associations}
    for i, trk in enumerate(trackers):
        if i not in associated_trk:
            trk['n_lost'] += 1
        if trk['n_matched'] >= n_confirm and not trk['confirmed']:
            trk['confirmed'] = True
        speed = np.sqrt(trk['x_trk'][2]**2 + trk['x_trk'][3]**2)
        if speed > v_dynamic_thresh and trk['n_matched'] >= n_confirm + n_dynamic_extra:
            trk['is_dynamic'] = True
    new_trackers = []
    for det_idx in range(len(detections)):
        if det_idx not in associated_det:
            det = detections[det_idx]
            new_trackers.append(create_tracker(det["center"][0], det["center"][1]))
    trackers = [t for t in trackers if t['n_lost'] < n_delete]
    trackers.extend(new_trackers)
    return trackers

def main():
    print("Kalman Tracker Demo")
    reset_tracker_id()
    trackers = [create_tracker(0.0, 0.0) for _ in range(3)]
    trackers = track_predict(trackers, dt=0.1)
    print(f"Created {len(trackers)} trackers")
    detections = [{"center": np.array([0.1, 0.1])}, {"center": np.array([5.0, 5.0])}]
    matches, unmatched_det, unmatched_trk = hungarian_associate(detections, trackers, sigma_obs=2.0)
    print(f"Association matches: {matches}")

if __name__ == '__main__':
    main()
