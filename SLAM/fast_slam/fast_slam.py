"""FastSLAM 2.0 with Ackermann bicycle model and particle filter

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [FastSLAM](https://www.aaai.org/Papers/AAAI/2002/AAAI02-089.pdf)
"""
import math
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import *

from SLAM.config import (WHEELBASE, UKF_DT, LIDAR_RANGE_MAX,
                          FASTSLAM_Q_V, FASTSLAM_Q_THETA,
                          FASTSLAM_R_RANGE, FASTSLAM_R_BEARING,
                          FASTSLAM_MAX_RANGE, FASTSLAM_LM_P_INIT)
from SLAM.slam_sim import angle_mod, generate_reference_trajectory, generate_landmarks

# === Phase 1: Parameters ===
N_PARTICLES = 100

Q_MOTION = np.diag([FASTSLAM_Q_V**2, FASTSLAM_Q_V**2, FASTSLAM_Q_THETA**2])
R_OBS = np.diag([FASTSLAM_R_RANGE**2, FASTSLAM_R_BEARING**2])
LM_INIT_P = np.eye(2) * FASTSLAM_LM_P_INIT

show_animation = True


# === Phase 2: FastSLAM 2.0 ===

class FastSLAM:

    def __init__(self, n_particles, dt, max_landmark_range, max_landmarks=200):
        self.n_particles = n_particles
        self.dt = dt
        self.max_landmark_range = max_landmark_range
        self.max_landmarks = max_landmarks
        self.poses = np.zeros((n_particles, 3))
        self.weights = np.ones(n_particles) / n_particles
        self.log_weights = np.zeros(n_particles)
        self.landmarks = np.zeros((n_particles, max_landmarks, 2))
        self.lm_covs = np.zeros((n_particles, max_landmarks, 2, 2))
        self.lm_initialized = np.zeros((n_particles, max_landmarks), dtype=bool)
        self.n_landmarks_per_particle = np.zeros(n_particles, dtype=int)
        self.n_landmarks = 0

    def init_particles(self, x0, n_lm):
        self.n_landmarks = n_lm
        self.poses[:] = x0[:3]
        self.weights[:] = 1.0 / self.n_particles
        self.log_weights[:] = -np.log(self.n_particles)
        self.landmarks[:] = 0.0
        self.lm_covs[:] = np.array([LM_INIT_P.copy()] * self.max_landmarks)
        self.lm_initialized[:] = False
        self.n_landmarks_per_particle[:] = n_lm

    def predict(self, v, delta):
        dt = self.dt
        theta = self.poses[:, 2]
        d_theta = v / WHEELBASE * np.tan(delta) * dt
        self.poses[:, 0] += v * np.cos(theta) * dt
        self.poses[:, 1] += v * np.sin(theta) * dt
        self.poses[:, 2] = angle_mod(theta + d_theta)
        noise = np.random.multivariate_normal(np.zeros(3), Q_MOTION, self.n_particles)
        self.poses += noise
        self.poses[:, 2] = angle_mod(self.poses[:, 2])

    def update(self, z_obs):
        if z_obs is None or len(z_obs) == 0:
            return

        Np = self.n_particles
        max_lm = self.n_landmarks

        for z in z_obs:
            dist, bearing = z[0], z[1]
            if dist >= self.max_landmark_range:
                continue

            theta_all = self.poses[:, 2]
            lm_x_all = self.poses[:, 0] + dist * np.cos(theta_all + bearing)
            lm_y_all = self.poses[:, 1] + dist * np.sin(theta_all + bearing)

            if max_lm > 0:
                lm_pos = np.column_stack([lm_x_all, lm_y_all])
                dx = lm_pos[:, np.newaxis, 0] - self.landmarks[:, :max_lm, 0]
                dy = lm_pos[:, np.newaxis, 1] - self.landmarks[:, :max_lm, 1]

                a = self.lm_covs[:, :max_lm, 0, 0] + 1e-9
                b = self.lm_covs[:, :max_lm, 0, 1]
                d = self.lm_covs[:, :max_lm, 1, 1] + 1e-9
                det = a * d - b * b
                inv_det = np.clip(1.0 / (np.abs(det) + 1e-300), 0, 1e15)
                mahal_raw = np.clip(d * dx * dx - 2.0 * b * dx * dy + a * dy * dy, -1e15, 1e15)
                mahal = np.clip(inv_det * mahal_raw, -1e18, 1e18)

                uninit_mask = (self.lm_covs[:, :max_lm, 0, 0] > 1e5) | (self.lm_covs[:, :max_lm, 1, 1] > 1e5)
                n_lm_per = self.n_landmarks_per_particle
                range_mask = np.arange(max_lm)[np.newaxis, :] >= n_lm_per[:, np.newaxis]
                invalid_mask = uninit_mask | range_mask

                mahal_masked = np.where(invalid_mask, 1e18, mahal)
                best_init_idx = np.argmin(mahal_masked, axis=1)
                best_init_mahal = mahal_masked[np.arange(Np), best_init_idx]

                first_uninit_idx = np.full(Np, -1, dtype=int)
                has_uninit = np.any(uninit_mask & ~range_mask, axis=1)
                if np.any(has_uninit):
                    uninit_only = np.where(uninit_mask & ~range_mask, np.arange(max_lm)[np.newaxis, :], max_lm)
                    first_uninit_idx[has_uninit] = np.argmin(uninit_only[has_uninit], axis=1)
            else:
                best_init_idx = np.zeros(Np, dtype=int)
                best_init_mahal = np.full(Np, 1e18)
                first_uninit_idx = np.full(Np, -1, dtype=int)
                has_uninit = np.zeros(Np, dtype=bool)

            lm_idx_arr = np.full(Np, -1, dtype=int)
            need_add = np.zeros(Np, dtype=bool)
            need_init = np.zeros(Np, dtype=bool)

            for p in range(Np):
                n_lm = self.n_landmarks_per_particle[p]
                if n_lm == 0:
                    need_add[p] = True
                    need_init[p] = True
                    continue

                bi = best_init_idx[p]
                bm = best_init_mahal[p]
                fu = first_uninit_idx[p]

                if bm < 9.0:
                    idx = bi
                elif fu >= 0:
                    idx = fu
                elif bm < 1e18:
                    idx = bi
                else:
                    need_add[p] = True
                    need_init[p] = True
                    continue

                if self.lm_covs[p, idx, 0, 0] > 1e5 or self.lm_covs[p, idx, 1, 1] > 1e5:
                    need_init[p] = True
                    lm_idx_arr[p] = idx
                else:
                    lm_idx_arr[p] = idx

            for p in np.where(need_add)[0]:
                lm_idx_arr[p] = self._add_landmark(p, dist, bearing)

            for p in np.where(need_init)[0]:
                self._init_landmark(p, lm_idx_arr[p], dist, bearing)

            ekf_mask = (lm_idx_arr >= 0) & ~need_init
            if np.any(ekf_mask):
                ekf_particles = np.where(ekf_mask)[0]
                ekf_lm_idx = lm_idx_arr[ekf_mask]
                log_w_arr = self._ekf_update_batch(ekf_particles, ekf_lm_idx, dist, bearing)
                self.log_weights[ekf_particles] += log_w_arr

            max_lm = self.n_landmarks

        self.log_weights = np.clip(self.log_weights, -700, 700)
        max_lw = np.max(self.log_weights)
        self.weights = np.exp(self.log_weights - max_lw)
        total_w = self.weights.sum()
        if total_w > 1e-300:
            self.weights /= total_w
        else:
            self.weights[:] = 1.0 / self.n_particles

        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2.0:
            self._resample()

    def _ekf_update_batch(self, particles, lm_indices, dist, bearing):
        n = len(particles)
        x = self.poses[particles, 0]
        y = self.poses[particles, 1]
        theta = self.poses[particles, 2]
        lx = self.landmarks[particles, lm_indices, 0]
        ly = self.landmarks[particles, lm_indices, 1]

        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy + 1e-18
        sqrt_q = np.sqrt(q)

        z_pred_b = angle_mod(np.arctan2(dy, dx) - theta)
        innov_d = dist - sqrt_q
        innov_b = angle_mod(bearing - z_pred_b)

        H00 = dx / sqrt_q
        H01 = dy / sqrt_q
        H10 = -dy / q
        H11 = dx / q

        P00 = self.lm_covs[particles, lm_indices, 0, 0]
        P01 = self.lm_covs[particles, lm_indices, 0, 1]
        P10 = self.lm_covs[particles, lm_indices, 1, 0]
        P11 = self.lm_covs[particles, lm_indices, 1, 1]

        HP00 = H00 * P00 + H01 * P10
        HP01 = H00 * P01 + H01 * P11
        HP10 = H10 * P00 + H11 * P10
        HP11 = H10 * P01 + H11 * P11

        S00 = HP00 * H00 + HP01 * H01 + R_OBS[0, 0]
        S01 = HP00 * H10 + HP01 * H11 + R_OBS[0, 1]
        S10 = HP10 * H00 + HP11 * H01 + R_OBS[1, 0]
        S11 = HP10 * H10 + HP11 * H11 + R_OBS[1, 1]

        S00 = S00 + 1e-9
        S01 = 0.5 * (S01 + S10)
        S10 = S01
        S11 = S11 + 1e-9

        det_S = S00 * S11 - S01 * S10
        det_S = np.maximum(np.abs(det_S), 1e-300)
        inv_det = 1.0 / det_S
        SI00 = S11 * inv_det
        SI01 = -S01 * inv_det
        SI10 = -S10 * inv_det
        SI11 = S00 * inv_det

        K00 = P00 * SI00 + P01 * SI10
        K01 = P00 * SI01 + P01 * SI11
        K10 = P10 * SI00 + P11 * SI10
        K11 = P10 * SI01 + P11 * SI11

        self.landmarks[particles, lm_indices, 0] += K00 * innov_d + K01 * innov_b
        self.landmarks[particles, lm_indices, 1] += K10 * innov_d + K11 * innov_b

        I_KH00 = 1.0 - K00 * H00 - K01 * H10
        I_KH01 = -K00 * H01 - K01 * H11
        I_KH10 = -K10 * H00 - K11 * H10
        I_KH11 = 1.0 - K10 * H01 - K11 * H11

        new_P00 = I_KH00 * P00 + I_KH01 * P10
        new_P01 = I_KH00 * P01 + I_KH01 * P11
        new_P10 = I_KH10 * P00 + I_KH11 * P10
        new_P11 = I_KH10 * P01 + I_KH11 * P11

        self.lm_covs[particles, lm_indices, 0, 0] = new_P00
        self.lm_covs[particles, lm_indices, 0, 1] = 0.5 * (new_P01 + new_P10)
        self.lm_covs[particles, lm_indices, 1, 0] = 0.5 * (new_P01 + new_P10)
        self.lm_covs[particles, lm_indices, 1, 1] = new_P11

        innov_Sinnov = innov_d * (SI00 * innov_d + SI01 * innov_b) + innov_b * (SI10 * innov_d + SI11 * innov_b)
        exponent = -0.5 * innov_Sinnov
        exponent = np.clip(exponent, -700, 700)
        w_inc = np.exp(exponent) / np.sqrt(2.0 * np.pi * det_S)
        w_inc = np.maximum(w_inc, 1e-300)
        return np.log(w_inc)

    def _add_landmark(self, p, dist, bearing):
        x, y, theta = self.poses[p]
        lm_x = x + dist * cos(theta + bearing)
        lm_y = y + dist * sin(theta + bearing)
        idx = self.n_landmarks_per_particle[p]
        assert idx < self.max_landmarks, f"路标数超过max_landmarks={self.max_landmarks}"
        self.landmarks[p, idx] = [lm_x, lm_y]
        self.lm_covs[p, idx] = LM_INIT_P.copy()
        self.lm_initialized[p, idx] = True
        self.n_landmarks_per_particle[p] = idx + 1
        self.n_landmarks = max(self.n_landmarks, idx + 1)
        return idx

    def _init_landmark(self, p, lm_idx, dist, bearing):
        x, y, theta = self.poses[p]
        self.landmarks[p, lm_idx, 0] = x + dist * cos(theta + bearing)
        self.landmarks[p, lm_idx, 1] = y + dist * sin(theta + bearing)

        G = np.array([
            [cos(theta + bearing), -dist * sin(theta + bearing)],
            [sin(theta + bearing),  dist * cos(theta + bearing)]
        ])
        self.lm_covs[p, lm_idx] = G @ R_OBS @ G.T

    def get_estimated_pose(self):
        weights = self.weights / (self.weights.sum() + 1e-300)
        x_est = np.average(self.poses[:, 0], weights=weights)
        y_est = np.average(self.poses[:, 1], weights=weights)
        sin_sum = np.sum(weights * np.sin(self.poses[:, 2]))
        cos_sum = np.sum(weights * np.cos(self.poses[:, 2]))
        theta_est = np.arctan2(sin_sum, cos_sum)
        return np.array([x_est, y_est, theta_est])

    def get_landmarks(self):
        best_p = np.argmax(self.weights)
        lm = self.landmarks[best_p, :self.n_landmarks_per_particle[best_p]].copy()
        initialized = np.max(self.lm_covs[best_p, :self.n_landmarks_per_particle[best_p]], axis=(1, 2)) < 1e5
        return lm[initialized]


    def _resample(self):
        n = self.n_particles
        weights = self.weights / (self.weights.sum() + 1e-300)
        cum_w = np.cumsum(weights)
        cum_w[-1] = 1.0
        positions = (np.arange(n) + np.random.uniform()) / n
        indices = np.searchsorted(cum_w, positions)

        self.poses[:] = self.poses[indices]
        self.weights[:] = 1.0 / n
        self.log_weights[:] = -np.log(n)
        self.landmarks[:] = self.landmarks[indices]
        self.lm_covs[:] = self.lm_covs[indices]
        self.lm_initialized[:] = self.lm_initialized[indices]
        self.n_landmarks_per_particle[:] = self.n_landmarks_per_particle[indices]


# === Phase 3: Observation simulation ===

def _simulate_observation(true_pose, landmarks, max_range):
    """
    :param true_pose: (ndarray) True robot pose [x, y, theta]
    :param landmarks: (ndarray) Landmark positions, shape (n_lm, 2)
    :param max_range: (float) Maximum observation range [m]
    :return: (ndarray) Observations, shape (n_obs, 2), columns [distance, bearing]
    """
    dx = landmarks[:, 0] - true_pose[0]
    dy = landmarks[:, 1] - true_pose[1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    bearings = np.arctan2(dy, dx) - true_pose[2]
    bearings = angle_mod(bearings)

    mask = dists < max_range
    if not np.any(mask):
        return np.zeros((0, 2))

    z = np.column_stack([dists[mask], bearings[mask]])
    z[:, 0] += np.random.randn(len(z)) * np.sqrt(R_OBS[0, 0])
    z[:, 1] += np.random.randn(len(z)) * np.sqrt(R_OBS[1, 1])
    z[:, 1] = angle_mod(z[:, 1])
    return z


# === Phase 5: Demo ===

def main():
    dt = UKF_DT
    ref_traj = generate_reference_trajectory('figure8', dt)
    landmarks = generate_landmarks(n_lm=50, map_size=10.0)
    n_steps = len(ref_traj)

    slam = FastSLAM(n_particles=N_PARTICLES, dt=dt, max_landmark_range=LIDAR_RANGE_MAX)
    slam.init_particles(ref_traj[0], n_lm=len(landmarks))

    traj_est = np.zeros((n_steps, 3))
    traj_est[0] = ref_traj[0]

    for i in range(1, n_steps):
        dx = ref_traj[i, 0] - ref_traj[i - 1, 0]
        dy = ref_traj[i, 1] - ref_traj[i - 1, 1]
        dtheta = angle_mod(ref_traj[i, 2] - ref_traj[i - 1, 2])
        v_cmd = np.sqrt(dx ** 2 + dy ** 2) / dt
        delta_cmd = np.arctan2(dtheta * WHEELBASE, max(v_cmd * dt, 1e-9))

        slam.predict(v_cmd, delta_cmd)

        z_obs = _simulate_observation(ref_traj[i], landmarks, LIDAR_RANGE_MAX)
        slam.update(z_obs)

        traj_est[i] = slam.get_estimated_pose()

        if i % 100 == 0:
            print(f"[FastSLAM] step {i}/{n_steps}")

    print(f"[FastSLAM] done, {n_steps} steps")

    if show_animation:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('FastSLAM 2.0 Demo')

        ax.plot(landmarks[:, 0], landmarks[:, 1], 'go', markersize=8, label='True Landmarks')

        true_line, = ax.plot([], [], 'k-', linewidth=1.5, label='True Trajectory')
        est_line, = ax.plot([], [], 'b--', linewidth=1.5, label='Estimated Trajectory')
        particle_dots, = ax.plot([], [], 'c.', alpha=0.3, markersize=2, label='Particles')
        est_lm_dots, = ax.plot([], [], 'rx', markersize=6, label='Estimated Landmarks')
        ax.legend(loc='upper left', frameon=True, fancybox=True)

        skip = max(1, n_steps // 300)

        def _init():
            true_line.set_data([], [])
            est_line.set_data([], [])
            particle_dots.set_data([], [])
            est_lm_dots.set_data([], [])
            return true_line, est_line, particle_dots, est_lm_dots

        def _update(frame):
            idx = min(frame * skip, n_steps - 1)
            true_line.set_data(ref_traj[:idx + 1, 0], ref_traj[:idx + 1, 1])
            est_line.set_data(traj_est[:idx + 1, 0], traj_est[:idx + 1, 1])

            px = slam.poses[:, 0]
            py = slam.poses[:, 1]
            particle_dots.set_data(px, py)

            lm_est = slam.get_landmarks()
            if len(lm_est) > 0:
                est_lm_dots.set_data(lm_est[:, 0], lm_est[:, 1])

            return true_line, est_line, particle_dots, est_lm_dots

        n_frames = n_steps // skip + 1
        ani = animation.FuncAnimation(fig, _update, frames=n_frames,
                                      init_func=_init, blit=True,
                                      interval=30, repeat=False)

        fig_dir = pathlib.Path(__file__).parent.parent / 'figs'
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / 'slam_fastslam_demo.png'

        try:
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
            print(f"[save] {save_path}")
        except Exception:
            pass

        plt.show()


if __name__ == '__main__':
    main()
