"""Extended Kalman Filter SLAM with Ackermann bicycle model

author: Kat-yuan-eng (RuiWen Liao)

Reference:
    - [EKF-SLAM](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from SLAM.config import (WHEELBASE, UKF_DT, LIDAR_RANGE_MAX,
    LIDAR_SIGMA_RANGE, LIDAR_SIGMA_BEARING)
from SLAM.slam_sim import (generate_reference_trajectory, generate_landmarks,
    angle_mod)

show_animation = True

# === Phase 1: EKF-SLAM class ===

class EKFSLAM:

    def __init__(self, dt, max_landmark_range, max_landmarks=100):
        self.dt = dt
        self.max_landmark_range = max_landmark_range
        self.max_landmarks = max_landmarks
        self.n_state = 3 + 2 * max_landmarks
        self.n_active = 3
        self.x_est = np.zeros(self.n_state)
        self.P_est = np.zeros((self.n_state, self.n_state))
        self.P_est[:3, :3] = np.eye(3) * 1e-6
        self.n_landmark = 0
        self.lm_initialized = np.zeros(max_landmarks, dtype=bool)
        self.R_obs = np.diag([LIDAR_SIGMA_RANGE**2, LIDAR_SIGMA_BEARING**2])
        self.Q_motion = np.diag([0.01, 0.01, np.deg2rad(0.5)**2])

    def predict(self, v, delta):
        x, y, theta = self.x_est[:3]
        dt = self.dt
        L = WHEELBASE
        theta_n = angle_mod(theta + v * np.tan(delta) / L * dt)
        self.x_est[0] += v * np.cos(theta) * dt
        self.x_est[1] += v * np.sin(theta) * dt
        self.x_est[2] = theta_n

        n = self.n_active
        a = -v * np.sin(theta) * dt
        b = v * np.cos(theta) * dt

        G_v = np.zeros((3, 3))
        G_v[0, 0] = np.cos(theta) * dt
        G_v[1, 0] = np.sin(theta) * dt
        G_v[2, 0] = np.tan(delta) / L * dt
        G_v[2, 1] = v * dt / (L * max(np.cos(delta)**2, 1e-9))

        P_sub = self.P_est[:n, :n]
        P_col2 = P_sub[:, 2].copy()
        P_row2 = P_sub[2, :].copy()
        P_22 = P_sub[2, 2]
        P_sub[0, :] += a * P_row2
        P_sub[1, :] += b * P_row2
        P_sub[:, 0] += a * P_col2
        P_sub[:, 1] += b * P_col2
        P_sub[0, 0] += a * a * P_22
        P_sub[0, 1] += a * b * P_22
        P_sub[1, 0] += a * b * P_22
        P_sub[1, 1] += b * b * P_22
        P_sub[:3, :3] += G_v @ self.Q_motion @ G_v.T
        diag_vals = np.diag(P_sub).copy()
        np.clip(diag_vals, 0.0, 1e6, out=diag_vals)
        np.fill_diagonal(P_sub, diag_vals)
        P_sub += np.eye(n) * 1e-12
        self.P_est[:n, :n] = 0.5 * (P_sub + P_sub.T)

    def update(self, z_obs, max_obs=20):
        if z_obs.ndim == 1:
            z_obs = z_obs.reshape(1, 2)
        n_obs = z_obs.shape[0]
        if n_obs > max_obs:
            step = max(1, n_obs // max_obs)
            z_obs = z_obs[::step][:max_obs]
            n_obs = z_obs.shape[0]
        n = self.n_active
        for i in range(n_obs):
            z_d, z_b = z_obs[i]
            lm_id = self._find_nearest_landmark(z_d, z_b)
            if lm_id is None:
                if self.n_landmark >= self.max_landmarks:
                    continue
                lm_id = self._add_landmark(z_d, z_b)
                n = self.n_active
            idx = 3 + 2 * lm_id
            dx = self.x_est[idx] - self.x_est[0]
            dy = self.x_est[idx + 1] - self.x_est[1]
            d2 = dx**2 + dy**2 + 1e-18
            d = np.sqrt(d2)
            z_pred = np.array([d, angle_mod(np.arctan2(dy, dx) - self.x_est[2])])
            innovation = np.array([z_d - z_pred[0], angle_mod(z_b - z_pred[1])])
            h00 = -dx / d; h01 = -dy / d; h0i = dx / d; h0i1 = dy / d
            h10 = dy / d2; h11 = -dx / d2; h1i = -dy / d2; h1i1 = dx / d2
            P_sub = self.P_est[:n, :n]
            PHt_col0 = h00 * P_sub[:, 0] + h01 * P_sub[:, 1] + h0i * P_sub[:, idx] + h0i1 * P_sub[:, idx + 1]
            PHt_col1 = h10 * P_sub[:, 0] + h11 * P_sub[:, 1] - P_sub[:, 2] + h1i * P_sub[:, idx] + h1i1 * P_sub[:, idx + 1]
            S00 = h00 * PHt_col0[0] + h01 * PHt_col0[1] + h0i * PHt_col0[idx] + h0i1 * PHt_col0[idx + 1] + self.R_obs[0, 0]
            S01 = h10 * PHt_col0[0] + h11 * PHt_col0[1] - PHt_col0[2] + h1i * PHt_col0[idx] + h1i1 * PHt_col0[idx + 1]
            S11 = h10 * PHt_col1[0] + h11 * PHt_col1[1] - PHt_col1[2] + h1i * PHt_col1[idx] + h1i1 * PHt_col1[idx + 1] + self.R_obs[1, 1]
            S_det = S00 * S11 - S01 * S01
            if abs(S_det) < 1e-30:
                continue
            S_inv00 = S11 / S_det; S_inv01 = -S01 / S_det; S_inv11 = S00 / S_det
            K_col0 = S_inv00 * PHt_col0 + S_inv01 * PHt_col1
            K_col1 = S_inv01 * PHt_col0 + S_inv11 * PHt_col1
            self.x_est[:n] += K_col0 * innovation[0] + K_col1 * innovation[1]
            self.x_est[2] = angle_mod(self.x_est[2])
            HP_row0 = h00 * P_sub[0, :] + h01 * P_sub[1, :] + h0i * P_sub[idx, :] + h0i1 * P_sub[idx + 1, :]
            HP_row1 = h10 * P_sub[0, :] + h11 * P_sub[1, :] - P_sub[2, :] + h1i * P_sub[idx, :] + h1i1 * P_sub[idx + 1, :]
            P_new = P_sub - np.outer(K_col0, HP_row0) - np.outer(K_col1, HP_row1)
            P_new = 0.5 * (P_new + P_new.T)
            diag_vals = np.diag(P_new).copy()
            np.clip(diag_vals, 0.0, 1e6, out=diag_vals)
            np.fill_diagonal(P_new, diag_vals)
            self.P_est[:n, :n] = P_new

    def get_state(self):
        robot_pose = self.x_est[:3].copy()
        if self.n_landmark > 0:
            landmarks = self.x_est[3:3 + 2 * self.n_landmark].reshape(self.n_landmark, 2)
        else:
            landmarks = np.zeros((0, 2))
        return robot_pose, landmarks

    def _add_landmark(self, z_d, z_b):
        idx = self.n_landmark
        lx = self.x_est[0] + z_d * np.cos(self.x_est[2] + z_b)
        ly = self.x_est[1] + z_d * np.sin(self.x_est[2] + z_b)
        lm_idx = 3 + 2 * idx
        self.x_est[lm_idx] = lx
        self.x_est[lm_idx + 1] = ly
        old_n = self.n_active
        new_n = old_n + 2
        self.P_est[old_n:new_n, :new_n] = 0.0
        self.P_est[:new_n, old_n:new_n] = 0.0
        G_l = np.zeros((2, old_n))
        G_l[0, 0] = 1; G_l[0, 2] = -z_d * np.sin(self.x_est[2] + z_b)
        G_l[1, 1] = 1; G_l[1, 2] = z_d * np.cos(self.x_est[2] + z_b)
        G_z = np.zeros((2, 2))
        G_z[0, 0] = np.cos(self.x_est[2] + z_b)
        G_z[0, 1] = -z_d * np.sin(self.x_est[2] + z_b)
        G_z[1, 0] = np.sin(self.x_est[2] + z_b)
        G_z[1, 1] = z_d * np.cos(self.x_est[2] + z_b)
        P_new = G_l @ self.P_est[:old_n, :old_n] @ G_l.T + G_z @ self.R_obs @ G_z.T
        self.P_est[old_n:new_n, old_n:new_n] = P_new
        self.n_landmark += 1
        self.n_active = new_n
        self.lm_initialized[idx] = True
        return idx

    def _find_nearest_landmark(self, z_d, z_b):
        """数据关联：向量化马氏距离计算"""
        if self.n_landmark == 0:
            return None
        n = self.n_active
        P_sub = self.P_est[:n, :n]
        lm_start = 3
        lm_arr = self.x_est[lm_start:lm_start + 2 * self.n_landmark].reshape(self.n_landmark, 2)
        dx = lm_arr[:, 0] - self.x_est[0]
        dy = lm_arr[:, 1] - self.x_est[1]
        q = dx**2 + dy**2 + 1e-18
        sq = np.sqrt(q)
        z_pred_d = sq
        z_pred_b = angle_mod(np.arctan2(dy, dx) - self.x_est[2])
        h00 = -dx / sq; h01 = -dy / sq; h0i = dx / sq; h0i1 = dy / sq
        h10 = dy / q; h11 = -dx / q; h1i = -dy / q; h1i1 = dx / q
        lm_indices = np.arange(self.n_landmark)
        lm_col0 = lm_indices * 2 + lm_start
        lm_col1 = lm_col0 + 1
        HP0_0 = h00 * P_sub[0, 0] + h01 * P_sub[1, 0] + h0i * P_sub[lm_col0, 0] + h0i1 * P_sub[lm_col1, 0]
        HP0_1 = h00 * P_sub[0, 1] + h01 * P_sub[1, 1] + h0i * P_sub[lm_col0, 1] + h0i1 * P_sub[lm_col1, 1]
        HP0_2 = h00 * P_sub[0, 2] + h01 * P_sub[1, 2] + h0i * P_sub[lm_col0, 2] + h0i1 * P_sub[lm_col1, 2]
        HP0_lm0 = h00 * P_sub[0, lm_col0] + h01 * P_sub[1, lm_col0] + h0i * P_sub[lm_col0, lm_col0] + h0i1 * P_sub[lm_col1, lm_col0]
        HP0_lm1 = h00 * P_sub[0, lm_col1] + h01 * P_sub[1, lm_col1] + h0i * P_sub[lm_col0, lm_col1] + h0i1 * P_sub[lm_col1, lm_col1]
        HP1_0 = h10 * P_sub[0, 0] + h11 * P_sub[1, 0] - P_sub[2, 0] + h1i * P_sub[lm_col0, 0] + h1i1 * P_sub[lm_col1, 0]
        HP1_1 = h10 * P_sub[0, 1] + h11 * P_sub[1, 1] - P_sub[2, 1] + h1i * P_sub[lm_col0, 1] + h1i1 * P_sub[lm_col1, 1]
        HP1_2 = h10 * P_sub[0, 2] + h11 * P_sub[1, 2] - P_sub[2, 2] + h1i * P_sub[lm_col0, 2] + h1i1 * P_sub[lm_col1, 2]
        HP1_lm0 = h10 * P_sub[0, lm_col0] + h11 * P_sub[1, lm_col0] - P_sub[2, lm_col0] + h1i * P_sub[lm_col0, lm_col0] + h1i1 * P_sub[lm_col1, lm_col0]
        HP1_lm1 = h10 * P_sub[0, lm_col1] + h11 * P_sub[1, lm_col1] - P_sub[2, lm_col1] + h1i * P_sub[lm_col0, lm_col1] + h1i1 * P_sub[lm_col1, lm_col1]
        S00 = HP0_0*h00 + HP0_1*h01 + HP0_lm0*h0i + HP0_lm1*h0i1 + self.R_obs[0, 0]
        S01 = HP0_0*h10 + HP0_1*h11 + HP0_2*(-1) + HP0_lm0*h1i + HP0_lm1*h1i1
        S11 = HP1_0*h10 + HP1_1*h11 + HP1_2*(-1) + HP1_lm0*h1i + HP1_lm1*h1i1 + self.R_obs[1, 1]
        S_det = S00 * S11 - S01 * S01
        valid = np.abs(S_det) > 1e-30
        S_inv00 = np.where(valid, S11 / np.where(valid, S_det, 1.0), 0.0)
        S_inv01 = np.where(valid, -S01 / np.where(valid, S_det, 1.0), 0.0)
        S_inv11 = np.where(valid, S00 / np.where(valid, S_det, 1.0), 0.0)
        innov_d = z_d - z_pred_d
        innov_b = angle_mod(z_b - z_pred_b)
        mah = innov_d * (S_inv00 * innov_d + S_inv01 * innov_b) + innov_b * (S_inv01 * innov_d + S_inv11 * innov_b)
        mah = np.where(valid, mah, np.inf)
        best_j = int(np.argmin(mah))
        if mah[best_j] < 25.0:
            return best_j
        return None

    def _observation_model(self, lm_id):
        idx = 3 + 2 * lm_id
        dx = self.x_est[idx] - self.x_est[0]
        dy = self.x_est[idx + 1] - self.x_est[1]
        d = np.sqrt(dx**2 + dy**2) + 1e-18
        bearing = angle_mod(np.arctan2(dy, dx) - self.x_est[2])
        return np.array([d, bearing])

    def _calc_jacobian_h(self, lm_id):
        idx = 3 + 2 * lm_id
        dx = self.x_est[idx] - self.x_est[0]
        dy = self.x_est[idx + 1] - self.x_est[1]
        d2 = dx**2 + dy**2 + 1e-18
        d = np.sqrt(d2)
        n = self.n_active
        H = np.zeros((2, n))
        H[0, 0] = -dx / d
        H[0, 1] = -dy / d
        H[0, idx] = dx / d
        H[0, idx + 1] = dy / d
        H[1, 0] = dy / d2
        H[1, 1] = -dx / d2
        H[1, 2] = -1.0
        H[1, idx] = -dy / d2
        H[1, idx + 1] = dx / d2
        return H


# === Phase 2: Observation simulation ===

def simulate_observation(x_true, landmarks, max_range, noise_std):
    dx = landmarks[:, 0] - x_true[0]
    dy = landmarks[:, 1] - x_true[1]
    dist = np.sqrt(dx**2 + dy**2)
    bearing = angle_mod(np.arctan2(dy, dx) - x_true[2])
    mask = dist < max_range
    if not np.any(mask):
        return np.zeros((0, 2))
    z_d = dist[mask] + np.random.randn(mask.sum()) * noise_std
    z_b = angle_mod(bearing[mask] + np.random.randn(mask.sum()) * LIDAR_SIGMA_BEARING)
    return np.column_stack([z_d, z_b])


# === Phase 3: Covariance ellipse ===

def _cov_ellipse(P_lm, n_std=2.0, n_pts=30):
    vals, vecs = np.linalg.eigh(P_lm)
    vals = np.maximum(vals, 1e-12)
    angle = np.arctan2(vecs[1, 0], vecs[0, 0])
    t = np.linspace(0, 2 * np.pi, n_pts)
    ellipse = n_std * np.sqrt(vals) * np.column_stack([np.cos(t), np.sin(t)])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return ellipse @ R.T


# === Phase 4: Main function ===

def main():
    dt = UKF_DT
    max_range = LIDAR_RANGE_MAX
    ref_traj = generate_reference_trajectory('figure8', dt)
    landmarks = generate_landmarks(n_lm=50, map_size=10.0)

    n_step = len(ref_traj)
    ekf = EKFSLAM(dt=dt, max_landmark_range=max_range)
    ekf.x_est[:3] = ref_traj[0]

    traj_est = np.zeros((n_step, 3))
    traj_est[0] = ref_traj[0]
    lm_est_hist = []

    v_cmd = 1.0
    delta_cmd = 0.1

    for i in range(1, n_step):
        dx = ref_traj[i, 0] - ref_traj[i - 1, 0]
        dy = ref_traj[i, 1] - ref_traj[i - 1, 1]
        d_theta = angle_mod(ref_traj[i, 2] - ref_traj[i - 1, 2])
        v_cmd = np.sqrt(dx**2 + dy**2) / dt
        delta_cmd = np.arctan2(d_theta * WHEELBASE, max(v_cmd * dt, 1e-9))

        ekf.predict(v_cmd, delta_cmd)
        z_obs = simulate_observation(ref_traj[i], landmarks, max_range, LIDAR_SIGMA_RANGE)
        if z_obs.shape[0] > 0:
            ekf.update(z_obs)

        robot_pose, lm_est = ekf.get_state()
        traj_est[i] = robot_pose
        lm_est_hist.append(lm_est.copy())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('EKF-SLAM Demo')

    ax.plot(landmarks[:, 0], landmarks[:, 1], 'go', markersize=6, label='True landmarks')
    ln_true, = ax.plot([], [], 'k-', linewidth=1.5, label='True trajectory')
    ln_est, = ax.plot([], [], 'b--', linewidth=1.5, label='Estimated trajectory')
    ln_lm_est, = ax.plot([], [], 'rx', markersize=6, label='Estimated landmarks')
    ax.legend(loc='upper left', frameon=True, fancybox=True)

    skip = max(1, n_step // 300)

    def _init():
        ln_true.set_data([], [])
        ln_est.set_data([], [])
        ln_lm_est.set_data([], [])
        return ln_true, ln_est, ln_lm_est

    def _update(frame):
        idx = min(frame * skip, n_step - 1)
        ln_true.set_data(ref_traj[:idx + 1, 0], ref_traj[:idx + 1, 1])
        ln_est.set_data(traj_est[:idx + 1, 0], traj_est[:idx + 1, 1])
        if idx > 0 and len(lm_est_hist) > 0:
            hist_idx = min(idx - 1, len(lm_est_hist) - 1)
            lm = lm_est_hist[hist_idx]
            if lm.shape[0] > 0:
                ln_lm_est.set_data(lm[:, 0], lm[:, 1])
                while ax.patches:
                    ax.patches[0].remove()
                n_lm_show = min(lm.shape[0], 50)
                for j in range(n_lm_show):
                    lm_idx_state = 3 + 2 * j
                    if lm_idx_state + 1 < ekf.n_active:
                        P_lm = ekf.P_est[lm_idx_state:lm_idx_state + 2, lm_idx_state:lm_idx_state + 2]
                    else:
                        P_lm = np.eye(2) * 0.01
                    ell = _cov_ellipse(P_lm, n_std=2.0)
                    ell += lm[j]
                    poly = plt.Polygon(ell, fill=False, edgecolor='red', linewidth=0.5, alpha=0.5)
                    ax.add_patch(poly)
        return ln_true, ln_est, ln_lm_est

    n_frames = n_step // skip + 1

    if show_animation:
        ani = animation.FuncAnimation(fig, _update, frames=n_frames,
            init_func=_init, blit=False, interval=20, repeat=False)
    else:
        _update(n_frames - 1)

    fig.tight_layout()
    fig_dir = pathlib.Path(__file__).parent.parent / 'figs'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_dir / 'slam_ekf_demo.png'), dpi=150)
    print(f"[save] EKF-SLAM demo saved to {fig_dir / 'slam_ekf_demo.png'}")
    plt.show()


if __name__ == '__main__':
    main()
