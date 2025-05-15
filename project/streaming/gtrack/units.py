# gtrack/unit.py (methods rewritten for 2D)

from .utilities_2d import (sph2cart_2d, cart2sph_2d,
                            compute_mahalanobis_2d,
                            calc_gating_limits_2d)
import numpy as np
from .config import GTrackConfig2D

class GTrackUnit2D:
    def __init__(self, cfg: GTrackConfig2D, F: np.ndarray, Q: np.ndarray):
        self.cfg = cfg
        self.F = F
        self.Q = Q
        self.uid = None
        # state vectors
        self.state = np.zeros(cfg.state_dim)
        self.P = np.eye(cfg.state_dim) * cfg.init_state_cov
        self.apriori_state = np.zeros_like(self.state)
        self.apriori_P = np.zeros_like(self.P)
        # measurement-space
        self.H = np.zeros((cfg.meas_dim, cfg.state_dim))
        self.S = np.zeros((cfg.meas_dim, cfg.meas_dim))
        self.S_inv = np.zeros_like(self.S)
        # track lifecycle
        self.status = 'FREE'
        self.hit_count = 0
        self.miss_count = 0
        # diagnostics
        self.dim = np.zeros(2)
        self.confidence = 0.0

    def predict(self):
        if self.status != 'ACTIVE':
            self.apriori_state = self.state.copy()
            self.apriori_P = self.P.copy()
        else:
            self.apriori_state = self.F @ self.state
            self.apriori_P = self.F @ self.P @ self.F.T + self.Q
        # build measurement Jacobian, guard r=0
        x, y, vx, vy = self.apriori_state
        r = np.hypot(x, y)
        if r < 1e-6:
            # avoid divide by zero: leave H unchanged
            return
        self.H = np.array([
            [x / r,      y / r,     0,  0],
            [-y / (r * r), x / (r * r), 0,  0],
        ], dtype=float)
        self.S, self.S_inv = calc_gating_limits_2d(self.apriori_P, self.H)

    def score(self, idx, point, best_score, best_id, second_score):
        # measurement vector
        z = np.array([point.range, point.azimuth])
        # predicted measurement
        r_pred, az_pred = cart2sph_2d(self.apriori_state[0], self.apriori_state[1])
        residual = z - np.array([r_pred, az_pred])
        m2 = compute_mahalanobis_2d(residual, self.S_inv)
        if m2 < self.cfg.gating_threshold:
            if m2 < best_score[idx]:
                second_score[idx] = best_score[idx]
                best_score[idx] = m2
                best_id[idx] = self.uid
            elif m2 < second_score[idx]:
                second_score[idx] = m2

    def start(self, cluster):
        # compute centroid in measurement space
        zs = np.array([[pt.range, pt.azimuth] for pt in cluster])
        mean_r, mean_az = zs.mean(axis=0)
        # initialize state
        x, y = sph2cart_2d(mean_r, mean_az)
        seed = max(cluster, key=lambda pt: getattr(pt, 'snr', 0))
        v = seed.doppler
        vx = v * np.cos(seed.azimuth)
        vy = v * np.sin(seed.azimuth)
        self.state = np.array([x, y, vx, vy], dtype=float)
        self.P = np.eye(self.cfg.state_dim) * self.cfg.init_state_cov
        self.status = 'DETECTION'
        self.hit_count = 1
        self.miss_count = 0

    def update(self, points):
        assigned = [pt for pt in points if pt.assigned_id == self.uid]
        if not assigned:
            self.miss_count += 1
            self.event()
            return
        zs = np.array([[pt.range, pt.azimuth] for pt in assigned])
        mean_z = zs.mean(axis=0)
        r_pred, az_pred = cart2sph_2d(self.apriori_state[0], self.apriori_state[1])
        residual = mean_z - np.array([r_pred, az_pred])
        K = self.apriori_P @ self.H.T @ self.S_inv
        self.state = self.apriori_state + K @ residual
        I = np.eye(self.cfg.state_dim)
        self.P = (I - K @ self.H) @ self.apriori_P
        self.hit_count += 1
        self.dim = np.array([np.ptp(zs[:, 0]), np.ptp(zs[:, 1])])
        self.confidence = min(1.0, self.hit_count / max(1, self.cfg.det_to_active_count))
        self.event()

    def event(self):
        c = self.cfg
        if self.status == 'DETECTION':
            if self.hit_count >= c.det_to_active_count:
                self.status = 'ACTIVE'
            elif self.miss_count >= c.det_to_free_count:
                self.status = 'FREE'
        elif self.status == 'ACTIVE' and self.miss_count >= c.act_to_free_count:
            self.status = 'FREE'

    def report(self):
        return {
            'uid': self.uid,
            'pos': self.state[:2].copy(),
            'vel': self.state[2:].copy(),
            'cov': np.diag(self.P).copy(),
            'dim': self.dim.copy(),
            'confidence': self.confidence,
            'status': self.status
        }

    def stop(self):
        self.__init__(self.cfg, self.F, self.Q)
