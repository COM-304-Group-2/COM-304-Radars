import numpy as np
from .utilities_2d import (sph2cart_2d, cart2sph_2d,
                           compute_mahalanobis_2d,
                           calc_gating_limits_2d, wrap_angle)
from .config import GTrackConfig2D


class GTrackUnit2D:
    """
    GTrackUnit2D represents a single tracking unit in the 2D ground tracking algorithm.
    """
    def __init__(self, cfg: GTrackConfig2D, F: np.ndarray, Q: np.ndarray):
        self.cfg = cfg
        self.F = F
        self.Q = Q
        self.uid = None

        self.state = np.zeros(cfg.state_dim)
        self.P = np.eye(cfg.state_dim) * cfg.init_state_cov
        self.apriori_state = np.zeros_like(self.state)
        self.apriori_P = np.zeros_like(self.P)

        self.H = np.zeros((cfg.meas_dim, cfg.state_dim))
        self.S = np.zeros((cfg.meas_dim, cfg.meas_dim))
        self.S_inv = np.zeros_like(self.S)

        self.status = 'FREE'
        self.hit_count = 0
        self.miss_count = 0

        self.dim = np.zeros(2)
        self.confidence = 0.0

    def predict(self):
        """
        Predict the next state and measurement matrix for the tracking unit.
        """

        if self.status != 'ACTIVE':
            self.apriori_state = self.state.copy()
            self.apriori_P = self.P.copy()
        else:
            self.apriori_state = self.F @ self.state
            self.apriori_P = self.F @ self.P @ self.F.T + self.Q

        x, y, vx, vy = self.apriori_state
        r = np.hypot(x, y)
        if r < 1e-6:
            return
        self.H = np.array([
            [x / r,      y / r,     0,  0],
            [-y / (r * r), x / (r * r), 0,  0],
        ], dtype=float)

        R = np.diag([self.cfg.meas_noise_range, self.cfg.meas_noise_az])
        self.S, self.S_inv = calc_gating_limits_2d(self.apriori_P, self.H, R)

    def score(self, idx, point, best_score, best_id, second_score):
        """
        Calculate the Mahalanobis score for a given point and update the best and second best scores.

        Parameters
        ----------
        idx : int
            The index of the point in the list of points.
        point : Detection
            The detection point with range and azimuth attributes.
        best_score : np.ndarray
            Array of best scores for each point.
        best_id : np.ndarray
            Array of best IDs for each point.
        second_score : np.ndarray
            Array of second best scores for each point.
        """

        z = np.array([point.range, point.azimuth])
        r_pred, az_pred = cart2sph_2d(self.apriori_state[0], self.apriori_state[1])

        residual = z - np.array([r_pred, wrap_angle(az_pred)])
        residual[1] = wrap_angle(residual[1])

        m2 = compute_mahalanobis_2d(residual, self.S_inv)
        if m2 < self.cfg.gating_threshold:
            if m2 < best_score[idx]:
                second_score[idx] = best_score[idx]
                best_score[idx] = m2
                best_id[idx] = self.uid
            elif m2 < second_score[idx]:
                second_score[idx] = m2

    def start(self, cluster):
        """
        Initialize the tracking unit with a cluster of points.

        Parameters
        ----------
        cluster : list of Detection
            List of detected points that form a cluster.
        """

        zs = np.array([[pt.range, pt.azimuth] for pt in cluster])
        mean_r, mean_az = zs.mean(axis=0)

        x, y = sph2cart_2d(mean_r, mean_az)

        vx = 0
        vy = 0

        self.state = np.array([x, y, vx, vy], dtype=float)
        self.P = np.eye(self.cfg.state_dim) * self.cfg.init_state_cov
        self.apriori_state = self.state.copy()
        self.apriori_P = self.P.copy()
        self.status = 'DETECTION'
        self.hit_count = 0
        self.miss_count = 0

    def update(self, points):
        """
        Update the tracking unit with new points and compute the new state.

        Parameters
        ----------
        points : list of Detection
            List of detected points, each with range and azimuth attributes.
        """

        assigned = [pt for pt in points if pt.assigned_id == self.uid]
        if not assigned:
            self.miss_count += 1
            self.event()
            return
        zs = np.array([[pt.range, pt.azimuth] for pt in assigned])
        mean_z = zs.mean(axis=0)
        r_pred, az_pred = cart2sph_2d(self.apriori_state[0], self.apriori_state[1])

        residual = mean_z - np.array([r_pred, az_pred])
        residual[1] = wrap_angle(residual[1])

        K = self.apriori_P @ self.H.T @ self.S_inv
        self.state = self.apriori_state + K @ residual
        I = np.eye(self.cfg.state_dim)
        self.P = (I - K @ self.H) @ self.apriori_P
        self.hit_count += 1
        self.miss_count = 0
        self.dim = np.array([np.ptp(zs[:, 0]), np.ptp(zs[:, 1])])
        self.confidence = min(1.0, self.hit_count / max(1, self.cfg.det_to_active_count))
        self.event()

    def event(self):
        """
        Update the status of the tracking unit based on hit and miss counts.
        """

        c = self.cfg
        if self.status == 'DETECTION':
            if self.hit_count >= c.det_to_active_count:
                self.status = 'ACTIVE'
                self.miss_count = 0
            elif self.miss_count >= c.det_to_free_count:
                self.status = 'FREE'
        elif self.status == 'ACTIVE' and self.miss_count >= c.act_to_free_count:
            self.status = 'FREE'

    def report(self):
        """
        Generate a report of the current state of the tracking unit.

        Returns
        -------
        dict
            A dictionary containing the tracking unit's unique ID, position, velocity, covariance, dimensions, confidence, and status.
        """

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
        """
        Stop the tracking unit by reinitializing it with the original configuration.
        """

        uid = self.uid
        self.__init__(self.cfg, self.F, self.Q)
        self.uid = uid
