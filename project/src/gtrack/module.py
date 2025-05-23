import numpy as np
import time
from math import *
from sklearn.cluster import DBSCAN

from .config import GTrackConfig2D
from .units import GTrackUnit2D
from .utilities_2d import *

class GTrackModule2D:
    def __init__(self, config: GTrackConfig2D):
        self.config = config
        self.F, self.Q = self._build_matrices(config)
        self.units = [GTrackUnit2D(config, self.F, self.Q) for _ in range(config.max_tracks)]
        for uid, u in enumerate(self.units):
            u.uid = uid
        self.active = []
        self.free = list(self.units)
        self.heartbeat = 0
        self.presence_flag = False
        self.pres_on_count = 0
        self.pres_off_count = 0

    def _build_matrices(self, cfg: GTrackConfig2D):
        dt = cfg.dt
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        F[2,2] = F[3,3] = 0.97
        q = cfg.process_noise
        q11 = (dt**4)/4 * q
        q13 = (dt**3)/2 * q
        q33 = dt**2 * q
        Q = np.array([[q11,0,q13,0],[0,q11,0,q13],[q13,0,q33,0],[0,q13,0,q33]], dtype=float)
        return F, Q

    def step(self, points, variances=None):
        time_1 = time.time()
        self.heartbeat += 1
        pts = points[:min(len(points), self.config.max_points)]
        for u in list(self.active):
            u.predict()

        time_2 = time.time()
        self._associate(pts)
        time_3 = time.time()
        self._allocate(pts)
        time_4 = time.time()

        for u in list(self.active):
            u.update(pts)
            if u.status == 'FREE':
                self._reclaim(u)

        time_5 = time.time()
        self._presence()
        time_6 = time.time()
        #print(f"Time taken predict: {time_2-time_1:.4f}, associate: {time_3-time_2:.4f}, allocate: {time_4-time_3:.4f}, update: {time_5-time_4:.4f}, presence: {time_6-time_5:.4f}")
        return {'tracks': [u.report() for u in self.active], 'presence': self.presence_flag}

    def _associate(self, points):
        n = len(points)
        best_score = [np.inf] * n
        best_id = [-1] * n
        second_score = [np.inf] * n
        for u in self.active:
            for i, pt in enumerate(points):
                u.score(i, pt, best_score, best_id, second_score)
        for i, pt in enumerate(points):
            if best_score[i] < self.config.gating_threshold:
                pt.assigned_id = best_id[i]
                pt.is_unique = (second_score[i] > self.config.gating_threshold)
            else:
                pt.assigned_id = -1
                pt.is_unique = False

    def _allocate(self, points):
        cfg = self.config
        # Select unassigned seeds
        seeds = [pt for pt in points if pt.assigned_id == -1]
        if not self.free or len(seeds) < cfg.min_cluster_points:
            return

        # Build normalized feature array
        X = np.array([[pt.range / cfg.alloc_range_gate,
                       pt.azimuth / cfg.alloc_az_gate,
                       pt.doppler / cfg.alloc_vel_gate]
                      for pt in seeds])

        # Cluster using DBSCAN
        db = DBSCAN(eps=1.0,
                    min_samples=cfg.min_cluster_points,
                    metric='euclidean',
                    n_jobs=-1).fit(X)
        labels = db.labels_

        # allocate each cluster above the SNR threshold
        for lab in set(labels):
            if lab == -1 or not self.free:
                continue
            idxs = np.where(labels == lab)[0]
            total_snr = sum(seeds[i].snr for i in idxs)
            if total_snr < cfg.alloc_snr_threshold:
                continue
            cluster = [seeds[i] for i in idxs]
            unit = self.free.pop(0)
            unit.start(cluster)
            self.active.append(unit)
            for pt in cluster:
                pt.assigned_id = unit.uid
                pt.is_unique = True

    def _reclaim(self, unit):
        self.active.remove(unit)
        unit.stop()
        self.free.append(unit)

    def _presence(self):
        present = False
        for u in self.active:
            x, y = u.state[0], u.state[1]
            for z in self.config.presence_zones:
                if z.x_min <= x <= z.x_max and z.y_min <= y <= z.y_max:
                    present = True
                    break
            if present:
                break
        if present:
            self.pres_off_count = 0
            self.pres_on_count += 1
            if self.pres_on_count >= self.config.pres_on_count:
                self.presence_flag = True
        else:
            self.pres_on_count = 0
            self.pres_off_count += 1
            if self.pres_off_count >= self.config.pres_off_count:
                self.presence_flag = False


