import numpy as np

from config import GTrackConfig2D
from units import GTrackUnit2D
from math import *
from utilities_2d import *

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
        q = cfg.process_noise
        q11 = (dt**4)/4 * q
        q13 = (dt**3)/2 * q
        q33 = dt**2 * q
        Q = np.array([[q11,0,q13,0],[0,q11,0,q13],[q13,0,q33,0],[0,q13,0,q33]], dtype=float)
        return F, Q

    def step(self, points, variances=None):
        self.heartbeat += 1
        pts = points[:min(len(points), self.config.max_points)]
        for u in list(self.active):
            u.predict()
        self._associate(pts)
        self._allocate(pts)
        for u in list(self.active):
            u.update(pts)
            if u.status == 'FREE':
                self._reclaim(u)
        self._presence()
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
        seeds = [pt for pt in points if getattr(pt, 'assigned_id', -1) == -1 and getattr(pt, 'snr', 0) > 0]
        seeds.sort(key=lambda pt: getattr(pt, 'snr', 0), reverse=True)
        for seed in seeds:
            if getattr(seed, '_clustered', False):
                continue
            cluster = [seed]
            seed._clustered = True
            queue = [seed]
            total_snr = getattr(seed, 'snr', 0)
            while queue:
                cur = queue.pop(0)
                for pt in points:
                    if getattr(pt, '_clustered', False) or getattr(pt, 'assigned_id', -1) != -1:
                        continue
                    dr = abs(pt.range - seed.range)
                    da = abs(pt.azimuth - seed.azimuth)
                    dv = abs(pt.doppler - seed.doppler)
                    if dr <= cfg.alloc_range_gate and da <= cfg.alloc_az_gate and dv <= cfg.alloc_vel_gate:
                        pt._clustered = True
                        queue.append(pt)
                        cluster.append(pt)
                        total_snr += getattr(pt, 'snr', 0)
            for pt in cluster:
                if hasattr(pt, '_clustered'):
                    del pt._clustered
            if len(cluster) < cfg.min_cluster_points or total_snr < cfg.alloc_snr_threshold:
                continue
            if not self.free:
                break
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


