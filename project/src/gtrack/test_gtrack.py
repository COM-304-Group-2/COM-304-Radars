import unittest
import numpy as np
from types import SimpleNamespace

from module import GTrackModule2D
from config import GTrackConfig2D
from config import PresenceZone2D


class TestGTrack2D(unittest.TestCase):
    def test_constant_velocity(self):
        cfg = GTrackConfig2D(
            max_points=10, max_tracks=2, dt=1.0,
            process_noise=0.1, meas_noise_range=1.0, meas_noise_az=0.01,
            gating_threshold=9.21, alloc_range_gate=1.0, alloc_az_gate=0.1,
            alloc_vel_gate=1.0, min_cluster_points=1, alloc_snr_threshold=0.0,
            init_state_cov=100.0, det_to_active_count=2,
            det_to_free_count=2, act_to_free_count=2,
            presence_zones=[PresenceZone2D(-100,100,-100,100)],
            pres_on_count=1, pres_off_count=1
        )
        tracker = GTrackModule2D(cfg)
        # simulate a target moving along x
        for k in range(5):
            true_x = 0 + 2*k
            true_y = 0
            r = np.hypot(true_x, true_y)
            az = np.arctan2(true_y, true_x)
            dop = 2.0
            pt = SimpleNamespace(range=r, azimuth=az, doppler=dop, snr=10.0)
            res = tracker.step([pt])
            tracks = res['tracks']
            if k>=1:
                self.assertTrue(len(tracks)>=1)
                pos = tracks[0]['pos']
                self.assertAlmostEqual(pos[0], true_x, places=1)
                self.assertAlmostEqual(pos[1], true_y, places=1)

if __name__ == '__main__':
    unittest.main()