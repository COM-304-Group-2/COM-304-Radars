from . import realtime_streaming
from gtrack.config import (GTrackConfig2D)
import numpy as np


def main():

    # Parameters for the range-azimuth beamforming
    r_idxs = np.arange(0, 150)
    phi = np.deg2rad(np.arange(0, 180, 1))

    # Radar  parameters
    cfg_radar = {
        "range_idx": r_idxs,
        "phi": phi,
        "num_tx": 3,
        "num_rx": 4,
        "num_doppler": 16,
        "num_range": 992,
        "sample_rate": 5166000,
        "c": 3e8,
        "lm": 3e8 / 77e9,
        "slope": 70.150e6,
    }

    # Parameters for CFAR
    cfg_cfar = {
        "num_train_r": 10,
        "num_train_d": 8,
        "num_guard_r": 2,
        "num_guard_d": 2,
        "threshold_scale": 1e-7
    }

    # Parameters for Gtrack
    cfg_gtrack = GTrackConfig2D(
        max_points=200,  # max detections per frame
        max_tracks=5,  # max simultaneous tracks
        dt=0.5,  # time between frames (s)
        process_noise=0.5,  # Q spectral density
        meas_noise_range=1.0,  # σ² range noise (m²)
        meas_noise_az=1,  # σ² azimuth noise (rad²)
        gating_threshold=16,  # ≈95% gate for 2-DOF chi²
        alloc_range_gate=1,  # cluster gate (m)
        alloc_az_gate=np.deg2rad(10),  # cluster gate (rad)
        alloc_vel_gate=20,  # cluster gate (m/s)
        min_cluster_points=10,  # you can increase if you want multi-point seeds
        alloc_snr_threshold=2,  # sum-SNR threshold
        min_snr_threshold=0.5,  # min SNR for new track
        init_state_cov=1.0,  # starting P for new tracks
        det_to_active_count=15,  # hits needed to go ACTIVE
        det_to_free_count=2,  # misses to drop DETECTION
        act_to_free_count=8,  # misses to drop ACTIVE
        presence_zones=[],  # e.g. [PresenceZone2D(-10,10,-5,5)]
        pres_on_count=5, # frames to confirm presence on
        pres_off_count=3 # frames to confirm presence off
    )

    # Start the streaming process
    realtime_streaming.main(cfg_radar, cfg_gtrack, cfg_cfar)

if __name__ == "__main__":
    main()