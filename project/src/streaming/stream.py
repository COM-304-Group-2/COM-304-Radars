from . import realtime_streaming_config
from . import realtime_streaming
from gtrack.config import (GTrackConfig2D)
import numpy as np


def main():
    """
    Main function to start the real-time radar streaming and processing.
    """

    # Parameters for the range-azimuth beamforming
    r_idxs = np.arange(0, 100, 1)
    phi = np.deg2rad(np.arange(0, 180, 1))
    width = 40 # azimuth width in degrees

    # Offsets for the radars
    offset_x_1 = 0.0  # x offset for the first radar
    offset_x_2 = 0.0  # x offset for the second radar
    offset_y_1 = 0.0  # y offset for the first radar
    offset_y_2 = 0.0  # y offset for the second radar

    # Radar  parameters
    cfg_radar = {
        "range_idx": r_idxs,
        "phi": phi,
        "width": width,
        "offset_x_1": offset_x_1,
        "offset_x_2": offset_x_2,
        "offset_y_1": offset_y_1,
        "offset_y_2": offset_y_2,
        "n_radar": 2,
        "num_tx": 3,
        "num_rx": 4,
        "num_doppler": 16,
        "num_range": 992,
        "sample_rate": 5166000,
        "c": 3e8,
        "lm": 3e8 / 77e9,
        "slope": 70.150e6
    }

    # Parameters for CFAR
    cfg_cfar = {
        "num_train_r": 10,
        "num_train_d": 8,
        "num_guard_r": 2,
        "num_guard_d": 2,
        "threshold_scale": 1e-3
    }

    # Parameters for Gtrack
    cfg_gtrack = GTrackConfig2D(
        max_points=200,  # max detections per frame
        max_tracks=5,  # max simultaneous tracks
        dt=0.6,  # time between frames (s)
        process_noise=0.5,  # Q spectral density
        meas_noise_range=2.0,  # σ² range noise (m²)
        meas_noise_az=1,  # σ² azimuth noise (rad²)
        gating_threshold=16,  # ≈95% gate for 2-DOF chi²
        alloc_range_gate=0.5,  # cluster gate (m)
        alloc_az_gate=np.deg2rad(10),  # cluster gate (rad)
        alloc_vel_gate=20,  # cluster gate (m/s)
        min_cluster_points=10,  # you can increase if you want multi-point seeds
        alloc_snr_threshold=1,  # sum-SNR threshold
        min_snr_threshold=0.01,  # min SNR for new track
        init_state_cov=1.0,  # starting P for new tracks
        det_to_active_count=12,  # hits needed to go ACTIVE
        det_to_free_count=3,  # misses to drop DETECTION
        act_to_free_count=8,  # misses to drop ACTIVE
        presence_zones=[],  # e.g. [PresenceZone2D(-10,10,-5,5)]
        pres_on_count=5, # frames to confirm presence on
        pres_off_count=3 # frames to confirm presence off
    )

    print("⌛️ Starting streaming...")

    # Start the streaming process
    realtime_streaming.main(cfg_radar, cfg_gtrack, cfg_cfar)

if __name__ == "__main__":
    main()