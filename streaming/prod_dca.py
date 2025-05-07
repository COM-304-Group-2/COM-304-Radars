import numpy as np
import queue
import time
from scipy.ndimage import median_filter
from streaming.mmwave.dataloader.adc_modified import DCA1000
from streaming import utils

from gtrack.tracker import Tracker

tracker = Tracker(dt=0.1, process_noise=0.1, measurement_noise=1.0)

def beamform_2d(beat_freq_data, phi, theta, x_locs, z_locs, r_idxs, radar_params):
    lm = radar_params['lm']
    num_phi, num_theta, num_r = len(phi), len(theta), len(r_idxs)
    sph_pwr = np.zeros((num_phi, num_theta, num_r), dtype=np.complex64)

    # Flatten antenna grid
    x_grid, z_grid = np.meshgrid(x_locs.flatten(), z_locs.flatten(), indexing='ij')
    x_flat = x_grid.ravel()
    z_flat = z_grid.ravel()
    
    for i, angle_phi in enumerate(phi):
        for j, angle_theta in enumerate(theta):
            proj = x_locs * np.sin(angle_theta) * np.cos(angle_phi)
            phase_shifts = np.exp((1j * 2 * np.pi / lm) * proj)
            
            beamformed_signal = beat_freq_data[:, r_idxs] * phase_shifts[:,:]
        
            sph_pwr[i, j, :] = np.abs(np.sum(beamformed_signal, axis=0))

    return sph_pwr


def producer_real_time_1843(q, index, lua_file):
    num_tx, num_rx, adc_samples = 3, 4, 512
    chirp_loops = 1  # mmWave studio sends 3 chirps per TX
    slope, sample_rate, c = 70.150e6, 10e6, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 140)
    phi = np.deg2rad(np.arange(0, 180, 2))
    theta = np.deg2rad(np.arange(70, 110, 2))

    radar_params = {
        "sample_rate": sample_rate,
        "num_samples": adc_samples,
        "slope": slope,
        "lm": lm,
        "num_z_stp": num_tx,
        "num_rx": num_rx,
        "adc_samples": adc_samples
    }

    num_virtual_ant = num_tx * num_rx
    x_locs, z_locs, _ = utils.get_ant_pos_2d(num_virtual_ant, num_tx, num_rx)

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    try:
        while True:
            raw = dca.read(timeout=0.5)
            if raw is None:
                continue
            beat_freq_data = np.fft.fft(raw, axis=-1)
            power_map = np.abs(beat_freq_data) ** 2
            detections = np.argwhere(power_map > 0.05)  # Threshold

            tracks = tracker.update_tracks(detections)
            track_positions = [(t['position'][0], t['position'][1]) for t in tracks]

            try:
                q.put_nowait(("bev", track_positions))
            except queue.Full:
                continue
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()

    