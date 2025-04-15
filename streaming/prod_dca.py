import numpy as np
import queue
import time
from scipy.ndimage import median_filter
from streaming.mmwave.dataloader.adc_modified import DCA1000
from streaming import utils

def beamform_2d(beat_freq_data, phi, theta, x_locs, z_locs, r_idxs, radar_params):
    lm = radar_params['lm']
    num_phi, num_theta, num_r = len(phi), len(theta), len(r_idxs)
    sph_pwr = np.zeros((num_phi, num_theta, num_r), dtype=np.complex64)

    x_grid, z_grid = np.meshgrid(x_locs.flatten(), z_locs.flatten(), indexing='ij')
    ant_coords = np.stack((x_grid.ravel(), z_grid.ravel()), axis=1)

    for i, angle_phi in enumerate(phi):
        for j, angle_theta in enumerate(theta):
            proj = (ant_coords[:, 0] * np.sin(angle_theta) * np.cos(angle_phi) +
                    ant_coords[:, 1] * np.cos(angle_theta))
            phase_shifts = np.exp((1j * 2 * np.pi / lm) * proj)
            shifted = beat_freq_data[:, r_idxs] * phase_shifts[:, np.newaxis]
            sph_pwr[i, j, :] = np.abs(np.sum(shifted, axis=0))

    return sph_pwr

def producer_real_time_1843(q, index, lua_file):
    num_rx, num_tx, adc_samples, chirp_loops = 4, 3, 512, 1
    num_chirps = num_tx * chirp_loops
    slope, sample_rate, c = 70.150e6, 10e6, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 140)
    phi = np.linspace(0, np.pi, 180)
    theta = np.deg2rad(np.arange(80, 100, 1))

    radar_params = {
        "sample_rate": sample_rate,
        "num_samples": adc_samples,
        "slope": slope,
        "lm": lm,
        "num_z_stp": num_tx,
        "num_rx": num_rx,
        "adc_samples": adc_samples
    }

    num_x_stp = num_tx * num_rx
    x_locs, z_locs, _ = utils.get_ant_pos_2d(num_x_stp, num_tx, num_rx)

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    try:
        while True:
            raw_frame = dca.read(timeout=0.5, chirps=num_chirps, rx=num_rx, tx=num_tx, samples=adc_samples)
            if raw_frame is None:
                continue

            # ⏱️ Skip frame if queue not empty
            if not q.empty():
                continue

            org_data = dca.organize(raw_frame, num_chirps, num_tx, num_rx, adc_samples)
            adc_windowed = org_data * np.hamming(adc_samples)
            beat_freq_data = adc_windowed.reshape(-1, adc_samples)
            range_fft = np.fft.fft(beat_freq_data, axis=-1)

            bf_output = beamform_2d(range_fft, phi, theta, x_locs, z_locs, r_idxs, radar_params)
            bf_output = np.abs(bf_output)
            bf_output = median_filter(bf_output, size=(1, 1, 1))

            to_plot = np.sum(bf_output, axis=1)
            to_plot /= np.max(to_plot)
            to_plot = to_plot ** 2

            try:
                q.put_nowait(("bev", (phi, r_idxs, to_plot)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()
