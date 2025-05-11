import numpy as np
import queue
import time
from scipy.ndimage import median_filter
from streaming.mmwave.dataloader.adc_modified import DCA1000
import utils


def beamform_2d(beat_freq_data, phi, theta, x_locs, z_locs, r_idxs, radar_params):
    lm = radar_params['lm']
    num_phi, num_theta, num_r = len(phi), len(theta), len(r_idxs)
    sph_pwr = np.zeros((num_phi, num_theta, num_r), dtype=np.complex64)

    # Flatten antenna grid
    #x_grid, z_grid = np.meshgrid(x_locs.flatten(), z_locs.flatten(), indexing='ij')
    #x_flat = x_grid.ravel()
    #z_flat = z_grid.ravel()

    for i, angle_phi in enumerate(phi):
        for j, angle_theta in enumerate(theta):
            proj = x_locs * np.sin(angle_theta) * np.cos(angle_phi)
            phase_shifts = np.exp((1j * 2 * np.pi / lm) * proj)

            beamformed_signal = beat_freq_data[:, r_idxs] * phase_shifts[:, :]

            sph_pwr[i, j, :] = np.abs(np.sum(beamformed_signal, axis=0))

    return sph_pwr


def producer_real_time_1843(q, index, lua_file):
    num_tx, num_rx, adc_samples = 3, 4, 512
    chirp_loops = 1  # mmWave studio sends 3 chirps per TX
    slope, sample_rate, c = 70.150e6, 10e6, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 120)
    phi = np.deg2rad(np.arange(0, 180, 1))
    theta = np.deg2rad(np.arange(70, 110, 1))

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

    last_frame = np.zeros((12, 512), dtype=np.complex64)

    try:
        while True:
            raw = dca.read(timeout=0.5, chirps=chirp_loops, rx=num_rx, tx=num_tx, samples=adc_samples)
            if raw is None:
                continue
            if not q.empty():
                continue

            raw = dca.organize(raw, chirp_loops, num_tx, num_rx, adc_samples) # shape = (chirp_loops*tx, rx, samples)

            adc_windowed = raw * np.hamming(adc_samples)

            # ✅ Transpose to (tx, rx, chirp, sample) and reshape to (12, 512)
            beat_freq_data = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            beat_freq_data = beat_freq_data.transpose(1, 2, 0, 3)
            beat_freq_data = beat_freq_data[:,:,0,:]
            beat_freq_data = beat_freq_data.reshape(12, 512)


            range_fft = np.fft.fft(beat_freq_data, axis=-1)
            last_frame_fft = np.fft.fft(last_frame, axis=-1)

            #range_fft_s = range_fft - last_frame_fft
            #last_frame = beat_freq_data

            bf_output = beamform_2d(range_fft, phi, theta, x_locs, z_locs, r_idxs, radar_params)

            bf_output = np.abs(bf_output)
            bf_output = median_filter(bf_output, size=(1, 1, 1))

            #threshold = np.percentile(bf_output, 99)
            #bf_output = np.where(bf_output > threshold, bf_output, 0)

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