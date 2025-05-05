import numpy as np
import queue
import time
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from streaming.mmwave.dataloader.adc_modified import DCA1000
from streaming import utils

def cfar_ca_2d(power_map, num_train_range=10, num_train_doppler=8, num_guard_range=2, num_guard_doppler=2, rate_fa=1e-5):
    Tr, Td = num_train_range, num_train_doppler
    Gr, Gd = num_guard_range, num_guard_doppler
    Wr, Wd = Tr + Gr, Td + Gd

    Nwin = (2*Wr+1)*(2*Wd+1)
    Nguard = (2*Gr+1)*(2*Gd+1)
    Ntrain = Nwin - Nguard

    kernel_win = np.ones((2*Wr+1, 2*Wd+1))
    kernel_guard = np.ones((2*Gr+1, 2*Gd+1))

    sum_win = convolve2d(power_map, kernel_win, mode='same', boundary='fill', fillvalue=0)
    sum_guard = convolve2d(power_map, kernel_guard, mode='same', boundary='fill', fillvalue=0)
    sum_train = sum_win - sum_guard

    noise_level = sum_train / float(Ntrain)
    alpha = Ntrain * (rate_fa ** (-1.0 / Ntrain) - 1.0)
    threshold = alpha * noise_level

    return power_map > threshold  # boolean mask

def process_frame(raw_data, cfar_params):
    N_ant, N_chirps, N_adc = raw_data.shape
    rng_ffted = np.fft.fft(raw_data, axis=2)
    rd_cube = np.fft.fft(rng_ffted, axis=1)
    rd_map = np.mean(np.abs(rd_cube) ** 2, axis=0)
    dets = cfar_ca_2d(rd_map,
                      cfar_params["num_train_r"],
                      cfar_params["num_train_d"],
                      cfar_params["num_guard_r"],
                      cfar_params["num_guard_d"],
                      cfar_params["threshold_scale"])
    return dets

def beamform_2d_s(beat_freq_data, phi_s, phi_e, phi_res, theta_s, theta_e, theta_res,
                  x_locs, z_locs, r_idxs, radar_params, index, dets):
    sample_rate = radar_params["sample_rate"]
    slope = radar_params["slope"]
    lm = radar_params["lm"]

    phi = np.arange(phi_s, phi_e, phi_res) * np.pi / 180
    theta = np.arange(theta_s, theta_e, theta_res) * np.pi / 180
    num_phi, num_theta = len(phi), len(theta)

    sph_pwr = np.zeros((num_phi, num_theta, len(r_idxs)), dtype=np.complex64)

    d_idx, r_idx = np.nonzero(dets)
    r_idx_set = set(r_idxs)

    for d, r in zip(d_idx, r_idx):
        if r not in r_idx_set:
            continue
        try:
            r_rel = np.where(r_idxs == r)[0][0]  # local index in the trimmed range dimension
        except IndexError:
            continue
        if d >= beat_freq_data.shape[1] or r_rel >= beat_freq_data.shape[2]:
            continue

        beat = beat_freq_data[:, d, r_rel]
        for i, angle_phi in enumerate(phi):
            for j, angle_theta in enumerate(theta):
                proj = x_locs * np.sin(angle_theta) * np.cos(angle_phi)
                phase_shifts = np.exp((1j * 2 * np.pi / lm) * proj)
                beamformed_signal = beat * phase_shifts
                sph_pwr[i, j, r_rel] += np.abs(np.sum(beamformed_signal))

    return sph_pwr, phi, theta

def producer_real_time_1843(q, index, lua_file):
    num_tx, num_rx, adc_samples = 3, 4, 512
    chirp_loops = 16
    slope, sample_rate, c = 70.150e6, 10e6, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 140)

    radar_params = {
        "sample_rate": sample_rate,
        "num_samples": adc_samples,
        "slope": slope,
        "lm": lm,
        "num_tx": num_tx,
        "num_rx": num_rx,
        "adc_samples": adc_samples
    }

    num_virtual_ant = num_tx * num_rx
    x_locs, z_locs, _ = utils.get_ant_pos_2d(num_virtual_ant, num_tx, num_rx)

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    frame_count = 0
    bg_accumulator = None
    bg_buffer_len = 5  # every 5 frames update the static background

    try:
        while True:
            raw = dca.read(timeout=0.5, chirps=chirp_loops, rx=num_rx, tx=num_tx, samples=adc_samples)
            if raw is None or not q.empty():
                continue

            raw = dca.organize(raw, chirp_loops, num_tx, num_rx, adc_samples)
            adc_windowed = raw * np.hamming(adc_samples)

            beat_freq_data = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            beat_freq_data = beat_freq_data.transpose(1, 2, 0, 3).reshape(num_virtual_ant, chirp_loops, adc_samples)

            dets = process_frame(beat_freq_data, {
                "num_train_r": 10,
                "num_train_d": 8,
                "num_guard_r": 2,
                "num_guard_d": 2,
                "threshold_scale": 1e-3
            })

            range_fft = np.fft.fft(beat_freq_data, axis=-1)
            range_fft_subset = range_fft[:, :, r_idxs]

            bf_output, phi, theta = beamform_2d_s(
                range_fft_subset, 0, 180, 2, 70, 110, 2,
                x_locs[:, 0], z_locs, r_idxs, radar_params, 0, dets
            )
            bf_output = np.abs(bf_output)
            bf_output = median_filter(bf_output, size=(1, 1, 1))

            frame_count += 1

            if bg_accumulator is None:
                bg_accumulator = np.zeros_like(bf_output)

            if frame_count % bg_buffer_len == 0:
                # Update background model every 5 frames
                bg_accumulator = 0.8 * bg_accumulator + 0.2 * bf_output

            # Subtract static background
            bf_output -= bg_accumulator
            bf_output[bf_output < 0] = 0

            to_plot = np.sum(bf_output, axis=1)
            if np.max(to_plot) > 0:
                to_plot /= np.max(to_plot)
                to_plot = to_plot ** 2
            else:
                to_plot[:] = 0

            try:
                q.put_nowait(("bev", (phi, r_idxs, to_plot)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()
