import numpy as np
import queue
import time
#from ..mmwave.dataloader.adc_modified import DCA1000
from mmwave.dataloader.adc_modified import DCA1000


from gtrack.config import (GTrackConfig2D)
from gtrack.module import GTrackModule2D
from processing.processing import process_frame, beamform_2d_s
from utils.utils import get_ant_pos_2d

def producer_real_time_1843(q, index, lua_file):
    num_tx, num_rx, adc_samples = 3, 4, 992
    chirp_loops = 16  # mmWave studio sends 3 chirps per TX
    slope, sample_rate, c = 70.150e6, 5166000, 3e8
    #slope, sample_rate, c = 70.150e6, 3416000, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 150)
    phi = np.deg2rad(np.arange(0, 180, 1))
    #theta = np.deg2rad(np.arange(70, 110, 1))

    radar_params = {
        "sample_rate": sample_rate,
        "num_doppler": chirp_loops,
        "lm": lm,
        "num_tx": num_tx,
        "num_rx": num_rx,
        "num_range": adc_samples
    }

    num_virtual_ant = num_tx * num_rx
    x_locs, z_locs, _ = get_ant_pos_2d(num_tx*num_rx, adc_samples, num_rx)

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    last_frame = np.zeros((num_rx*num_tx, chirp_loops, adc_samples), dtype=np.complex64)

    cfg = GTrackConfig2D(
        max_points=300,  # max detections per frame
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
        init_state_cov=1.0,  # starting P for new tracks
        det_to_active_count=15,  # hits needed to go ACTIVE
        det_to_free_count=2,  # misses to drop DETECTION
        act_to_free_count=8,  # misses to drop ACTIVE
        presence_zones=[],  # e.g. [PresenceZone2D(-10,10,-5,5)]
        pres_on_count=5,
        pres_off_count=3
    )

    tracker = GTrackModule2D(cfg)

    t = 0
    count = 0

    try:
        while True:
            raw = dca.read(timeout=0.5, chirps=chirp_loops, rx=num_rx, tx=num_tx, samples=adc_samples)
            if raw is None:
                continue
            if not q.empty():
                continue

            t_prod = time.time()

            raw = dca.organize(raw, chirp_loops, num_tx, num_rx, adc_samples) # shape = (chirp_loops*tx, rx, samples)

            adc_windowed = raw * np.hamming(adc_samples)

            # ✅ Transpose to (tx, rx, chirp, sample) and reshape to (12, 512)
            beat_freq_data = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            beat_freq_data = beat_freq_data.transpose(1, 2, 0, 3)
            beat_freq_data = beat_freq_data.reshape(num_tx*num_rx, chirp_loops, adc_samples)

            #
            #beat_freq_data[:,:, 0:10] = 0

            range_fft = np.fft.fft(beat_freq_data, axis=-1)
            last_frame_fft = np.fft.fft(last_frame, axis=-1)

            range_fft_s = range_fft - last_frame_fft
            range_fft_s[:,:, 0:10] = 0
            range_fft_s[:, :, 100:150] = 0
            last_frame = beat_freq_data

            dets = process_frame(range_fft_s[:, :, r_idxs], {
                "num_train_r": 10,
                "num_train_d": 8,
                "num_guard_r": 2,
                "num_guard_d": 2,
                "threshold_scale": 1e-7
            })

            t_beam = time.time()

            bf_output, detection = beamform_2d_s(range_fft_s[:,:,r_idxs], 0, 180, 1, 70, 110, 1, x_locs[:,0], z_locs, r_idxs, radar_params, 0, dets)

            t_beam_2 = time.time()

            #print(t_beam_2 - t_beam)

            snrs = np.array([d.snr for d in detection])
            snrs_max = np.max(snrs)

            #print(len(detection))

            detection_tuned = []
            for d in detection:
                d_t = d
                d_t.snr = d_t.snr / snrs_max
                detection_tuned.append(d_t)

            detection = [d for d in detection_tuned if d.snr >= 0.5]

            #print(len(detection))


            bf_output = np.abs(bf_output)
            #bf_output = median_filter(bf_output, size=(1, 1, 1))

            to_plot = bf_output
            to_plot /= np.max(to_plot)
            to_plot = to_plot ** 6
            #output_top = to_plot

            # DBSCAN clustering
            # Build full coordinate grid
            #phi_rad_2d, r_idxs_2d = np.meshgrid(phi, r_idxs, indexing='ij')  # shape: (180, 140)

            #x_coords_m = np.cos(phi_rad_2d) * r_idxs_2d  # shape: (180, 140)
            #z_coords_m = np.sin(phi_rad_2d) * r_idxs_2d  # shape: (180, 140)

            # Flatten for DBSCAN
            #points = np.stack([x_coords_m.ravel(), z_coords_m.ravel()], axis=1)
            #powers = output_top.ravel()

            # Optional: keep only high-power points
            #threshold = np.percentile(powers, 98)
            #valid_mask = powers > threshold
            #points_thresh = points[valid_mask]

            # --- DBSCAN ---
            #db = DBSCAN(eps=3, min_samples=10).fit(points_thresh)

            current_time = time.time()

            #print(len(detection))

            output_det = tracker.step(detection)

            next_time = time.time()

            #t  += next_time - current_time
            #count += 1

            #print(next_time - current_time)

            t_prod_2 = time.time()
            #print(t_prod_2 - t_prod)


            try:
                q.put_nowait(("bev", (phi, r_idxs, to_plot, output_det)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()