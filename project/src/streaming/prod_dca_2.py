import numpy as np
import queue
import time

from mmwave.dataloader.adc_modified import DCA1000
from gtrack.config import (GTrackConfig2D)
from gtrack.module import GTrackModule2D
from processing.processing import process_frame, beamform_2d_s
from utils.utils import get_ant_pos_2d
from processing.processing import compute_dbscan


def producer_real_time_1843(q, cfg_radar, cfg_gtrack, db=False, gtrack=True):


    r_idxs = cfg_radar["range_idx"]
    phi = cfg_radar["phi"]
    num_tx = cfg_radar["num_tx"]
    num_rx = cfg_radar["num_rx"]
    chirp_loops = cfg_radar["num_doppler"]
    adc_samples = cfg_radar["num_range"]
    sample_rate = cfg_radar["sample_rate"]
    c = cfg_radar["c"]


    x_locs, _, _ = get_ant_pos_2d(num_tx*num_rx, adc_samples, num_rx)

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    last_frame = np.zeros((num_rx*num_tx, chirp_loops, adc_samples), dtype=np.complex64)

    tracker = GTrackModule2D(cfg_gtrack)

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


            bf_output, detection = beamform_2d_s(range_fft_s[:,:, r_idxs], cfg_radar, x_locs[:,0], dets)



            snrs = np.array([d.snr for d in detection])
            snrs_max = np.max(snrs)


            detection_tuned = []
            for d in detection:
                d_t = d
                d_t.snr = d_t.snr / snrs_max
                detection_tuned.append(d_t)

            detection = [d for d in detection_tuned if d.snr >= 0.5]




            bf_output = np.abs(bf_output)
            #bf_output = median_filter(bf_output, size=(1, 1, 1))

            to_plot = bf_output
            to_plot /= np.max(to_plot)
            to_plot = to_plot ** 6

            if db:
                db_output = compute_dbscan(to_plot, r_idxs, phi)


            output_det = tracker.step(detection)

            try:
                q.put_nowait(("bev", (phi, r_idxs, to_plot, output_det)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()