import numpy as np
import queue
import time

from mmwave.dataloader.adc_modified import DCA1000
from gtrack.config import (GTrackConfig2D)
from gtrack.module import GTrackModule2D
from processing.processing import process_frame, beamform_2d_s
from utils.utils import get_ant_pos_2d
from processing.processing import compute_dbscan


def producer_real_time_1843(q, cfg_radar, cfg_gtrack, cfg_cfar, gtrack):
    # Parameters
    r_idxs = cfg_radar["range_idx"]
    phi = cfg_radar["phi"]
    num_tx = cfg_radar["num_tx"]
    num_rx = cfg_radar["num_rx"]
    chirp_loops = cfg_radar["num_doppler"]
    adc_samples = cfg_radar["num_range"]

    last_frame = np.zeros((num_rx * num_tx, chirp_loops, adc_samples), dtype=np.complex64)
    gtrack_output = []

    # Initialize the GTrack module
    tracker = GTrackModule2D(cfg_gtrack)

    # Get the antenna positions
    x_locs, _, _ = get_ant_pos_2d(num_tx*num_rx, adc_samples, num_rx)

    # Setup the DCA1000
    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    try:
        while True:
            # Read data from DCA1000
            raw = dca.read(timeout=0.5, chirps=chirp_loops, rx=num_rx, tx=num_tx, samples=adc_samples)
            if raw is None:
                continue
            if not q.empty():
                continue

            # Reshape the data
            raw = dca.organize(raw, chirp_loops, num_tx, num_rx, adc_samples) # shape = (chirp_loops*tx, rx, samples)

            # Apply Hamming window
            adc_windowed = raw * np.hamming(adc_samples)

            # ✅ Reshape the data to (num_tx*num_rx, chirp_loops, adc_samples)
            beat_freq_data = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            beat_freq_data = beat_freq_data.transpose(1, 2, 0, 3)
            beat_freq_data = beat_freq_data.reshape(num_tx*num_rx, chirp_loops, adc_samples)

            # Apply FFT along the range dimension
            range_fft = np.fft.fft(beat_freq_data, axis=-1)
            last_frame_fft = np.fft.fft(last_frame, axis=-1)

            # Substract the last fram and 0 the static clutter
            range_fft_s = range_fft - last_frame_fft
            range_fft_s[:,:, 0:20] = 0
            range_fft_s[:, :, 120:150] = 0
            range_fft_s = range_fft_s[:, :, r_idxs]
            last_frame = beat_freq_data

            # Compute CFAR
            dets = process_frame(range_fft_s, cfg_cfar)

            # Compute beamforming
            bf_output, detection = beamform_2d_s(range_fft_s, cfg_radar, x_locs[:,0], dets)

            # Compute the SNR
            snrs = np.array([d.snr for d in detection])
            snrs_max = np.max(snrs)
            detection_tuned = []
            for d in detection:
                d_t = d
                d_t.snr = d_t.snr / snrs_max
                detection_tuned.append(d_t)

            # Keep only the strong detections
            detection = [d for d in detection_tuned if d.snr >= cfg_gtrack.min_snr_threshold]

            # Normalize the output
            bf_output = np.abs(bf_output)
            #bf_output = median_filter(bf_output, size=(1, 1, 1))
            to_plot = bf_output
            to_plot /= np.max(to_plot)
            to_plot = to_plot ** 8

            # Compute GTrack (optional)
            if gtrack:
                gtrack_output = tracker.step(detection)

            # Send the data to the queue
            try:
                q.put_nowait(("bev", (to_plot, gtrack_output)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()