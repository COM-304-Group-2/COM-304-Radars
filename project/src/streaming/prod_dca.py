import numpy as np
import queue
import time

from mmwave.dataloader.adc import DCA1000
from gtrack.config import (GTrackConfig2D)
from gtrack.module import GTrackModule2D
from processing.processing import process_frame, beamform_2d_s
from utils.utils import get_ant_pos_2d
from processing.processing import compute_dbscan


def producer_real_time_1843(q, cfg_radar, cfg_cfar, config_port, data_port, static_ip, system_ip):
    """
    Producer function for real-time data acquisition from the DCA1000 connected to the AWR1843 radar.

    Parameters
    ----------
    q : queue.Queue
        The queue to which the processed data will be sent.
    cfg_radar : dict
        Configuration parameters for the radar, including range indices, number of transmitters, receivers, chirp loops, and ADC samples.
    cfg_cfar : dict
        Configuration parameters for the CFAR processing, including number of training and guard cells, and threshold scale.
    config_port : str
        The port for the DCA1000 configuration.
    data_port : str
        The port for the DCA1000 data.
    static_ip : str
        The static IP address for the DCA1000.
    system_ip : str
        The system IP address.
    """

    # Parameters
    r_idxs = cfg_radar["range_idx"]
    num_tx = cfg_radar["num_tx"]
    num_rx = cfg_radar["num_rx"]
    chirp_loops = cfg_radar["num_doppler"]
    adc_samples = cfg_radar["num_range"]

    last_frame = np.zeros((num_rx * num_tx, chirp_loops, adc_samples), dtype=np.complex64)

    # Get the antenna positions
    x_locs, _, _ = get_ant_pos_2d(num_tx*num_rx, adc_samples, num_rx)

    # Setup the DCA1000
    print("⌛️ Starting producer for DCA1000 with ip " + static_ip + " and system ip " + system_ip)
    dca = DCA1000(config_port=config_port, data_port=data_port, static_ip=static_ip, system_ip=system_ip)
    print("✅ DCA1000 initialized.")

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

            # Update the last frame
            last_frame = beat_freq_data

            # Substract the last frame and keep only the corresponding range indices
            range_fft_s = range_fft - last_frame_fft
            range_fft_s = range_fft_s[:, :, r_idxs]

            # Set the static range indices to zero
            range_fft_s[:, :, 0:20] = 0
            range_fft_s[:, :, 120:150] = 0

            # Compute CFAR
            dets = process_frame(range_fft_s, cfg_cfar)

            # Compute beamforming
            bf_output = beamform_2d_s(range_fft_s, cfg_radar, x_locs[:,0], dets)

            # Send the data to the queue
            try:
                q.put_nowait(("bev", (bf_output)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print("🛑 Producer for DCA1000 with ip " + static_ip + " and system ip " + system_ip + " stopped by user.")
    finally:
        dca.close()