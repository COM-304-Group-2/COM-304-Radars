import numpy as np
import time
from streaming.mmwave.dataloader.adc_modified import DCA1000

def producer_real_time_1843(q, index, lua_file):
    # Configuration Parameters - match your .lua
    num_rx = 4
    num_tx = 3
    chirp_loops = 1
    samples_per_chirp = 512
    num_chirps = num_tx * chirp_loops

    print("Starting DCA1000...")
    dca = DCA1000()
    
    print("Reading data...")
    try:
        while True:
            raw_frame = dca.read(timeout=1.0, chirps=num_chirps, rx=num_rx, tx=num_tx, samples=samples_per_chirp)
            if raw_frame is None:
                print("No frame received.")
                continue

            # Organize to shape (num_chirps * num_tx, num_rx, num_samples)
            org_data = dca.organize(raw_frame, num_chirps, num_tx, num_rx, samples_per_chirp)

            # Basic Beamforming (range-FFT only)
            fft_out = np.fft.fft(org_data, axis=-1)
            beamform = np.abs(np.sum(fft_out, axis=(0, 1)))  # sum over chirps and antennas

            beamform = np.abs(beamform[:512])
            q.put(("rfft", beamform))

    except KeyboardInterrupt:
        print("Streaming stopped by user.")
    except Exception as e:
        print("Error during streaming:", e)
    finally:
        dca.close()
