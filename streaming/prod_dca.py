import numpy as np
import queue
from streaming.mmwave.dataloader.adc_modified import DCA1000

def producer_real_time_1843(q, index, lua_file):
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

            org_data = dca.organize(raw_frame, num_chirps, num_tx, num_rx, samples_per_chirp)

            # --- Range FFT ---
            range_fft = np.fft.fft(org_data, axis=-1)
            range_fft = range_fft[:, :, :samples_per_chirp // 2]

            # --- Angle FFT (Rx) ---
            angle_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)
            range_angle_map = np.abs(angle_fft)
            range_angle_map = 20 * np.log10(range_angle_map + 1e-6)

            # 1D beamformed signal
            beamform = np.abs(np.sum(range_fft, axis=(0, 1)))
            beamform_full = np.zeros(samples_per_chirp)
            beamform_full[:samples_per_chirp // 2] = beamform

            # Put in queue if not overloaded
            if q.qsize() < 5:
                q.put(("rfft", beamform_full), timeout=0.05)
                q.put(("range_angle", range_angle_map.mean(axis=0)), timeout=0.05)

    except KeyboardInterrupt:
        print("🔴 Streaming stopped by user.")
    except Exception as e:
        print("❌ Error during streaming:", e)
    finally:
        dca.close()
