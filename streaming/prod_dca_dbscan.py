from sklearn.cluster import DBSCAN
import numpy as np
import queue
import time
from scipy.ndimage import median_filter
from streaming.mmwave.dataloader.adc_modified import DCA1000
from streaming import utils

def producer_real_time_1843(q, index, lua_file):
    num_tx, num_rx, adc_samples = 3, 4, 512
    slope, sample_rate, c = 70.150e6, 10e6, 3e8
    lm = c / 77e9

    r_idxs = np.arange(0, 140)
    phi = np.deg2rad(np.arange(0, 180, 2))
    theta = np.deg2rad(np.arange(70, 110, 2))

    print("Starting DCA1000...")
    dca = DCA1000()
    print("Reading data...")

    try:
        while True:
            start_time = time.time()

            raw = dca.read(timeout=0.5)
            if raw is None:
                continue

            beat_freq_data = np.fft.fft(raw, axis=-1)
            power_map = np.abs(beat_freq_data) ** 2

            # Get thresholded points
            points_thresh = np.argwhere(power_map > 0.05)
            if points_thresh.size == 0:
                continue

            # Perform DBSCAN
            db = DBSCAN(eps=2.5, min_samples=90).fit(points_thresh)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Detected {n_clusters} clusters (people).")

            # Prepare data for plotting
            clusters = []
            for label in set(labels):
                if label == -1:
                    continue  # Noise
                cluster_pts = points_thresh[labels == label]
                centroid = np.mean(cluster_pts, axis=0)
                clusters.append((label, cluster_pts, centroid))

            # Send clusters to the queue
            try:
                if not q.full():
                    q.put_nowait(("bev", clusters))
            except queue.Full:
                continue

            # Frame rate monitoring
            frame_processing_time = time.time() - start_time
            print(f"Frame processed in {frame_processing_time:.4f} seconds")

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        dca.close()
