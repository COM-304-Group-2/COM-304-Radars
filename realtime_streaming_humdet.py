# Enhanced Real-Time Streaming with Human Detection
import sys
import warnings
import os
import time
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from streaming.prod_dca import producer_real_time_1843

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

# Parameters for Background Subtraction
ALPHA = 0.05  # Learning rate for background update
THRESHOLD = 0.02  # Threshold for movement detection

# Parameters for Clustering
NUM_CLUSTERS = 3  # Maximum number of moving objects

# Background Model
background = None

# Adaptive Background Subtraction
def adaptive_bg_subtraction(frame):
    global background
    if background is None:
        background = frame.astype(float)
    else:
        background = ALPHA * frame + (1 - ALPHA) * background
    diff = np.abs(frame - background)
    return diff

# K-means Clustering
def apply_clustering(data):
    if data.size == 0:
        return []
    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(data)), random_state=0).fit(data)
    return kmeans.cluster_centers_

# Plot Detected Clusters
def plot_clusters(ax, clusters):
    for cluster in clusters:
        ax.plot(cluster[1], cluster[0], 'ro')

# Main Visualization and Detection
def plot_2d_heatmap(ax, data, theta, r):
    R, Theta = np.meshgrid(r, theta)
    ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=0, vmax=0.1)
    ax.set_xlim(theta[0], theta[-1])
    ax.set_ylim(r[0], r[-1])
    ax.grid(False)

# Main Function for Streaming and Detection
def consumer(q, index):
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Real-Time Human Detection")

    while True:
        if not q.empty():
            msg = q.get()
            if msg[0] == "bev":
                phi, r_idxs, frame = msg[1]
                processed_frame = adaptive_bg_subtraction(frame)
                processed_frame = gaussian_filter(processed_frame, sigma=2)
                moving_points = np.argwhere(processed_frame > THRESHOLD)
                clusters = apply_clustering(moving_points)

                ax.clear()
                plot_2d_heatmap(ax, processed_frame, phi, r_idxs)
                plot_clusters(ax, clusters)
                plt.pause(0.01)


def main():
    q = Queue(maxsize=1)
    p = Process(target=producer_real_time_1843, args=(q, 0, '1843_config.lua'))
    p.start()
    consumer(q, 0)

if __name__ == "__main__":
    main()
