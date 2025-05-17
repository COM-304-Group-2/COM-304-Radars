# Real-Time Human Detection with Enhanced Kalman Filtering and DBSCAN Clustering
import sys
import warnings
import os
import time
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from streaming.prod_dca import producer_real_time_1843

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

# Parameters for Background Subtraction
ALPHA = 0.05  # Learning rate for background update
THRESHOLD = 0.02  # Threshold for movement detection

# Parameters for Clustering
DBSCAN_EPS = 5  # DBSCAN radius for neighborhood
DBSCAN_MIN_SAMPLES = 10  # Minimum points for a cluster

# Kalman Filter Parameters
PROCESS_NOISE = 1e-2  # Process noise for position update
MEASUREMENT_NOISE = 1e-1  # Measurement noise

# Background Model
background = None

# Kalman Filter Prediction and Update
def kalman_filter(x, P, z, R, Q):
    x_pred = x
    P_pred = P + Q
    y = z - x_pred
    S = P_pred + R
    K = P_pred / S
    x_new = x_pred + K * y
    P_new = (1 - K) * P_pred
    return x_new, P_new

# Trackers
tracker = {}
next_id = 0

# Adaptive Background Subtraction
def adaptive_bg_subtraction(frame):
    global background
    if background is None:
        background = frame.astype(float)
    else:
        background = ALPHA * frame + (1 - ALPHA) * background
    diff = np.abs(frame - background)
    return diff

# DBSCAN Clustering
def apply_dbscan(data):
    if data.size == 0:
        return []
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(data)
    clusters = []
    for label in set(clustering.labels_):
        if label != -1:  # Ignore noise points
            points = data[clustering.labels_ == label]
            centroid = np.mean(points, axis=0)
            #clusters.append(centroid) # Count IDs
    return clusters

# Update Tracker with Kalman Filtering
def update_tracker(clusters):
    global tracker, next_id
    new_tracker = {}
    if len(tracker) == 0:
        for cluster in clusters:
            new_tracker[next_id] = {'pos': cluster, 'P': PROCESS_NOISE, 'x': cluster}
            next_id += 1
    else:
        cost_matrix = np.zeros((len(tracker), len(clusters)))
        for i, obj in enumerate(tracker.values()):
            for j, cluster in enumerate(clusters):
                cost_matrix[i, j] = np.linalg.norm(obj['pos'] - cluster)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()
        for r, c in zip(row_ind, col_ind):
            obj_id = list(tracker.keys())[r]
            z = clusters[c]
            x, P = kalman_filter(tracker[obj_id]['x'], tracker[obj_id]['P'], z, MEASUREMENT_NOISE, PROCESS_NOISE)
            new_tracker[obj_id] = {'pos': x, 'P': P, 'x': x}
            matched.add(c)

        for j, cluster in enumerate(clusters):
            if j not in matched:
                new_tracker[next_id] = {'pos': cluster, 'P': PROCESS_NOISE, 'x': cluster}
                next_id += 1
    tracker = new_tracker

# Plot Detected Clusters with IDs
def plot_clusters(ax, clusters):
    for obj_id, obj in tracker.items():
        ax.plot(obj['pos'][1], obj['pos'][0], 'ro')
        ax.text(obj['pos'][1], obj['pos'][0], str(obj_id), color='white', fontsize=12)

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
    ax.set_title("Real-Time Human Detection with Enhanced Kalman Filter")

    while True:
        if not q.empty():
            msg = q.get()
            if msg[0] == "bev":
                phi, r_idxs, frame = msg[1]
                processed_frame = adaptive_bg_subtraction(frame)
                processed_frame = gaussian_filter(processed_frame, sigma=2)
                moving_points = np.argwhere(processed_frame > THRESHOLD)
                clusters = apply_dbscan(moving_points)
                update_tracker(clusters)

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
