# Real-Time Human Detection with Corrected Cluster Plotting and Unified Heatmap
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
MIN_CLUSTER_INTENSITY = 0.05  # Increased to reduce noise

# Kalman Filter Parameters
PROCESS_NOISE = 1e-2  # Process noise for position update
MEASUREMENT_NOISE = 1e-1  # Measurement noise

# Background Model
background = None

# Trackers
tracker = {}
next_id = 0

# Adaptive Background Subtraction
# Remove close-to-origin noise by hardcoding small radius values to zero
def remove_close_origin_noise(data, r, threshold=5):
    # Create a 2D mask with the same shape as data
    R, _ = np.meshgrid(r, np.arange(data.shape[0]))
    # Zero out data where radius is below the threshold
    data[R < threshold] = 0
    return data

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
            clusters.append(centroid)
    return clusters

# Plot Heatmap using Polar Projection
def plot_2d_heatmap(ax, data, theta, r):
    R, Theta = np.meshgrid(r, theta)
    ax.pcolormesh(Theta, R, data, shading='auto', cmap='jet', vmin=0, vmax=np.max(data))
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.max(r))
    ax.grid(False)
    return R, Theta

# Plot Clusters using Unified Polar Projection
def plot_clusters(ax, clusters, R, Theta):
    stats_text = []
    for obj_id, obj in tracker.items():
        r_index = int(obj['pos'][0])
        theta_index = int(obj['pos'][1])
        r_value = R[r_index, theta_index]
        theta_value = np.degrees(Theta[r_index, theta_index])  # Convert theta to degrees
        ax.plot(np.radians(theta_value), r_value, 'ro')
        ax.text(np.radians(theta_value), r_value, str(obj_id), color='white', fontsize=12)

        # Calculate speed in polar coordinates
        delta_r = np.abs(obj['x'][0] - obj['pos'][0])
        delta_theta = np.abs(obj['x'][1] - obj['pos'][1])
        speed = np.sqrt((delta_r) ** 2 + (r_value * np.radians(delta_theta)) ** 2)

        # Ensure speed is in m/s and calculate agility
        agility = np.abs(speed - obj.get('last_speed', 0))
        obj['last_speed'] = speed

        # Append formatted statistics with correct units
        stats_text.append(f"ID {obj_id}:\nr={r_value:.2f} m\nθ={theta_value:.2f}°\nspeed={speed:.2f} m/s\nagility={agility:.2f}")

    # Create a frame on the left for the statistics
    ax.text(-0.3, 1, "\n\n".join(stats_text), transform=ax.transAxes, fontsize=10, 
            color='black', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))


# Update Tracker with ID Management
def update_tracker(clusters):
    global tracker, next_id
    new_tracker = {}
    if len(clusters) == 0:
        tracker = {}
        next_id = 0
        return

    cost_matrix = np.zeros((len(tracker), len(clusters)))
    for i, obj in enumerate(tracker.values()):
        for j, cluster in enumerate(clusters):
            cost_matrix[i, j] = np.linalg.norm(obj['pos'] - cluster)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = set()
    id_map = {}
    for r, c in zip(row_ind, col_ind):
        obj_id = list(tracker.keys())[r]
        id_map[obj_id] = c
        matched.add(c)

    # Reassign IDs from 0 to n-1
    sorted_ids = sorted(id_map.keys())
    for new_id, old_id in enumerate(sorted_ids):
        new_tracker[new_id] = tracker[old_id]
        new_tracker[new_id]['pos'] = clusters[id_map[old_id]]
    next_id = len(new_tracker)

    # Add new clusters with new IDs
    for j, cluster in enumerate(clusters):
        if j not in matched:
            new_tracker[next_id] = {'pos': cluster, 'P': PROCESS_NOISE, 'x': cluster}
            next_id += 1

    tracker = new_tracker

# Main Function for Streaming and Detection
def consumer(q, index):
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Real-Time Human Detection with Unified Cluster Plotting")

    while True:
        if not q.empty():
            msg = q.get()
            if msg[0] == "bev":
                phi, r_idxs, frame = msg[1]
                processed_frame = adaptive_bg_subtraction(frame)
                processed_frame = remove_close_origin_noise(processed_frame, r_idxs)
                processed_frame = gaussian_filter(processed_frame, sigma=2)
                moving_points = np.argwhere(processed_frame > THRESHOLD)
                clusters = apply_dbscan(moving_points)
                update_tracker(clusters)

                ax.clear()
                R, Theta = plot_2d_heatmap(ax, processed_frame, phi, r_idxs)
                plot_clusters(ax, clusters, R, Theta)
                plt.pause(0.01)


def main():
    q = Queue(maxsize=1)
    p = Process(target=producer_real_time_1843, args=(q, 0, '1843_config.lua'))
    p.start()
    consumer(q, 0)

if __name__ == "__main__":
    main()
