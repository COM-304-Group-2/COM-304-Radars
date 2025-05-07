import numpy as np
from .kalman_filter import KalmanFilter
from .clustering import cluster_points

class Tracker:
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=1.0):
        self.tracks = []
        self.track_id_count = 0
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def add_track(self, position):
        kf = KalmanFilter(self.dt, self.process_noise, self.measurement_noise)
        kf.x[:2, 0] = position
        self.tracks.append({'id': self.track_id_count, 'kf': kf, 'age': 1})
        self.track_id_count += 1

    def update_tracks(self, detections):
        updated_tracks = []
        labels = cluster_points(detections)

        for label in set(labels):
            if label == -1:
                continue
            clustered_points = detections[labels == label]  # Renamed variable
            centroid = np.mean(clustered_points, axis=0)

            # Prediction and update
            for track in self.tracks:
                predicted = track['kf'].predict()
                updated = track['kf'].update(centroid[:, np.newaxis])
                track['age'] += 1
                track['position'] = updated[:2].flatten()
                updated_tracks.append(track)

        # Adding new tracks for unassociated clusters
        for label in set(labels):
            if label == -1:
                continue
            clustered_points = detections[labels == label]  # Renamed variable
            if not any(track['id'] == label for track in self.tracks):
                centroid = np.mean(clustered_points, axis=0)
                self.add_track(centroid)

        self.tracks = updated_tracks
        return self.tracks
