from sklearn.cluster import DBSCAN
import numpy as np

def cluster_points(points, eps=1.0, min_samples=3):
    if len(points) == 0:
        return np.array([])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_
