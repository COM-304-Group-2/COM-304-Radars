# gtrack/utilities_2d.py
import numpy as np

def sph2cart_2d(r, az):
    return np.array([r * np.cos(az), r * np.sin(az)])

def cart2sph_2d(x, y):
    r = np.hypot(x, y)
    az = np.arctan2(y, x)
    return r, az

def calc_gating_limits_2d(P, H, R=None):
    if R is None:
        R = np.diag([1.0, 1.0])
    S = H @ P @ H.T + R
    return S, np.linalg.inv(S)

def compute_mahalanobis_2d(residual, S_inv):
    return float(residual.T @ S_inv @ residual)
