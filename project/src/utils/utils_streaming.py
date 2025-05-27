import numpy as np


def cart2pol(x_flat, y_flat):
    phi_flat = np.arctan2(y_flat, x_flat)
    r_flat = np.hypot(x_flat, y_flat)

    return np.column_stack((phi_flat, r_flat))