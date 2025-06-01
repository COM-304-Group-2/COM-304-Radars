import numpy as np


def sph2cart_2d(r, az):
    """
    Convert spherical coordinates (r, azimuth) to Cartesian coordinates (x, y) in 2D.

    Parameters
    ----------
    r : float or np.ndarray
        The radial distance.
    az : float or np.ndarray
        The azimuth angle in radians.

    Returns
    -------
    np.ndarray
        The Cartesian coordinates as a 2D array [x, y].
    """

    return np.array([r * np.cos(az), r * np.sin(az)])

def cart2sph_2d(x, y):
    """
    Convert Cartesian coordinates (x, y) to spherical coordinates (r, azimuth) in 2D.

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate.
    y : float or np.ndarray
        The y-coordinate.

    Returns
    -------
    r : float or np.ndarray
        The radial distance.
    az : float or np.ndarray
        The azimuth angle in radians.
    """

    r = np.hypot(x, y)
    az = np.arctan2(y, x)
    return r, az

def calc_gating_limits_2d(P, H, R=None):
    """
    Calculate the gating limits for a 2D Kalman filter.

    Parameters
    ----------
    P : np.ndarray
        The state covariance matrix.
    H : np.ndarray
        The measurement matrix.
    R : np.ndarray, optional
        The measurement noise covariance matrix. If None, an error is raised.

    Returns
    -------
    S : np.ndarray
        The innovation covariance matrix.
    S_inv : np.ndarray
        The inverse of the innovation covariance matrix.
    """

    if R is None:
        raise ValueError("R must be provided")
    S = H @ P @ H.T + R
    return S, np.linalg.inv(S)

def compute_mahalanobis_2d(residual, S_inv):
    """
    Compute the Mahalanobis distance for a 2D residual vector.

    Parameters
    ----------
    residual : np.ndarray
        The residual vector (difference between predicted and observed values).
    S_inv : np.ndarray
        The inverse of the innovation covariance matrix.

    Returns
    -------
    float
        The Mahalanobis distance.
    """

    return float(residual.T @ S_inv @ residual)

def wrap_angle(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Parameters
    ----------
    angle : float or np.ndarray
        The angle in radians to be wrapped.

    Returns
    -------
    float or np.ndarray
        The wrapped angle in radians, within the range [-pi, pi].
    """

    return (angle + np.pi) % (2 * np.pi) - np.pi