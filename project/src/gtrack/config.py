from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PresenceZone2D:
    """
    Represents a 2D presence zone for occupancy tracking.
    """
    x_min: float # minimum x-coordinate of the zone
    x_max: float # maximum x-coordinate of the zone
    y_min: float # minimum y-coordinate of the zone
    y_max: float # maximum y-coordinate of the zone


@dataclass
class GTrackConfig2D:
    """
    Configuration for the 2D ground track algorithm.
    """
    max_points: int
    max_tracks: int
    dt: float                       # time step between frames (s)
    process_noise: float            # continuous white-noise spectral density
    meas_noise_range: float         # range measurement variance (m^2)
    meas_noise_az: float            # azimuth measurement variance (rad^2)
    gating_threshold: float         # chi-squared gating limit
    alloc_range_gate: float         # clustering gate: range difference (m)
    alloc_az_gate: float            # clustering gate: angle difference (rad)
    alloc_vel_gate: float           # clustering gate: velocity difference (m/s)
    min_cluster_points: int         # minimum detections to form a new track
    alloc_snr_threshold: float      # minimum summed SNR for a new track
    min_snr_threshold: float        # minimum SNR for a point
    init_state_cov: float           # initial covariance for new tracks
    det_to_active_count: int        # DETECTION->ACTIVE transition hit count
    det_to_free_count: int          # DETECTION->FREE transition miss count
    act_to_free_count: int          # ACTIVE->FREE transition miss count
    presence_zones: List[PresenceZone2D]  # occupancy zones
    pres_on_count: int              # frames to confirm presence on
    pres_off_count: int             # frames to confirm presence off

    @property
    def state_dim(self): return 4  # [x, y, vx, vy]
    @property
    def meas_dim(self): return 2   # [range, azimuth]


@dataclass
class Detection:
    """
    Represents a detection in the 2D ground track algorithm.
    """
    def __init__(self, r, az, v, snr):
        self.range   = r # range in meters
        self.azimuth = az # azimuth in radians
        self.doppler = v # doppler velocity in m/s
        self.snr     = snr # signal-to-noise ratio