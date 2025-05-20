import numpy as np
import queue
import time
from scipy.ndimage import median_filter


from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

#from project.src.gtrack.config import Detection


from gtrack.config import Detection


################# Change the values based on how much of the azimuth angles you want to see and the resolution ##################
# Define field of view in degrees that you want to process in theta, phi and range bins
def beamform_2d_s(beat_freq_data, phi_s, phi_e, phi_res, theta_s, theta_e, theta_res, x_locs, z_locs, r_idxs, radar_params, index, dets):
    """
    Performs 2D beamforming along the azimuth (horizontal) dimension, this results in a bird eye view image.
    - beat_freq_data: beat data AKA the range FFT (size: num_x_stps * num_z_stps * num TX * num RX, num ADC samples)
    - phi_s: first azimuth angle that you want to start computing
    - phi_e: last azimuth angle that you want to compute
    - phi_res: resolution of the azimuth angles you want to compute
    - theta_s: first elevation angle that you want to start computing
    - theta_e: last elevation angle that you want to compute
    - theta_res: resolution of the elevation angles you want to compute
    - x_locs: x coordinate of antenna locations
    - z_locs: z coordinate of antenna locations
    - r_idx: range bins to calculate
    - radar_parms: radar_params if needed

    Returns:
    - sph_pwr: beamformed result (size: n_phi, n_theta, n_range)
    - phi: array of azimuth angles
    - theta: array of elevation angles
    """

    # Radar parameters
    lm = radar_params["lm"]
    fs       = radar_params["sample_rate"]  # [Hz]
    num_samps= radar_params["num_range"]  # ADC samples per chirp
    N_dop    = radar_params["num_doppler"]                           # chirps per frame
    lam      = radar_params["lm"]           # wavelength [m]

    ## Compute velocity resolution
    # 1) chirp duration (neglecting idle time)
    T_chirp = num_samps / fs                # [s]

    # 2) PRF
    PRF     = 1.0 / T_chirp                 # [Hz]

    # 3) velocity‐resolution
    vel_res = lam/2 * PRF / N_dop           # [m/s per Doppler bin]

    # Convert angles to radians
    phi = np.arange(phi_s, phi_e, phi_res) * np.pi / 180
    num_phi = len(phi)

    angles = x_locs * np.cos(phi[:, np.newaxis])
    phase_shifts = np.exp((1j * 2 * np.pi / lm) * angles)

    r_idx, d_idx = np.nonzero(dets)

    # Initialize output
    sph_pwr = np.zeros((num_phi, r_idxs.shape[0]), dtype=np.complex64)

    detections = []


    for d, r in zip(r_idx, d_idx):

        beat = beat_freq_data[:, d, r]
        beamformed_signal = beat[np.newaxis, :] * phase_shifts
        sph_pwr[:, r] = np.maximum(sph_pwr[:, r], np.abs(np.sum(beamformed_signal, axis=-1)))

        snr = np.abs(np.sum(beamformed_signal, axis=-1))**6 ## rajouter variance? #shape (num_phi)

        rang = np.repeat(r, num_phi)
        #v = (d - N_dop/2) * vel_res
        v = 0
        v_all = np.repeat(v, num_phi)

        small_detection = [
        Detection(r_m, az, v, snr)
        for r_m, az, v, snr in zip(rang, phi, v_all, snr)
        ]

        detections.extend(small_detection)

    return sph_pwr, detections

def cfar_ca_2d(power_map,
               num_train_range: int = 10,
               num_train_doppler: int = 8,
               num_guard_range: int = 2,
               num_guard_doppler: int = 2,
               rate_fa: float = 1e-3):
    """
    2D Cell-Averaging CFAR on a (range × Doppler) power map.

    Parameters
    ----------
    power_map : 2D np.ndarray
        The incoherent power map |X|^2 over (range, Doppler).
    num_train_range : int
        # of training cells on each side in range
    num_train_doppler : int
        # of training cells on each side in Doppler
    num_guard_range : int
        # of guard cells on each side in range
    num_guard_doppler : int
        # of guard cells on each side in Doppler
    rate_fa : float
        Desired probability of false alarm

    Returns
    -------
    detection_map : 2D bool np.ndarray
        True where power_map exceeds the CFAR threshold.
    """
    Tr, Td = num_train_range, num_train_doppler
    Gr, Gd = num_guard_range, num_guard_doppler

    # full window half–sizes
    Wr = Tr + Gr
    Wd = Td + Gd

    # number of training cells total
    Nwin = (2*Wr+1)*(2*Wd+1)
    Nguard = (2*Gr+1)*(2*Gd+1)
    Ntrain = Nwin - Nguard

    # build convolution kernels
    kernel_win   = np.ones((2*Wr+1, 2*Wd+1), dtype=float)
    kernel_guard = np.ones((2*Gr+1,2*Gd+1), dtype=float)

    # sum over full window
    sum_win   = convolve2d(power_map, kernel_win,   mode='same', boundary='fill', fillvalue=0)
    # sum over guard+CUT region
    sum_guard = convolve2d(power_map, kernel_guard, mode='same', boundary='fill', fillvalue=0)

    # training‐cell sum = window minus guard (which includes the CUT)
    sum_train = sum_win - sum_guard

    # noise estimate (average of training cells)
    noise_level = sum_train / float(Ntrain)

    # CFAR threshold multiplier (cell–averaging formula)
    alpha = Ntrain * (rate_fa**(-1.0/Ntrain) - 1.0)
    threshold = alpha * noise_level

    # detection mask
    return power_map > threshold


def process_frame(raw_data, cfar_params):
    """
    Full pipeline for one frame:
      raw_data : np.array, shape (N_ant, N_adc, N_chirps)
      radar_params : dict with at least "fc" (Hz)
      x_locs, z_locs : 1D arrays of antenna x,z positions (meters)
      phi_*, theta_* : angle scan bounds/resolution (degrees)
      cfar_params : dict with keys
        num_train_r, num_train_d,
        num_guard_r, num_guard_d,
        threshold_scale

    returns
        dets : list of (r_idx, d_idx) tuples
    """
    N_ant, N_adc, N_chirps = raw_data.shape

    # 1) Range FFT
    #rng_ffted = np.fft.fft(raw_data, axis=2)   # → (N_ant, N_adc, N_R=N_chirps)

    # 2) Doppler FFT
    rd_cube = np.fft.fft(raw_data, axis=1)    # → (N_ant, N_D=N_adc, N_R=N_chirps)

    # 3) Build RD magnitude for CFAR (average across antennas)
    rd_map = np.mean(np.abs(rd_cube)**2, axis=0)  # shape (N_R, N_D)

    # 4) CFAR detections
    dets = cfar_ca_2d(rd_map,
                    cfar_params["num_train_r"],
                    cfar_params["num_train_d"],
                    cfar_params["num_guard_r"],
                    cfar_params["num_guard_d"],
                    cfar_params["threshold_scale"])

    return dets

