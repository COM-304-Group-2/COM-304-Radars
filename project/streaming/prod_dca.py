from numpy.lib.function_base import unwrap
from mmwave.dataloader import DCA1000
import numpy as np
import time
from scipy.fftpack import fft
import processing.utility as utility

def producer_real_time_1843(q, index, lua_file):
    """
    This function is one of the processed called in realtime_streaming.py.
    Paramters:
    - q: Queue to push new data
    - index: dummy index
    - lua_file: lua file containing radar chirp paramters

    This function reads data from the ethernet port connected to the radar, reformats the data, and pushes the range fft to q.
    Note: This function can be called in parallel with other functions (see realtime_streaming.py for more details). 
    """
    num_rx, num_tx, samples_per_chirp, periodicity, num_frames, chirp_loops, data_rate, freq_plot_len, range_plot_len = utility.read_radar_params(lua_file) 
    dca = DCA1000()
    # chirp configurations based on your config lua file
    dca.sensor_config(chirps=num_tx*chirp_loops, chirp_loops=1, num_rx=num_rx, num_samples=samples_per_chirp)
    prev_time = time.time()
    while True:
        # here we actually reads from the ethernet port
        adc_data = dca.read()
        # reorganize the interleaved data into channels corresponding to (tx*chirp loops, rx, adc samples)
        # number of frames is 1 because we are always reading the 
        org_data = dca.organize(raw_frame=adc_data, num_chirps=num_tx*chirp_loops,
        num_rx=num_rx, num_samples=samples_per_chirp, num_frames=1, model='1843')
        x = 0
        
        # here we are just processing the range fft to plot later on
        x = np.sum(org_data, axis=(0, 1))
        x -= np.mean(x, axis=-1, keepdims=True)
        fx = fft(x, axis=-1)
        afx = np.squeeze(np.abs(fx))

        now = time.time()
        # So as to not overload the upating, we will just refresh the data every 0.1 seconds
        if now - prev_time > 0.1:
            # put the tuple containing the data's name and the data into the queue that will be sent to plotting (see realtime_streaming.py)
            q.put(["rfft", afx])
            prev_time = now