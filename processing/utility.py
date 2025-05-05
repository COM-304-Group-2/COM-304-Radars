      
import os
import numpy as np

def read_radar_params(lua_script):
    """
    Helper function that reads the radar paramters from a input lua script.
    It assumes that the lua script has the following parameters input to variables 
        (chirp loops, number rx, number tx, adc samples, periodicity, number of frames).
    Variables should be named as listed below.
    Paramters:
    - lua_script: lua configuration file
    """
    file1 = open(os.path.join(lua_script), 'r')
    Lines = file1.readlines()
    for line in Lines:
        if("CHIRP_LOOPS =" in line):
            chirp_loops = int(line[13:line.find('-')].strip())
        if("NUM_RX =" in line):
            num_rx = int(line[9:line.find('-')].strip())
        if("NUM_TX =" in line):
            num_tx = int(line[9:line.find('-')].strip())
        if("ADC_SAMPLES =" in line):
            samples_per_chirp = int(line[14:line.find('-')].strip())
        if("PERIODICITY = " in line):
            periodicity = float(line[14:line.find('-')].strip())
        if("NUM_FRAMES = " in line):
            num_frames= float(line[12:line.find('-')].strip())

    data_rate = int(1 / (periodicity * 0.001) / 2)
    freq_plot_len = data_rate  // 2
    range_plot_len = samples_per_chirp

    return num_rx, num_tx, samples_per_chirp, periodicity, num_frames, chirp_loops, data_rate, freq_plot_len, range_plot_len


