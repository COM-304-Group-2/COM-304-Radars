import time

import os
import clr
import processing.utility as utility

# helper functions
def replace_filename(lua_file, exp_name, exp_path):
    with open(lua_file, 'r') as file:
        data = file.readlines()
    for i,line in enumerate(data):
        if("capture_file               =" in line):
            data[i] = 'capture_file               =   "%s"\n' % exp_name
        if("SAVE_DATA_PATH = " in line):
            data[i] = 'SAVE_DATA_PATH = "%s" .. capture_file .. ".bin"\n' % exp_path

    with open(lua_file, 'w') as file:
        file.writelines(data)

# radar class:
# initiates connection with mmWave Studio
# measures data
# can do processing if required later on
class radar():
    def __init__(self,home_dir,path_name,lua_script):
        self.captured = False
        self.chirp_loops = 1
        self.num_rx = 4
        self.num_tx = 2
        self.samples_per_chirp = 128 
        self.periodicity = 20
        self.num_frames = 10
        self.homedirectory = home_dir
        self.num_rx, self.num_tx, self.samples_per_chirp, self.periodicity, self.num_frames, self.chirp_loops, self.data_rate, self.freq_plot_len, self.range_plot_len = utility.read_radar_params(os.path.join(self.homedirectory,lua_script))

        self.power_dict = dict()
        self.RtttNetClientAPI = self.Init_RSTD_Connection(path_name)
    
    
    def Init_RSTD_Connection(self, RSTD_DLL_Path):
        RSTD_Assembly = clr.AddReference(RSTD_DLL_Path)
        import RtttNetClientAPI
        try:
            RtttNetClientAPI.RtttNetClient.IsConnected()
            Init_RSTD_Connection = 0
        except:
            Init_RSTD_Connection = 1
        if Init_RSTD_Connection:
            print('Initializing RSTD client')
            ErrStatus = RtttNetClientAPI.RtttNetClient.Init()
            if not ErrStatus == 0:
                print('Unable to initialize NetClient DLL')
                return
            print('Connecting to RSTD client')
            ErrStatus = RtttNetClientAPI.RtttNetClient.Connect('127.0.0.1',2777)
            if not ErrStatus == 0:
                print('Unable to connect to mmWaveStudio')
                print('Reopen port in mmWaveStudio. Type RSTD.NetClose() followed by RSTD.NetStart()')
                return
            time.sleep(1)

        print('Sending test message to RSTD')
        Lua_String = r'WriteToLog("Running script from Python\\n", "green")'
        ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String)
        if not ErrStatus == (0, None):
            print ('mmWaveStudio Connection Failed')
        else:
            print('Test message success')
        return RtttNetClientAPI

    def mmwave_config(self, script_name):
        file1 = os.path.join(self.homedirectory,script_name)
        file2 = file1.replace("\\", "\\\\\\\\") 
        Lua_String = 'dofile("'+ file2 + '")'
        # update the lua file with new location to save the data
        ErrStatus = self.RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String)
        if not ErrStatus == (0, None):
            print ('The config did not update :(')
        else:
            print('Radar configurated!')

    def mmwave_capture(self, exp_name, exp_path, script_name):
        exp_path = exp_path.replace("\\", "\\\\\\\\") + "\\\\\\\\"
        print(exp_path)
        script_name = script_name.replace("\\", "\\") 
        file1 = os.path.join(self.homedirectory,script_name)
        file2 = file1.replace("\\", "\\\\\\\\") 
        Lua_String = 'dofile("'+ file2 + '")'
        # update the lua file with new location to save the data
        replace_filename(file1, exp_name, exp_path)
        ErrStatus = self.RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String)
        if not ErrStatus == (0, None):
            print ('The frame did not get collected :(')
        else:
            print('Frame collected!')
