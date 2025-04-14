-- Radar Settings (Original)
-- 3 Tx 4 Rx | complex 1x
--------------------------------------------------------------


COM_PORT = 4 -- TODO: edit this!!! (to the correct port number that you see in your Device Manager)
-- Firmware paths
RADARSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr18xx_radarss.bin"
MASTERSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr18xx_masterss.bin"

-------- VERY IMPORTANT AND SERIOUS RADAR SETTINGS --------
-- General
-- Number of transmitters you want to use (note that you still have to change ChirpConfig to reflect this)
NUM_TX = 3 
-- Number of receivers you want to use (note that you still have to change ChaNAdcConfig to reflect this)
NUM_RX = 4

-- ProfileConfig
-- START_FREQ: starting frequency of the FMCW chirp, TI1843 only starts from 77 GHz
START_FREQ = 77 -- GHz
-- IDLE_TIME: time between subsequent chirps (check RampTimingCalculator in mmWave Studio 
-- to calculate different values (dependent on FREQ_SLOP, ADC_SAMPLES, and SAMPLE_RATE))
IDLE_TIME = 7 -- us
-- RAMP_END_TIME: length of the entire frequency sweep (check RampTimingCalculator in mmWave 
-- Studio to calculate different values (dependent on FREQ_SLOP, ADC_SAMPLES, and SAMPLE_RATE))
RAMP_END_TIME = 57 -- us
-- ADC_START_TIME: time that samples start to be taken (check RampTimingCalculator in mmWave 
-- Studio to calculate different values (dependent on FREQ_SLOP, ADC_SAMPLES, and SAMPLE_RATE))
ADC_START_TIME = 5.28 --usn
-- FREQ_SLOPE: slope of the frequency sweep in the FMCW chirp, changing this along with 
-- ADC_SAMPLES and SAMPLE_RATE will affect the bandwidth
FREQ_SLOPE = 70.150 -- MHz/us
-- ADC_SAMPLES: number of samples we want to take (ex. when taking range FFT, this translates 
-- to number of range bins we get), changing this along with FREQ_SLOPE and SAMPLE_RATE will affect the bandwidth
ADC_SAMPLES = 512
-- SAMPLE_RATE: sample rate for ADC_SAMPLES, changing this along with ADC_SAMPLES and 
-- FREQ_SLOPE will affect the bandwidth
SAMPLE_RATE = 10000 -- ksps
-- RX_GAIN: reciever gain
RX_GAIN = 30 -- dB

-- ChirpConfig

-- FrameConfig
-- START_CHIRP_TX: index of the first chirp
START_CHIRP_TX = 0
-- END_CHIRP_TX: index of the last chirp (dependent on number of transmitters)
END_CHIRP_TX = NUM_TX-1 
-- number of subsequent chirps for each transmitter to repeat (check Programming 
-- Chirp Parameters in TI Radar Devices for more details)
CHIRP_LOOPS = 1   
-- NUM_FRAMES: number of frames you are collecting (each frame consists of each of the
-- trasmitter chirps as well as amount of chirp loops per transmitter chirp)
NUM_FRAMES = 0 -- 0 sets it to continous streaming!
-- PERIODICITY: period of each frame (aka, time between each set of (tx chirps and all of their chirp loops))
PERIODICITY = 20 -- ms 
-----------------------------------------------------------

-------- Reset the Radar and set up some basic stuff (static for 1843) --------
ar1.FullReset()
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(COM_PORT,921600,1000)
------------------------------

-------- Select the starting frequency band and the CHip version (static for 1843) --------
ar1.Calling_IsConnected()
ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("XWR1843")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-------- DOWNLOAD FIRMARE (static for 1843) --------
ar1.DownloadBSSFw(RADARSS_PATH)
ar1.DownloadMSSFw(MASTERSS_PATH)
ar1.PowerOn(1, 1000, 0, 0)
ar1.RfEnable()
--------

-------- STATIC CONFIG STUFF --------
-- ChanNAdcConfig: the first 7 inputs are the enables for each transmitter (0,1,2)
-- and each receiver (3,4,5,6), if you don't want to save all reciever values, then set them to 0
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0) 
ar1.LPModConfig(0, 0)
ar1.RfInit()
--------------------------------------

-------- DATA CONFIG STUFF (static for 1843) --------
ar1.DataPathConfig(1, 1, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
---------------------------------------------

-------- SENSOR CONFIG STUFF --------
-- Sets the chirp paramters based on your constants above
ar1.ProfileConfig(0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN)
-- inputs 0,1 are the chirp index (0 indexed), last three inputs are the transmitter enables
-- 2nd transmitters chirp will be sent first
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
-- 1st transmitters chirp will be sent second
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 1, 0, 0)
-- 3rd transmitters chirp will be sent third
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 0, 1)
-- frame configuration based on constants above
ar1.FrameConfig(START_CHIRP_TX, END_CHIRP_TX, NUM_FRAMES, CHIRP_LOOPS, PERIODICITY, 0, 0, 1)
-------------------------------------

-------- ETHERNET STUFF (static for 1843) --------
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(250)
--------------------------------


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

-------- CALCULATED AND NOT TOO SERIOUS PARAMETERS --------
CHIRPS_PER_FRAME = (END_CHIRP_TX - START_CHIRP_TX + 1) * CHIRP_LOOPS
NUM_DOPPLER_BINS = CHIRPS_PER_FRAME / NUM_TX
NUM_RANGE_BINS = ADC_SAMPLES
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
MAX_RANGE = (300 * 0.9 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)

print("Chirps Per Frame:", CHIRPS_PER_FRAME)
print("Num Doppler Bins:", NUM_DOPPLER_BINS)
print("Num Range Bins:", NUM_RANGE_BINS)
print("Range Resolution:", RANGE_RESOLUTION)
print("Max Unambiguous Range:", MAX_RANGE)
print("Doppler Resolution:", DOPPLER_RESOLUTION)
print("Max Doppler:", MAX_DOPPLER)