import numpy as np
import os
import scipy.io as sio
from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
def main():
    home_dir = '/Volumes/Expansion/NLOS_adcData/'
    save_path = '/Users/shanbhag/Documents/School/comm-proj-radars/given_data/'
    capture_data_dirs =  ["1108_los_taj_close", "1108_los_taj"]
    num_x_stp = 80 
    for capture_data_dir in capture_data_dirs:
        capture_data_dir = os.path.join(capture_data_dir,'exp2')
        startFrame = 58 - 1  
        endFrame = 796 - 1 
        num_z_stp = endFrame-startFrame  
        adcData_SAR_MF = np.zeros([num_x_stp, num_z_stp, 1, 4, 512],dtype=np.complex128)

        for filename in range(num_x_stp):
            adcData_file = 'exp2_h%d'%filename
            bin_data = sio.loadmat(os.path.join(home_dir,capture_data_dir,'adcData',adcData_file+ '.mat'))
            raw_data = np.array(bin_data['adcData'])
            adcData = raw_data[startFrame:endFrame,:,:,:] 
            if filename % 2 == 1:
                adcData = adcData[::-1,:,:,:]
            for ii in range(adcData.shape[0]):
                for jj in range(adcData.shape[1]):
                    for kk in range(adcData.shape[2]):
                        adcData[ii,jj,kk,:] = butter_highpass_filter(adcData[ii,jj,kk,:],350000,10e6)
            
            adcData_SAR_MF[filename, :, :, :,:] = adcData
            print("Done processing x: %d" % filename, end='\r')
        
        R = np.real(adcData_SAR_MF).astype(np.float32)
        I = np.imag(adcData_SAR_MF).astype(np.float32)

        save_file_folder = os.path.join(save_path,capture_data_dir)
        real_file = os.path.join(save_file_folder,'exp2_MF_real.bin')
        imag_file = os.path.join(save_file_folder,'exp2_MF_imag.bin')
        if not os.path.isfile(save_file_folder):
            os.mkdir(save_file_folder)

        print(real_file)
        print(imag_file)
        with open(real_file, 'wb') as fileID_real:
            R.tofile(fileID_real)

        with open(imag_file, 'wb') as fileID_imag:
            I.tofile(fileID_imag)
    
main()
