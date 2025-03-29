import json
import os
import numpy as np
import scipy.io as sio

# Global parameters
Params = {}
ui = {}
dataSet = {}
EXIT_KEY_PRESSED = 0

def dp_numberOfEnabledChan(chanMask):
    MAX_RXCHAN = 4
    count = 0
    for chan in range(MAX_RXCHAN):
        bitVal = 1 << chan
        if chanMask & bitVal == bitVal:
            count += 1
            chanMask = chanMask - bitVal
            if chanMask == 0:
                break
    return count

def dp_generateADCDataParams(mmwaveJSON):
    global Params
    adcDataParams = {}
    frameCfg = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlFrameCfg_t"]

    adcDataParams["dataFmt"] = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlAdcOutCfg_t"]["fmt"]["b2AdcOutFmt"]
    adcDataParams["iqSwap"] = mmwaveJSON["mmWaveDevices"]["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["iqSwapSel"]
    adcDataParams["chanInterleave"] = mmwaveJSON["mmWaveDevices"]["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["chInterleave"]
    adcDataParams["numChirpsPerFrame"] = frameCfg["numLoops"] * (frameCfg["chirpEndIdx"] - frameCfg["chirpStartIdx"] + 1)
    adcDataParams["adcBits"] = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlAdcOutCfg_t"]["fmt"]["b2AdcBits"]
    rxChanMask = int(mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlChanCfg_t"]["rxChannelEn"], 16)

    adcDataParams["numRxChan"] = dp_numberOfEnabledChan(rxChanMask)
    adcDataParams["numAdcSamples"] = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlProfiles"]["rlProfileCfg_t"]["numAdcSamples"]

    dp_printADCDataParams(adcDataParams)

    # Calculate ADC data size
    if adcDataParams["adcBits"] == 2:
        if adcDataParams["dataFmt"] == 0:
            gAdcOneSampleSize = 2  # real data, one sample is 16bits=2bytes
        elif adcDataParams["dataFmt"] in (1, 2):
            gAdcOneSampleSize = 4  # complex data, one sample is 32bits=4 bytes
        else:
            print("Error: unsupported ADC dataFmt")
    else:
        print("Error: unsupported ADC bits ({})".format(adcDataParams["adcBits"]))

    dataSizeOneChirp = gAdcOneSampleSize * adcDataParams["numAdcSamples"] * adcDataParams["numRxChan"]
    Params["dataSizeOneFrame"] = dataSizeOneChirp * adcDataParams["numChirpsPerFrame"]
    Params["dataSizeOneChirp"] = dataSizeOneChirp

    Params["adcDataParams"] = adcDataParams
    return adcDataParams

def dp_printADCDataParams(adcDataParams):
    print('Input ADC data parameters:')
    print('    dataFmt:{}'.format(adcDataParams["dataFmt"]))
    print('    iqSwap:{}'.format(adcDataParams["iqSwap"]))
    print('    chanInterleave:{}'.format(adcDataParams["chanInterleave"]))
    print('    numChirpsPerFrame:{}'.format(adcDataParams["numChirpsPerFrame"]))
    print('    adcBits:{}'.format(adcDataParams["adcBits"]))
    print('    numRxChan:{}'.format(adcDataParams["numRxChan"]))
    print('    numAdcSamples:{}'.format(adcDataParams["numAdcSamples"]))

def dp_generateRadarCubeParams(mmwaveJSON):
    global Params

    radarCubeParams = {}
    frameCfg = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlFrameCfg_t"]

    radarCubeParams["iqSwap"] = mmwaveJSON["mmWaveDevices"]["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["iqSwapSel"]
    rxChanMask = int(mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlChanCfg_t"]["rxChannelEn"], 16)
    radarCubeParams["numRxChan"] = dp_numberOfEnabledChan(rxChanMask)
    radarCubeParams["numTxChan"] = frameCfg["chirpEndIdx"] - frameCfg["chirpStartIdx"] + 1

    radarCubeParams["numRangeBins"] = 2 ** np.ceil(np.log2(mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlProfiles"]["rlProfileCfg_t"]["numAdcSamples"]))
    radarCubeParams["numDopplerChirps"] = frameCfg["numLoops"]

    radarCubeParams["radarCubeFmt"] = 1  # RADAR_CUBE_FORMAT_1

    dp_printRadarCubeParams(radarCubeParams)
    Params["radarCubeParams"] = radarCubeParams
    return radarCubeParams

def dp_printRadarCubeParams(radarCubeParams):
    print('Radarcube parameters:')
    print('    iqSwap:{}'.format(radarCubeParams["iqSwap"]))
    print('    radarCubeFmt:{}'.format(radarCubeParams["radarCubeFmt"]))
    print('    numDopplerChirps:{}'.format(radarCubeParams["numDopplerChirps"]))
    print('    numRxChan:{}'.format(radarCubeParams["numRxChan"]))
    print('    numTxChan:{}'.format(radarCubeParams["numTxChan"]))
    print('    numRangeBins:{}'.format(radarCubeParams["numRangeBins"]))

def dp_generateRFParams(mmwaveJSON, radarCubeParams, adcDataParams):
    C = 3e8
    profileCfg = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlProfiles"]["rlProfileCfg_t"]

    RFParams = {}
    RFParams["startFreq"] = profileCfg["startFreqConst_GHz"]
    RFParams["freqSlope"] = profileCfg["freqSlopeConst_MHz_usec"]
    RFParams["sampleRate"] = profileCfg["digOutSampleRate"] / 1e3

    RFParams["numRangeBins"] = 2 ** np.ceil(np.log2(adcDataParams["numAdcSamples"]))
    RFParams["numDopplerBins"] = radarCubeParams["numDopplerChirps"]
    RFParams["bandwidth"] = abs(RFParams["freqSlope"] * profileCfg["numAdcSamples"] / profileCfg["digOutSampleRate"])

    RFParams["rangeResolutionsInMeters"] = C * RFParams["sampleRate"] / (2 * RFParams["freqSlope"] * RFParams["numRangeBins"] * 1e6)
    RFParams["dopplerResolutionMps"] = C / (
                2 * RFParams["startFreq"] * 1e9 * (profileCfg["idleTimeConst_usec"] + profileCfg["rampEndTime_usec"]) * 1e-6 * radarCubeParams[
            "numDopplerChirps"] * radarCubeParams["numTxChan"])
    RFParams["framePeriodicity"] = mmwaveJSON["mmWaveDevices"]["rfConfig"]["rlFrameCfg_t"]["framePeriodicity_msec"]
    return RFParams

def dp_reshape2LaneLVDS(rawData):
    rawData4 = np.reshape(rawData, (len(rawData) // 4,4))
    rawDataI = np.reshape(rawData4[:,0:2], (-1, 1))
    rawDataQ = np.reshape(rawData4[:,2:4], (-1, 1))
    frameData = np.concatenate((rawDataI, rawDataQ), axis=1)
    return frameData

def dp_exportData(rawDataFileName, radarCubeDataFileName):
    global dataSet
    global Params

    rawData = []

    for frameIdx in range(1, Params["NFrame"] + 1):
        dp_updateFrameData(frameIdx)
        rawData.append(np.array(dataSet["rawFrameData"]))

    if radarCubeDataFileName:
        radarCubeParams = Params["radarCubeParams"]
        radarCube = {
            "rfParams": Params["RFParams"],
            "data_raw": rawData,
            "dim": {
                "numFrames": Params["NFrame"],
                "numChirps": radarCubeParams["numTxChan"] * radarCubeParams["numDopplerChirps"],
                "numRxChan": radarCubeParams["numRxChan"],
                "numRangeBins": radarCubeParams["numRangeBins"],
                "iqSwap": radarCubeParams["iqSwap"],
            },
        }
        sio.savemat(radarCubeDataFileName+".mat", radarCube)

def dp_updateFrameData(frameIdx):
    global Params
    global dataSet

    currFrameIdx = 0
    fidIdx = 0
    for idx in [1]:
        if frameIdx <= (Params["NFrame"] + currFrameIdx):
            fidIdx = idx
            break
        else:
            currFrameIdx += Params["NFramePerFile"][idx - 1]

    rawDataComplex = dp_loadOneFrameData(Params["fid_rawData"], Params["dataSizeOneFrame"],
                                            frameIdx - currFrameIdx)
    dataSet["rawDataUint16"] = np.uint16(rawDataComplex)
    timeDomainData = rawDataComplex - (rawDataComplex >= 2 ** 15) * 2 ** 16
    dp_generateFrameData(timeDomainData)

def dp_getNumberOfFrameFromBinFile(binFileName):
    global Params
    try:
        binFile = os.path.getsize(binFileName)
        fileSize = binFile
    except Exception as e:
        raise ValueError("Reading Bin file failed") from e
    return int(fileSize / Params["dataSizeOneFrame"])

def dp_loadOneFrameData(fid_rawData, dataSizeOneFrame, frameIdx):
    try:
        fid_rawData.seek((frameIdx - 1) * dataSizeOneFrame)
        rawData = np.fromfile(fid_rawData, dtype=np.uint16, count=dataSizeOneFrame // 2).astype(np.float32)
    except Exception as e:
        raise ValueError("Error reading binary file") from e

    if dataSizeOneFrame != len(rawData) * 2:
        print("dp_loadOneFrameData, size = {}, expected = {}".format(len(rawData), dataSizeOneFrame))
        raise ValueError("Read data from bin file, have the wrong length")

    return rawData

def dp_generateFrameData(rawData):
    global Params
    global dataSet

    frameData = dp_reshape2LaneLVDS(rawData)

    frameComplex = frameData[:, 0] + 1j * frameData[:, 1]
    frameComplex = frameComplex.reshape((Params["NChirp"],Params["NChan"],Params["NSample"]))
    dataSet["rawFrameData"] = frameComplex

def dp_numberOfEnabledChan(chanMask):
    MAX_RXCHAN = 4
    count = 0
    for chan in range(MAX_RXCHAN):
        bitVal = 2 ** chan
        if (chanMask & bitVal) == bitVal:
            count += 1
            chanMask -= bitVal
            if chanMask == 0:
                break
    return count

def dp_printADCDataParams(adcDataParams):
    print('Input ADC data parameters:')
    print('    dataFmt: {}'.format(adcDataParams["dataFmt"]))
    print('    iqSwap: {}'.format(adcDataParams["iqSwap"]))
    print('    chanInterleave: {}'.format(adcDataParams["chanInterleave"]))
    print('    numChirpsPerFrame: {}'.format(adcDataParams["numChirpsPerFrame"]))
    print('    adcBits: {}'.format(adcDataParams["adcBits"]))
    print('    numRxChan: {}'.format(adcDataParams["numRxChan"]))
    print('    numAdcSamples: {}'.format(adcDataParams["numAdcSamples"]))

def dp_printRadarCubeParams(radarCubeParams):
    print('Radarcube parameters:')
    print('    iqSwap: {}'.format(radarCubeParams["iqSwap"]))
    print('    radarCubeFmt: {}'.format(radarCubeParams["radarCubeFmt"]))
    print('    numDopplerChirps: {}'.format(radarCubeParams["numDopplerChirps"]))
    print('    numRxChan: {}'.format(radarCubeParams["numRxChan"]))
    print('    numTxChan: {}'.format(radarCubeParams["numTxChan"]))
    print('    numRangeBins: {}'.format(radarCubeParams["numRangeBins"]))

def rawDataReader(radarCfgFileName,rawDataFileName,radarCubeDataFileName):
    global Params
    global ui
    global dataSet

    with open(radarCfgFileName, 'r') as json_file:
        setupJSON = json.load(json_file)
    mmwaveJSONfilename = setupJSON["configUsed"]

    with open(mmwaveJSONfilename, 'r') as json_file:
        mmwaveJSON = json.load(json_file)
    binFile = os.path.join(setupJSON["capturedFiles"]["fileBasePath"], setupJSON["capturedFiles"]["files"]["rawFileName"])
    Params["fid_rawData"] = open(binFile, "rb")

    adcDataParams = dp_generateADCDataParams(mmwaveJSON)
    radarCubeParams = dp_generateRadarCubeParams(mmwaveJSON)
    Params["RFParams"] = dp_generateRFParams(mmwaveJSON, radarCubeParams, adcDataParams)

    Params["NSample"] = adcDataParams["numAdcSamples"]
    Params["NChirp"] = adcDataParams["numChirpsPerFrame"]
    Params["NChan"] = adcDataParams["numRxChan"]
    Params["NTxAnt"] = radarCubeParams["numTxChan"]
    Params["numRangeBins"] = radarCubeParams["numRangeBins"]
    Params["numDopplerBins"] = radarCubeParams["numDopplerChirps"]
    Params["rangeWinType"] = 0
    Params["NFrame"] = dp_getNumberOfFrameFromBinFile(binFile)

    dp_exportData(rawDataFileName, radarCubeDataFileName)

