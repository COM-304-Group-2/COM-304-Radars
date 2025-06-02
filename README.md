# Real-Time Human Tracking with mmWave Radar

![1p_compressed](https://github.com/user-attachments/assets/37036c83-481f-4b11-ac04-62e2eae2f077)
![2p_compressed](https://github.com/user-attachments/assets/22e57142-43b2-45de-a17e-6d9fbf7356a0)
![2r_1p_c_compressed](https://github.com/user-attachments/assets/18994243-973c-43d5-8c42-ad328c673484)

**Final Project – COM-304: Communication Systems**

**Institution**: École Polytechnique Fédérale de Lausanne (EPFL)

**Team**: Timofey Kreslo, Thomas Kemper, Usejd Aliu, Timaël Andrié

## 🚀 Overview

This project was developed as part of COM-304: Communication Systems at EPFL and implements a real-time system for detecting and tracking humans using millimeter-wave (mmWave) radar. By leveraging the TI AWR1843BOOST radar and a fully Python-based pipeline, we demonstrate privacy-respecting and robust human tracking in diverse environments.

### Core Features:

- Real-time radar signal processing
- Angular beamforming with CFAR filtering
- Kalman-based multi-target tracking (GTrack)
- 2D Cartesian localization and motion trails
- Platform-independent Python implementation (macOS + Windows)

## System Pipeline
ADC Capture → Background Removal → CFAR Detection → Beamforming → GTrack → Visualization

## 🔧 Hardware Setup

- Radar: TI AWR1843BOOST (3 Tx × 4 Rx)
- Capture Interface: DCA1000EVM
- Frequency: 77 GHz center
- Bandwidth: 500 MHz
- Data Format: 992 range bins, 16 Doppler bins
- Virtual Antennas: 12

Firmware configs: 
```
scripts/
├── 1843_config.lua
├── 1843_config_streaming.lua
├── 1843_record.lua
└── profile_super.cfg
```

## ⚙️ Running the Project

Windows (with Lua & mmWaveStudio backend).
Only one radar supported.

MacOS/Linux (fully Python + mmwave-capture-std backend). 
No mmWave Studio required. Multiple radars supported.

## 📊 Visualization
- Matplotlib: Range FFT and beamformed heatmaps
- Panda3D: Real-time 3D object tracking with motion trails, IDs, and velocity vectors
  
<img src="https://github.com/user-attachments/assets/12e61d04-63ec-4272-ad92-c2d0ebf2068b" width="270">
<img src="https://github.com/user-attachments/assets/09d10cb4-bf4e-45f1-8119-903886d0683d" width="270">
<img src="https://github.com/user-attachments/assets/72523497-35d3-4343-9169-a88f0616de22" width="270">


## 🏛️ Architecture

| Stage | Description |
| -------- | ------- |
| ADC Capture | Raw radar data via DCA1000EVM | 
| Background Removal | Frame differencing to remove static clutter | 
| CFAR Detection | Identifies active motion in range-Doppler domain |
| Beamforming | Azimuth angle estimation (optimized with CFAR gating) | 
| Normalization | Scales beam power by max value for visualization & detection |
| GTrack | Real-time multi-target Kalman tracking + DBSCAN cluster initialization | 
| Visualization | Live 2D with object IDs, trails, and motion |

## 🔑 Key Results
- 15 FPS real-time performance (MacBook Pro M1 + DCA1000)
- Accurate tracking of 1 person.
- Multiple people tracking with unique IDs. (less accurate)
- Works through thin walls and occlusions
- Dual-radar fusion for enhanced angular resolution

## 📃 References

- [TI Developer Zone - Tracking with GTRACK](https://dev.ti.com/tirex/explore/node?node=A__AYZwK7t1GX7lsaN.HegOQw__RADAR-ACADEMY__GwxShWe__LATEST)
- [People Counting Using Low-Cost FMCW MIMO Radar](https://repository.tudelft.nl/record/uuid:a7450fad-43ff-446e-ba8e-d7a10fc50029)
- [mmwave-capture-std](https://github.com/mmwave-capture-std/mmwave-capture-std/)

## 📋 License Notice

Portions of this project include third-party code located in `src/mmwavecapture/`  
Copyright (c) 2023 Louie Lu <louielu@cs.unc.edu>  
Licensed under the Clear BSD License.

The license terms are included in the headers of the relevant source files.
