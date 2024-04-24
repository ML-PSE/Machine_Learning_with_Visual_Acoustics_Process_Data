# -*- coding: utf-8 -*-
"""
Explore air compressor sound data
"""

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

fs = 50000/3 # 50000 samples per 3 seconds

#%% healthy machine
path = "AirCompressor_Data/Healthy/preprocess_Reading1.dat"
data = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(data, 'grey', linewidth=0.3)
plt.title('Healthy')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(data, fs)

#%% Leakage inlet valve (LIV) fault
path = "AirCompressor_Data/LIV/preprocess_Reading1.dat"
dataLIV = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataLIV, 'grey', linewidth=0.3)
plt.title('LIV fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataLIV, fs)

#%% Leakage outlet valve fault
path = "AirCompressor_Data/LOV/preprocess_Reading1.dat"
dataLOV = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataLOV, 'grey', linewidth=0.3)
plt.title('LOV fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataLOV, fs)

#%% Non-Return valve fault
path = "AirCompressor_Data/NRV/preprocess_Reading1.dat"
dataNRV = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataNRV, 'grey', linewidth=0.3)
plt.title('NRV fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataNRV, fs)

#%% Piston ring fault
path = "AirCompressor_Data/Piston/preprocess_Reading1.dat"
dataPiston = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataPiston, 'grey', linewidth=0.3)
plt.title('Piston ring fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataPiston, fs)

#%% Flywheel fault
path = "AirCompressor_Data/Flywheel/preprocess_Reading1.dat"
dataFlywheel = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataFlywheel, 'grey', linewidth=0.3)
plt.title('Flywheel fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataFlywheel, fs)

#%% Riderbelt fault
path = "AirCompressor_Data/Riderbelt/preprocess_Reading1.dat"
dataRiderbelt = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataRiderbelt, 'grey', linewidth=0.3)
plt.title('Riderbelt fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataRiderbelt, fs)

#%% Bearing fault
path = "AirCompressor_Data/Bearing/preprocess_Reading1.dat"
dataBearing = np.loadtxt(path, delimiter=',')

plt.figure(figsize=(5,2))
plt.plot(dataBearing, 'grey', linewidth=0.3)
plt.title('Bearing fault')
plt.xlabel('sample #'), plt.ylabel('amplitude')

sd.play(dataBearing, fs)




