# -*- coding: utf-8 -*-
"""
Explore air compressor sound data
"""

#%%
import numpy as np
import sounddevice as sd

# clipPath = "AirCompressor_Data/Healthy/preprocess_Reading1.dat"
clipPath = "AirCompressor_Data/Bearing/preprocess_Reading1.dat"

data = np.loadtxt(clipPath, delimiter=',')
fs = 50000/3 # 50000 samples per 3 seconds
sd.play(data, fs)





