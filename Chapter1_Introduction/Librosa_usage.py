"""
Generate spectrogram using Librosa library
"""

#%% load audio data
import numpy as np, matplotlib.pyplot as plt
import librosa

clipPath = "AirCompressor_Data/Healthy/preprocess_Reading1.dat"
data = np.loadtxt(clipPath, delimiter=',')

# time waveform
plt.figure(figsize=(5,2))
plt.plot(data, 'grey', linewidth=0.3)
plt.title('Healthy'), plt.xlabel('sample #'), plt.ylabel('amplitude')

# log-mel spectrogram
sr = 50000/3 # sampling rate: 50000 samples per 3 seconds
mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=64)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(5,2))
img = librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
plt.colorbar(), plt.title('log-mel spectrogram')

