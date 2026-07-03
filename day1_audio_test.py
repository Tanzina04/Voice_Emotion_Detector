import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd  # For Jupyter, optional

# Path to one sample audio file (adjust to your file)
file_path = "data/Actor_01/03-01-05-01-01-01-01.wav"

# Load audio file
audio, sample_rate = librosa.load(file_path)

print(f"Sample Rate: {sample_rate}")
print(f"Audio Duration (seconds): {len(audio)/sample_rate:.2f}")

# Listen to audio (if using Jupyter Notebook)
# ipd.Audio(file_path)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title("Audio Waveform")
plt.show()

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
print(f"MFCC shape: {mfccs.shape}")

# Plot MFCC
plt.figure(figsize=(12, 6))
plt.imshow(mfccs, cmap='viridis', origin='lower')
plt.title("MFCC")
plt.colorbar()
plt.show()
