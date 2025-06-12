import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

        # Total: 40 + 12 + 1 + 1 = 54 (verify this is what you trained with)
        return np.hstack([mfcc, chroma, mel[:1], contrast[:1]])
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

