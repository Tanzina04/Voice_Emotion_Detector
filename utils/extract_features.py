import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)

        # Total: 40 (mfcc) + 12 (chroma) + 1 (zcr) + 1 (spectral centroid) = 54 features
        return np.hstack([mfcc, chroma, zcr, spec_centroid])
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None


