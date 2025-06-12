import librosa
import numpy as np
import pickle

# Load the trained model
with open("models/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Emotion label map
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Feature extractor matching 54 features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract components
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)           # 40
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)              # 12
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)      # 7

        # Combine only 2 contrast features to match training setup (or slice as needed)
        combined = np.hstack([mfcc, chroma, contrast[:2]])  # Total = 40 + 12 + 2 = 54
        return combined
    except Exception as e:
        print("❌ Error extracting features:", e)
        return None

# Path to the test audio file
test_file = "new_audio/test.wav"

# Predict
features = extract_features(test_file)
if features is not None:
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    print("🎧 Predicted Emotion:", prediction[0])
else:
    print("❌ Couldn't process audio.")

