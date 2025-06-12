import librosa
import numpy as np
import pickle
import os

# Load the trained model
with open("models/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the feature extraction function (same as training)
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)      # 40
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)         # 12
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0) # 7

        combined = np.hstack([mfcc, chroma, contrast])  # Total = 40 + 12 + 7 = 59
        return combined[:54]  # ✅ Take only first 54 features to match model
    except Exception as e:
        print("❌ Error extracting features:", e)
        return None


# Path to new audio file
test_file = "new_audio/test.wav"
if not os.path.exists(test_file):
    print("❌ test.wav not found in new_audio/")
    exit()

# Extract features
features = extract_features(test_file)

if features is not None:
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0]

    print("🎧 Predicted Emotion:", prediction[0])
    print("\n📊 Emotion Probabilities:")
    for label, prob in sorted(zip(model.classes_, proba), key=lambda x: -x[1]):
        print(f"{label:<10}: {prob:.2f}")
else:
    print("❌ Failed to extract features from audio.")
