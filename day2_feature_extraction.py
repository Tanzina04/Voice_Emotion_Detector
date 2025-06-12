import os
import librosa
import numpy as np
import pandas as pd

# Emotion mapping from filename
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Set your dataset folder
DATA_DIR = "data"
features_list = []

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
        combined = np.hstack([mfccs, chroma, zcr, spec_centroid])
        return combined
    except Exception as e:
        print("❌ Error in:", file_path, "→", e)
        return None

# Walk through each file
for actor_folder in os.listdir(DATA_DIR):
    actor_path = os.path.join(DATA_DIR, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                emotion_code = file.split("-")[2]  # ✅ CORRECTED INDEX
                emotion_label = emotion_dict.get(emotion_code)
                features = extract_features(file_path)
                if features is not None and emotion_label is not None:
                    features_list.append([features, emotion_label])

# Convert to DataFrame
df = pd.DataFrame(features_list, columns=["features", "emotion"])

# Save features
os.makedirs("features", exist_ok=True)
df.to_pickle("features/emotion_features.pkl")
print("✅ Feature extraction complete! Saved as emotion_features.pkl")

# Show class distribution
print("\n📊 Class Distribution:\n")
print(df["emotion"].value_counts())

