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

from utils.extract_features import extract_features


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

