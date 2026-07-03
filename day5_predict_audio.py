import librosa
import numpy as np
import pickle
import os

# Load the trained model
model_path = "models/emotion_model.pkl"
if not os.path.exists(model_path):
    print("❌ trained model not found in models/")
    exit()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load scaler if it exists
scaler = None
scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

from utils.extract_features import extract_features

# Path to the test audio file
test_file = "new_audio/test.wav"
if not os.path.exists(test_file):
    print(f"❌ test file not found at {test_file}")
    exit()

# Predict
features = extract_features(test_file)
if features is not None:
    features = features.reshape(1, -1)
    
    # Scale if scaler is available
    if scaler is not None:
        features = scaler.transform(features)
        
    prediction = model.predict(features)
    print("🎧 Predicted Emotion:", prediction[0])
else:
    print("❌ Couldn't process audio.")
