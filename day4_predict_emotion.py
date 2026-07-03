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

# Load scaler if it exists (necessary for scaled models like SVM and Neural Network)
scaler = None
scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("ℹ️ Standard scaler loaded successfully.")

from utils.extract_features import extract_features

# Path to new audio file
test_file = "new_audio/test.wav"
if not os.path.exists(test_file):
    print("❌ test.wav not found in new_audio/")
    exit()

# Extract features
features = extract_features(test_file)

if features is not None:
    features = features.reshape(1, -1)
    
    # Scale features if scaler exists
    if scaler is not None:
        features = scaler.transform(features)
        
    prediction = model.predict(features)
    
    # Print prediction details
    print("\n🎧 Predicted Emotion:", prediction[0])
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        print("\n📊 Emotion Probabilities:")
        for label, prob in sorted(zip(model.classes_, proba), key=lambda x: -x[1]):
            print(f"{label:<10}: {prob * 100:.2f}%")
else:
    print("❌ Failed to extract features from audio.")
