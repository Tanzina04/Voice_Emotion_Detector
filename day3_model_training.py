import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from imblearn.over_sampling import SMOTE

# Load feature data
df = pd.read_pickle("features/emotion_features.pkl")

# Extract features and labels
X = np.array(df["features"].tolist())
y = np.array(df["emotion"].tolist())

# Split before SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n✅ After SMOTE class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Standardize features (crucial for SVM and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define models to train and compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "Support Vector Classifier (SVM)": SVC(kernel='rbf', C=10, probability=True, random_state=42),
    "Multi-Layer Perceptron (Neural Network)": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, clf in models.items():
    print(f"\n🌀 Training {name}...")
    clf.fit(X_train_scaled, y_train_resampled)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"👉 {name} Test Accuracy: {acc * 100:.2f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = clf
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")

# Save the best model and the scaler
os.makedirs("models", exist_ok=True)
with open("models/emotion_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\n✅ Saved best model to models/emotion_model.pkl")
print("✅ Saved standard scaler to models/scaler.pkl")

# Generate and save evaluation visualization for the best model
y_pred_best = best_model.predict(X_test_scaled)
print(f"\n📊 Final Classification Report ({best_model_name}):\n")
print(classification_report(y_test, y_pred_best))

plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_model.classes_, yticklabels=best_model.classes_, cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("models/confusion_matrix.png")
print("✅ Confusion matrix plot saved to models/confusion_matrix.png")
