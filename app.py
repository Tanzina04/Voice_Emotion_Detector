from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import joblib
from utils.extract_features import extract_features

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Set upload folder to temporary directory (safe for deployment on Render)
UPLOAD_FOLDER = "/tmp/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained emotion detection model
model = joblib.load("models/emotion_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def results():
    prediction = request.args.get('emotion', default="Not Detected")
    return render_template('results.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    features = extract_features(file_path).reshape(1, -1)
    if features.shape[1] != 54:
        return jsonify({'error': f'Expected 54 features, got {features.shape[1]}'})

    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Required for Render
    app.run(host='0.0.0.0', port=port)
