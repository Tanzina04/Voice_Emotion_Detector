from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from utils.extract_features import extract_features
import joblib

app = Flask(__name__, template_folder='templates')
CORS(app)  # Allow cross-origin requests for fetch()

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
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

    try:
        features = extract_features(file_path).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {str(e)}'})

    if features.shape[1] != 54:
        return jsonify({'error': f'Expected 54 features, got {features.shape[1]}'})

    try:
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT via env variable
    app.run(host='0.0.0.0', port=port)
