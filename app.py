from flask import Flask, render_template, request, jsonify
import os
from utils.extract_features import extract_features
import joblib

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    port = int(os.environ.get("PORT", 10000))  # Render sets this PORT env variable
    app.run(host='0.0.0.0', port=port)         # Bind to 0.0.0.0 so Render can access it
