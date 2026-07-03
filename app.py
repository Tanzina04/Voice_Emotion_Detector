from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import joblib
import librosa
import soundfile as sf
import speech_recognition as sr
import nltk
from utils.extract_features import extract_features

# Check and download VADER lexicon
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"⚠️ Warning: Could not download vader_lexicon: {e}")
        sia = None
except Exception as e:
    print(f"⚠️ Warning: Error loading SentimentIntensityAnalyzer: {e}")
    sia = None

# Supported audio/video formats
ALLOWED_EXTENSIONS = {
    '.wav', '.mp3', '.ogg', '.flac', '.aac',
    '.m4a', '.mp4', '.webm', '.wma', '.aiff', '.au'
}

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Set upload folder relative to the app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained emotion detection model
model_path = os.path.join(BASE_DIR, "models", "emotion_model.pkl")
model = joblib.load(model_path)

# Load the scaler if it exists
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)

def transcribe_and_analyze(file_path):
    transcription = ""
    sentiment_scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    sentiment_label = "neutral"
    
    try:
        # Load audio using librosa to ensure compatibility with non-WAV formats
        # Convert to 16000Hz mono WAV (best for Google Speech Recognition)
        y, sr_rate = librosa.load(file_path, sr=16000, mono=True)
        
        # Save to temporary WAV file in the uploads folder
        temp_wav_name = "temp_transcribe_" + os.path.basename(file_path) + ".wav"
        temp_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_wav_name)
        
        sf.write(temp_wav_path, y, sr_rate, subtype='PCM_16')
        
        # Initialize recognizer
        r = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = r.record(source)
            
        try:
            transcription = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcription = ""
        except sr.RequestError as e:
            transcription = "[Transcription service unavailable]"
            print(f"⚠️ SpeechRecognition request error: {e}")
            
        # Clean up temp WAV
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as cleanup_err:
            print(f"⚠️ Failed to remove temp WAV {temp_wav_path}: {cleanup_err}")
            
    except Exception as e:
        print(f"❌ Error in transcription: {e}")
        transcription = "[Failed to process audio for transcription]"
        
    # Analyze text sentiment using VADER
    if transcription and not transcription.startswith("[") and sia is not None:
        try:
            sentiment_scores = sia.polarity_scores(transcription)
            compound = sentiment_scores.get("compound", 0.0)
            if compound >= 0.05:
                sentiment_label = "positive"
            elif compound <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
        except Exception as e:
            print(f"⚠️ VADER sentiment error: {e}")
            
    return {
        "text": transcription,
        "sentiment": {
            "score": sentiment_scores.get("compound", 0.0),
            "label": sentiment_label,
            "scores": sentiment_scores
        }
    }

def get_psychological_alignment(acoustic_emotion, nlp_result):
    text = nlp_result.get("text", "")
    sentiment = nlp_result.get("sentiment", {})
    label = sentiment.get("label", "neutral")
    
    if not text or text.startswith("["):
        return {
            "type": "acoustic_only",
            "title": "Acoustic Analysis Only",
            "description": "No clear spoken words were detected or transcription was unavailable. Analysis is based solely on vocal tone.",
            "badge_color": "var(--text-muted)"
        }
        
    acoustic_emotion = acoustic_emotion.lower().strip()
    
    if acoustic_emotion in ["happy", "calm"]:
        voice_valence = "positive"
    elif acoustic_emotion in ["sad", "angry", "fearful", "disgust"]:
        voice_valence = "negative"
    else:
        voice_valence = "neutral"
        
    if voice_valence == "positive" and label == "negative":
        return {
            "type": "sarcasm",
            "title": "Sarcasm / Emotional Masking",
            "description": "Vocal tone sounds pleasant or calm, but the words spoken contain negative sentiment. This is a common pattern in sarcastic remarks or when someone is masking negative feelings behind a polite facade.",
            "badge_color": "var(--accent-secondary)"
        }
    elif voice_valence == "negative" and label == "positive":
        return {
            "type": "vocal_distress",
            "title": "Vocal Distress / Masking",
            "description": "The speaker is using positive words, but their vocal delivery is classified as negative (sad, anxious, or angry). This suggests emotional masking, where the speaker is trying to sound upbeat but vocal cues reveal underlying stress or sadness.",
            "badge_color": "#ff9e00"
        }
    elif voice_valence == "neutral" and label != "neutral":
        return {
            "type": "controlled_expression",
            "title": "Controlled Expression",
            "description": "The speaker's voice is calm and neutral, but the spoken content is highly positive or negative. This suggests a controlled or professional communication style where emotion is contained.",
            "badge_color": "var(--accent-cyan)"
        }
    elif voice_valence != "neutral" and label == "neutral":
        emotion_description_map = {
            "happy": "talking enthusiastically about a neutral subject.",
            "sad": "speaking in a downcast, slow manner even though the words are neutral.",
            "angry": "expressing annoyance or intense emphasis on neutral words.",
            "fearful": "exhibiting stress or anxiety in vocal tone during neutral statements.",
            "disgust": "speaking with vocal contempt or aversion.",
            "surprised": "vocalizing with sudden excitement or alarm on a neutral phrase."
        }
        desc = emotion_description_map.get(acoustic_emotion, "exhibiting vocal emotion during a neutral statement.")
        return {
            "type": "subtle_emotion",
            "title": "Subtle / Passive Emotion",
            "description": f"The spoken content is neutral, but the voice displays distinct emotional coloring ({acoustic_emotion}). The speaker is {desc}",
            "badge_color": "#b55fe6"
        }
    elif voice_valence == label:
        if voice_valence == "positive":
            return {
                "type": "congruent_positive",
                "title": "Congruent Positive Expression",
                "description": "Vocal tone and word sentiment are fully aligned! The speaker feels positive and expresses it openly through both their tone and their words.",
                "badge_color": "#10b981"
            }
        elif voice_valence == "negative":
            return {
                "type": "congruent_negative",
                "title": "Congruent Negative Expression",
                "description": "Vocal tone and word sentiment are fully aligned. The speaker's negative feelings (sadness, anger, or fear) are expressed directly in both voice and text.",
                "badge_color": "#ef4444"
            }
            
    return {
        "type": "congruent_neutral",
        "title": "Congruent Balanced State",
        "description": "A stable and balanced expression where both vocal tone and word choice are neutral and objective.",
        "badge_color": "var(--accent-cyan)"
    }

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
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    # Validate file extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        supported = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({'error': f'Unsupported format "{ext}". Supported: {supported}'})

    # Save file to upload folder preserving original extension
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract acoustic features (librosa handles all supported formats)
    features = extract_features(file_path)

    # Transcribe and analyze sentiment BEFORE deleting the uploaded file
    nlp_result = transcribe_and_analyze(file_path)

    # Clean up temp file regardless of result
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Warning: Could not remove temp file {file_path}: {e}")

    if features is None:
        return jsonify({'error': 'Could not extract audio features. For MP4/M4A/AAC formats, please ensure ffmpeg is installed on the server.'})

    features = features.reshape(1, -1)

    # Scale features if standard scaler is available
    if scaler is not None:
        features = scaler.transform(features)

    # Perform prediction
    prediction = model.predict(features)[0]

    # Calculate probabilities if available
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
        probabilities = {label: float(prob) for label, prob in zip(model.classes_, probs)}

    # Determine psychological alignment
    alignment = get_psychological_alignment(prediction, nlp_result)

    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities,
        'transcription': nlp_result.get('text', ''),
        'text_sentiment': nlp_result.get('sentiment', {}),
        'alignment': alignment
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Required for Render
    app.run(host='0.0.0.0', port=port)
