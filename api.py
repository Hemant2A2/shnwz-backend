from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import datetime
import os
import uuid
import pulse

import numpy as np
import xgboost as xgb
from scipy.stats import kurtosis
import cv2


# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Directory to temporarily store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.environ.get('XGB_MODEL_PATH', 'xgboost_model.json')
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Stub: Replace these with real sensor integrations
import random

def get_health_data_from_sensors():
    """
    Fetch data from actual sensors. This is a stub returning random values.
    Replace with your hardware interfacing code.
    """

    temperature = pulse.bme.temperature
    humidity = pulse.bme.humidity
    pressure = pulse.bme.pressure
    gas = pulse.bme.gas

    return {
        'heartRate': random.randint(60, 100),
        'temperature': round(temperature, 2),
        'pressure': round(pressure, 2),
        'humidity': round(humidity, 2),
        'gasLevel': int(gas),
        'timestamp': datetime.datetime.utcnow().isoformat()
    }

# Stub: Replace with actual ML model inference for jaundice detection

def analyze_jaundice_image(image_path: str):
    """
    Analyze the uploaded image for signs of jaundice.
    Replace this stub with actual model loading and prediction code.
    """
    # Example dummy result
    result = {
        'isJaundiced': random.choice([True, False]),
        'confidence': round(random.uniform(0.5, 1.0), 2),
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    return result

@app.route('/api/health-data', methods=['GET'])
def health_data():
    """
    GET endpoint to fetch health sensor data.
    """
    try:
        data = get_health_data_from_sensors()
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Failed to fetch health data: {e}")
        abort(500, description="Internal Server Error: unable to fetch health data")

def extract_image_features(image_path):
    """
    Extract statistical features from BGR and YCrCb channels.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img = cv2.resize(img, (256, 256))
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    feats = []
    for cs in (img, img_ycrcb):
        for ch in range(3):
            arr = cs[:, :, ch].flatten()
            feats.append(np.mean(arr))
            feats.append(np.std(arr))
            feats.append(kurtosis(arr))
    return feats

@app.route('/api/analyze-jaundice', methods=['POST'])
def analyze_jaundice():
    if 'image' not in request.files:
        abort(400, description="No image provided")

    image = request.files['image']
    if image.filename == '':
        abort(400, description="Empty filename")

    # Save image
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    try:
        # Feature extraction
        feats = extract_image_features(path)
        x = np.array([feats])
        # Prediction
        prob = float(model.predict_proba(x)[0][1])
        pred = int(model.predict(x)[0])

        result = {
            'isJaundiced': bool(pred == 1),
            'confidence': round(prob, 4),
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Jaundice analysis error: {e}")
        abort(500, description="Analysis failed")
    finally:
        # Optional cleanup
        try:
            os.remove(path)
        except OSError:
            pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)