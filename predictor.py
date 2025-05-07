import sys
import os
import cv2
import numpy as np
import xgboost as xgb
from scipy.stats import kurtosis

# Function to extract features from an image
def extract_image_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image from path: {image_path}")

    img = cv2.resize(img, (256, 256))  # Resize for uniformity
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    features = []
    for color_space in [img, img_ycrcb]:  # BGR and YCrCb
        for channel in range(3):
            channel_data = color_space[:, :, channel].flatten()
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(kurtosis(channel_data))
    return features

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_from_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Extract features from image
    try:
        features = extract_image_features(image_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Convert to 2D array for prediction (1 sample, n features)
    input_array = np.array([features])

    # Load pretrained XGBoost model
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")

    # Predict probability and class
    prob = model.predict_proba(input_array)[0][1]  # Probability of class 1
    pred = model.predict(input_array)[0]
# 0 okay and 1 means jaundice
    print(f"Predicted Class: {pred}")
    print(f"Probability of Class 1: {prob:.4f}")

if __name__ == "__main__":
    main()
