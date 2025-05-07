import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# Function to extract features from an image
def extract_image_features(image_path):
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize for uniformity

    # Convert to different color spaces
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Calculate mean, std, and kurtosis for each channel in both color spaces
    features = []
    for color_space in [img, img_ycrcb]:  # BGR and YCrCb
        for channel in range(3):
            channel_data = color_space[:, :, channel].flatten()
            features.append(np.mean(channel_data))  # Mean
            features.append(np.std(channel_data))   # Standard Deviation
            features.append(kurtosis(channel_data)) # Kurtosis

    return features

# Main function to process Normal and Jaundiced folders and save features
def process_normal_and_jaundiced_folders(base_folder, output_csv):
    data = []
    labels = []

    # Define folder-to-label mapping
    folder_label_mapping = {
        "normal": 0,
        "jaundice": 1
    }
    cnt = 0

    for folder, label in folder_label_mapping.items():
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder} not found in {base_folder}, skipping.")
            continue

        print(f"Processing folder: {folder} with label: {label}...")

        # Process each image in the folder
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    features = extract_image_features(image_path)
                    data.append(features)
                    labels.append(label)
                    cnt += 1    
                    if cnt % 100 == 0:
                        print(f"Processed {cnt} images...")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Convert to DataFrame
    feature_columns = [
        f"{color}_{stat}_{space}" 
        for space in ["RGB", "YCrCb"] 
        for color in ["R", "G", "B"] 
        for stat in ["mean", "std", "kurt"]
    ]
    df = pd.DataFrame(data, columns=feature_columns)
    df['Label'] = labels

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Example usage
base_folder = "."  # Replace with the path to your image folders
output_csv = "output_dataset.csv"       # Name of the output CSV file
process_normal_and_jaundiced_folders(base_folder, output_csv)
# print(extract_image_features("jaundice (1).jpg"))