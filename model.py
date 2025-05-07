import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

# Load CSV into a DataFrame
df = pd.read_csv("output_dataset.csv")
X = df.drop('Label', axis=1)
y = df['Label']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize XGBoost Classifier
xgb_model = XGBClassifier(
    objective="binary:logistic",  # Loss function for binary classification
    eval_metric="logloss",       # Evaluation metric
    use_label_encoder=False,     # Suppress label encoding warnings
    random_state=42
)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)
Y_prob = xgb_model.predict_proba(X_test_scaled)
xgb_model.save_model("xgboost_model.json")

print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
