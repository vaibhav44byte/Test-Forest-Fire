
import joblib
import json
import pandas as pd
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Artifact paths
MODEL_PATH = BASE_DIR / "artifacts" / "ridge_model.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "artifacts" / "feature_columns.json"

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_columns = json.load(f)


def predict_fwi(input_data):
    """
    Predict Fire Weather Index using Ridge Regression.

    input_data: Dictionary containing the nine model features.
    """

    # Validate missing features
    missing_features = [
        col for col in feature_columns
        if col not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Arrange features in training order
    input_df = pd.DataFrame(
        [[input_data[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Validate numeric values
    input_df = input_df.astype(float)

    if input_df.isnull().values.any():
        raise ValueError("Input contains missing values")

    # Apply saved scaler
    scaled_input = scaler.transform(input_df)

    # Generate prediction
    prediction = model.predict(scaled_input)

    return float(prediction[0])
