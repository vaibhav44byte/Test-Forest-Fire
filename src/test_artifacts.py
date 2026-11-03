import joblib
import json
import pandas as pd
import numpy as np

# Load artifacts
ridge = joblib.load("artifacts/ridge_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

with open("artifacts/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Example input (use an actual test record)
sample = pd.DataFrame(
    [[29, 57, 18, 0.0, 65.7, 3.4, 1.3, 0, 0]],
    columns=feature_columns
)

# Apply the original scaler
scaled_input = scaler.transform(sample)

# Predict
prediction = ridge.predict(scaled_input)

print("Predicted FWI:", prediction[0])