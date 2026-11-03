
from src.predict import predict_fwi

# Sample weather conditions
sample_input = {
    "Temperature": 29,
    "RH": 57,
    "Ws": 18,
    "Rain": 0.0,
    "FFMC": 65.7,
    "DMC": 3.4,
    "ISI": 1.3,
    "Classes": 0,
    "Region": 0
}

# Generate prediction
result = predict_fwi(sample_input)

print("Predicted Fire Weather Index:", result)
