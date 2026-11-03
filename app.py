
import streamlit as st
import json
from pathlib import Path
from src.predict import predict_fwi

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Forest Fire FWI Predictor",
    page_icon="🔥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------
# Load Model Metrics
# --------------------------------

with open(BASE_DIR / "artifacts" / "model_metrics.json", "r") as f:
    metrics = json.load(f)

# --------------------------------
# Application Header
# --------------------------------

st.title("🔥 Algerian Forest Fire Prediction")
st.subheader("Fire Weather Index (FWI) Prediction")

st.write(
    "Enter the weather and fire-weather parameters "
    "to predict the Fire Weather Index using Ridge Regression."
)

st.divider()

# --------------------------------
# Model Performance
# --------------------------------

st.markdown("### Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("ML Model", metrics["model"])
col2.metric("R² Score", f'{metrics["R2_Score"]:.4f}')
col3.metric("Mean Absolute Error", f'{metrics["MAE"]:.4f}')

st.divider()

# --------------------------------
# User Input Form
# --------------------------------

st.markdown("### Enter Weather Parameters")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Weather Conditions")

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=29.0,
            step=1.0
        )

        rh = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=57.0
        )

        ws = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=200.0,
            value=18.0
        )

        rain = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=0.0
        )

    with col2:
        st.markdown("#### Fire Weather Indices")

        ffmc = st.number_input(
            "FFMC",
            min_value=0.0,
            max_value=101.0,
            value=65.7
        )

        dmc = st.number_input(
            "DMC",
            min_value=0.0,
            max_value=500.0,
            value=3.4
        )

        isi = st.number_input(
            "ISI",
            min_value=0.0,
            max_value=100.0,
            value=1.3
        )

        classes = st.selectbox(
            "Fire Class (encoded)",
            options=[0, 1],
            format_func=lambda x: (
                "Not Fire (0)" if x == 0 else "Fire (1)"
            )
        )

        region = st.selectbox(
            "Region (encoded)",
            options=[0, 1],
            format_func=lambda x: (
                "Bejaia (0)" if x == 0
                else "Sidi-Bel Abbes (1)"
            )
        )

    submitted = st.form_submit_button(
        "Predict Fire Weather Index",
        use_container_width=True
    )

# --------------------------------
# Prediction
# --------------------------------

if submitted:

    input_data = {
        "Temperature": temperature,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": ffmc,
        "DMC": dmc,
        "ISI": isi,
        "Classes": classes,
        "Region": region
    }

    try:
        prediction = predict_fwi(input_data)

        st.divider()
        st.markdown("### Prediction Result")

        st.metric(
            label="Predicted Fire Weather Index (FWI)",
            value=f"{prediction:.4f}"
        )

        st.success("Prediction generated successfully!")

        st.caption(
            "This is an educational ML prediction based on "
            "the Algerian forest fires dataset. It is not an "
            "operational wildfire warning."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

st.caption(
    "Built using Python | Scikit-learn | Ridge Regression | Streamlit"
)
