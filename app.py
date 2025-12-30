import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ride Price Predictor", layout="centered")

st.title("Ride Sharing Price Prediction")

MODEL_PATH = "ride_price_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None
try:
    model = load_model()
except Exception as e:
    st.warning("Model not found yet. Run the training notebook/script first to create ride_price_model.joblib.")
    st.caption(f"Details: {e}")

with st.form("inputs"):
    st.subheader("Ride details")

    distance_miles = st.number_input("Distance (miles)", min_value=0.1, value=2.0, step=0.1)
    duration_minutes = st.number_input("Duration (minutes)", min_value=1.0, value=10.0, step=1.0)

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour (0–23)", 0, 23, 12)
    with col2:
        day_of_week = st.slider("Day of week (0=Mon … 6=Sun)", 0, 6, 2)

    weather = st.selectbox("Weather", ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Snow"])
    temperature = st.number_input("Temperature (°F)", min_value=0.0, max_value=120.0, value=65.0, step=1.0)

    pickup_location = st.selectbox("Pickup location type", ["Residential", "Downtown", "Business", "Entertainment", "Airport"])
    dropoff_location = st.selectbox("Dropoff location type", ["Residential", "Downtown", "Business", "Entertainment", "Airport"])

    vehicle_type = st.selectbox("Vehicle type", ["Shared", "Standard", "Premium", "Luxury"])
    driver_rating = st.number_input("Driver rating (1–5)", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

    submitted = st.form_submit_button("Predict price")

if submitted:
    if model is None:
        st.error("No model loaded. Train and save the model first.")
    else:
        row = pd.DataFrame([{
            "distance_miles": distance_miles,
            "duration_minutes": duration_minutes,
            "hour": hour,
            "day_of_week": day_of_week,
            "weather": weather,
            "temperature": temperature,
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "vehicle_type": vehicle_type,
            "driver_rating": driver_rating,
        }])

        pred = float(model.predict(row)[0])
        st.success(f"Predicted price: **${pred:,.2f}**")
