import streamlit as st
import joblib
import json
import pandas as pd

model   = joblib.load("model.joblib")
scaler  = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

with open("categories.json") as f:
    categories = json.load(f)

st.title("CO2 Emission Predictor 🚗")

vehicle_class    = st.selectbox("Vehicle Class", categories["Vehicle Class"])
fuel_type        = st.selectbox("Fuel Type", categories["Fuel Type"])
transmission     = st.selectbox("Transmission", categories["Transmission"])
cylinders        = st.selectbox("Cylinders", categories["Cylinders"])
engine_size      = st.number_input("Engine Size (L)", value=2.0)
fuel_consumption = st.number_input("Fuel Consumption Comb (L/100 km)", value=9.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Vehicle Class": vehicle_class,
        "Fuel Type": fuel_type,
        "Transmission": transmission,
        "Cylinders": cylinders,
        "Engine Size(L)": engine_size,
        "Fuel Consumption Comb (L/100 km)": fuel_consumption
    }])

    input_df[["Engine Size(L)", "Fuel Consumption Comb (L/100 km)"]] = scaler.transform(
        input_df[["Engine Size(L)", "Fuel Consumption Comb (L/100 km)"]]
    )
    input_df[["Vehicle Class", "Fuel Type", "Transmission"]] = encoder.transform(
        input_df[["Vehicle Class", "Fuel Type", "Transmission"]]
    )

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted CO2 Emission: {prediction:.2f} g/km")