import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---- Load the trained model and scaler ----
model = load_model("dmodel.h5")

with open("scaler1.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---- Streamlit UI ----
st.title("Diabetes Prediction App 🩺")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Collect input into array
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    probability = model.predict(input_scaled)[0][0]
    prediction = int(probability > 0.5)

    st.write(f"**Prediction:** {'Diabetes' if prediction==1 else 'No Diabetes'}")
    st.write(f"**Probability:** {probability*100:.2f}%")
