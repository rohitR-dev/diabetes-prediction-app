import streamlit as st
import joblib
import numpy as np
import os

st.title("Diabetes Prediction App")

# Load model and scaler from files in current directory
model_path = "pickle.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model or scaler file not found! Please check your folder.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.write("Enter the patient details below:")

    preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    st.info("Number of times pregnant (0 to 20)")

    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    st.info("Plasma glucose concentration (0 to 200)")

    bp = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
    st.info("Diastolic blood pressure in mm Hg (0 to 140)")

    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    st.info("Triceps skin fold thickness in mm (0 to 100)")

    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    st.info("2-Hour serum insulin in mu U/ml (0 to 900)")

    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
    st.info("Body mass index (weight in kg/(height in m)^2), typical range 0 to 70")

    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
    st.info("Diabetes pedigree function (0.0 to 2.5)")

    age = st.number_input("Age", min_value=0, max_value=120, value=33)
    st.info("Age of the patient (0 to 120)")

    if st.button("Predict"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ Likely Diabetic (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Likely Not Diabetic (Confidence: {prob:.2f})")
