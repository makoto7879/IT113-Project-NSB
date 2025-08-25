import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('initial_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Sleep Disorder Prediction App")
st.write("Enter the features below (raw values) to predict the sleep disorder using the initial Logistic Regression model.")

# Input fields for raw numerical features
age = st.number_input("Age (years, e.g., 27)", min_value=10, max_value=60, value=27)
sleep_duration = st.number_input("Sleep Duration (hours, e.g., 7)", min_value=4, max_value=12, value=7)
physical_activity_level = st.number_input("Physical Activity Level (minutes/day, e.g., 45)", min_value=0, max_value=120, value=45)
stress_level = st.number_input("Stress Level (0-10, e.g., 5)", min_value=0, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate (bpm, e.g., 70)", min_value=60, max_value=100, value=70)
daily_steps = st.number_input("Daily Steps (steps/day, e.g., 5000)", min_value=0, max_value=10000, value=5000)
systolic = st.number_input("Systolic Blood Pressure (mmHg, e.g., 120)", min_value=90, max_value=180, value=120)

# Input fields for categorical numeric features (raw as per encoding)
gender_numeric = st.selectbox("Gender (0: Male, 1: Female)", [0, 1])
occupation_numeric = st.selectbox("Occupation (Numeric code, e.g., 0-5 based on your encoding)", [0, 1, 2, 3, 4, 5])
bmi_category_numeric = st.selectbox("BMI Category (0: Normal, 1: Overweight, 2: Obese)", [0, 1, 2])

if st.button("Predict"):
    # Validate inputs
    if any(x <= 0 for x in [age, sleep_duration, physical_activity_level, daily_steps, systolic, heart_rate]) or stress_level < 0.1:
        st.error("Please enter positive values (e.g., no zeros for Sleep Duration, Daily Steps, etc.).")
    else:
        log_transformed_data = np.log([
            age, sleep_duration, physical_activity_level, stress_level,
            heart_rate, daily_steps, systolic, gender_numeric, occupation_numeric, bmi_category_numeric
        ])
        input_data = pd.DataFrame([log_transformed_data], columns=[
            'Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level',
            'Heart Rate', 'Daily Steps', 'Systolic', 'Gender_Numeric', 'Occupation_Numeric', 'BMI_category_numeric'
        ])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Sleep Disorder: {prediction}")

# Optional: Add a note about log transformation
st.write("Note: Input values are log-transformed internally to match the model's training data.")