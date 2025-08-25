import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Function to safely load models
@st.cache_resource
def load_models():
    """Load the saved model and scaler with error handling"""
    try:
        # Check if files exist
        model_file = 'initial_logistic_model.pkl'
        scaler_file = 'scaler.pkl'
        
        if not os.path.exists(model_file):
            st.error(f"Model file '{model_file}' not found. Available files: {os.listdir('.')}")
            return None, None
            
        if not os.path.exists(scaler_file):
            st.error(f"Scaler file '{scaler_file}' not found. Available files: {os.listdir('.')}")
            return None, None
        
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def validate_inputs(age, sleep_duration, physical_activity_level, stress_level, 
                   heart_rate, daily_steps, systolic):
    """Validate input values"""
    errors = []
    
    # Check for valid ranges
    if age <= 0 or age > 120:
        errors.append("Age must be between 1 and 120 years")
    
    if sleep_duration <= 0 or sleep_duration > 24:
        errors.append("Sleep Duration must be between 0 and 24 hours")
    
    if physical_activity_level < 0:
        errors.append("Physical Activity Level cannot be negative")
    
    if stress_level < 0 or stress_level > 10:
        errors.append("Stress Level must be between 0 and 10")
    
    if heart_rate <= 0 or heart_rate > 200:
        errors.append("Heart Rate must be between 1 and 200 bpm")
    
    if daily_steps < 0:
        errors.append("Daily Steps cannot be negative")
    
    if systolic <= 0 or systolic > 300:
        errors.append("Systolic BP must be between 1 and 300 mmHg")
    
    return errors

def clean_and_transform_data(age, sleep_duration, physical_activity_level, stress_level,
                            heart_rate, daily_steps, systolic, gender_numeric, 
                            occupation_numeric, bmi_category_numeric):
    """Clean and transform input data"""
    try:
        # Apply small epsilon to avoid log(0)
        epsilon = 1e-8
        
        # Ensure all values are positive for log transformation
        values = [
            max(age, epsilon),
            max(sleep_duration, epsilon), 
            max(physical_activity_level, epsilon),
            max(stress_level + 0.1, epsilon),  # Add 0.1 to stress level to avoid log(0)
            max(heart_rate, epsilon),
            max(daily_steps, epsilon),
            max(systolic, epsilon),
            max(gender_numeric + epsilon, epsilon),
            max(occupation_numeric + epsilon, epsilon),
            max(bmi_category_numeric + epsilon, epsilon)
        ]
        
        # Apply log transformation
        log_transformed_data = np.log(values)
        
        # Create DataFrame
        input_data = pd.DataFrame([log_transformed_data], columns=[
            'Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level',
            'Heart Rate', 'Daily Steps', 'Systolic', 'Gender_Numeric', 
            'Occupation_Numeric', 'BMI_category_numeric'
        ])
        
        # Check for any remaining invalid values
        if input_data.isnull().any().any():
            st.error("Data contains missing values after transformation")
            return None
            
        if np.isinf(input_data.values).any():
            st.error("Data contains infinite values after transformation")
            return None
            
        return input_data
        
    except Exception as e:
        st.error(f"Error in data transformation: {str(e)}")
        return None

# Load models
model, scaler = load_models()

if model is None or scaler is None:
    st.stop()

# Streamlit app title
st.title("Sleep Disorder Prediction App")
st.write("Enter the features below (raw values) to predict the sleep disorder using the initial Logistic Regression model.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=27)
    gender_numeric = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    
    st.subheader("Physical Metrics")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=200, value=120)
    bmi_category_numeric = st.selectbox("BMI Category", [0, 1, 2], 
                                       format_func=lambda x: ["Normal", "Overweight", "Obese"][x])

with col2:
    st.subheader("Lifestyle Factors")
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=1.0, max_value=12.0, value=7.0, step=0.1)
    physical_activity_level = st.number_input("Physical Activity Level (minutes/day)", 
                                            min_value=0, max_value=300, value=45)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)
    stress_level = st.number_input("Stress Level (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    st.subheader("Work Information")
    occupation_options = ["Software Engineer", "Doctor", "Sales Representative", 
                         "Teacher", "Nurse", "Engineer", "Other"]
    occupation_display = st.selectbox("Occupation", occupation_options)
    occupation_numeric = occupation_options.index(occupation_display)

# Prediction button
if st.button("Predict Sleep Disorder", type="primary"):
    # Validate inputs
    validation_errors = validate_inputs(age, sleep_duration, physical_activity_level, 
                                      stress_level, heart_rate, daily_steps, systolic)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        with st.spinner("Making prediction..."):
            # Clean and transform data
            input_data = clean_and_transform_data(
                age, sleep_duration, physical_activity_level, stress_level,
                heart_rate, daily_steps, systolic, gender_numeric, 
                occupation_numeric, bmi_category_numeric
            )
            
            if input_data is not None:
                try:
                    # Scale the data
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    
                    # Display results
                    st.success(f"**Predicted Sleep Disorder:** {prediction}")
                    
                    # Show prediction probabilities if available
                    if len(prediction_proba) > 1:
                        st.subheader("Prediction Confidence")
                        prob_df = pd.DataFrame({
                            'Class': [f'Class {i}' for i in range(len(prediction_proba))],
                            'Probability': prediction_proba
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                    
                    # Show input summary
                    with st.expander("Input Summary"):
                        input_summary = pd.DataFrame({
                            'Feature': ['Age', 'Sleep Duration', 'Physical Activity', 'Stress Level',
                                      'Heart Rate', 'Daily Steps', 'Systolic BP', 'Gender', 
                                      'Occupation', 'BMI Category'],
                            'Value': [age, sleep_duration, physical_activity_level, stress_level,
                                    heart_rate, daily_steps, systolic, 
                                    "Female" if gender_numeric == 1 else "Male",
                                    occupation_options[occupation_numeric],
                                    ["Normal", "Overweight", "Obese"][bmi_category_numeric]]
                        })
                        st.table(input_summary)
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# Add information section
st.markdown("---")
st.subheader("About This App")
st.info("""
This app uses a logistic regression model to predict sleep disorders based on various health and lifestyle factors.
The model applies log transformation to numerical features before scaling and prediction.

**Note:** This is for educational/demonstration purposes only and should not be used for actual medical diagnosis.
""")

# Add footer
st.markdown("---")
st.caption("Sleep Disorder Prediction App | Built with Streamlit")
