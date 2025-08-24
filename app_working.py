import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# App title
st.title("Sleep Disorder Prediction")

# Sidebar for data loading
st.sidebar.header("Load cleaned_data_v2.csv")
csv_path = "cleaned_data_v2.csv"
df = None

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.error(f"File {csv_path} not found in repository. Please upload it manually.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Loaded your uploaded file!")

# Data processing and model training
if df is not None:
    target_col = 'Sleep Disorder'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data (for training only, not displayed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Train Tuned Decision Tree with Colab parameters
    dt_classifier = DecisionTreeClassifier(
        random_state=42,
        splitter='best',
        min_samples_split=30,
        min_samples_leaf=10,
        min_impurity_decrease=0.001,
        max_features=0.5,
        max_depth=3,
        criterion='entropy',
        ccp_alpha=0.01
    )
    dt_classifier.fit(X_train, y_train)

    # Internal diagnostic (not displayed) to check test accuracy
    test_pred = dt_classifier.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    if abs(test_acc - 0.9067) > 0.01:
        # Log for debugging, not shown to user
        with open("debug_log.txt", "w") as f:
            f.write(f"Tuned model test accuracy: {test_acc:.4f}, expected ~0.9067\n")
    # Add this right after training your model, before the user input section
    st.write("=== DEBUG SECTION ===")
    st.write("Testing model on first training example:")
        
    # Interactive user input for prediction
    st.header("Predict Sleep Disorder")

    occupation_options = {
        0: "Software Engineer",
        1: "Doctor",
        2: "Sales Representative",
        3: "Teacher",
        4: "Nurse",
        5: "Engineer",
        6: "Accountant",
        7: "Scientist",
        8: "Lawyer",
        9: "Salesperson",
        10: "Manager"
    }

    # Define categorical columns that are not log-transformed
    categorical_cols = ['Gender_Numeric', 'BMI_category_numeric', 'Occupation_Numeric']

    user_input = {}
    for col in X_original.columns:  # Use original column order
        if col == "Gender_Numeric":
            user_input[col] = st.radio(
                "Gender", 
                options=[1, 0], 
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
        elif col == "BMI_category_numeric":
            user_input[col] = st.radio(
                "BMI Category",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Normal Weight", 1: "Overweight", 2: "Obese"}[x]
            )
        elif col == "Occupation_Numeric":
            user_input[col] = st.radio(
                "Occupation",
                options=list(occupation_options.keys()),
                format_func=lambda x: occupation_options[x]
            )
        elif col == "Sleep Duration" and np.issubdtype(X_original[col].dtype, np.number):
            # Special handling for Sleep Duration - allow 1 decimal place
            original_max = float(np.expm1(X_original[col].max()))
            original_mean = float(np.expm1(X_original[col].mean()))
            user_input[col] = st.number_input(
                f"{col}",
                min_value=0.0,
                max_value=original_max,
                value=original_mean,
                step=0.1,  # Allow 0.1 increments
                format="%.1f"  # Show 1 decimal place
            )
        elif np.issubdtype(X_original[col].dtype, np.number):
            # Your CSV is already log-transformed, so convert back to original scale for user input
            original_max = int(np.expm1(X_original[col].max()))  # Convert from log scale
            original_mean = int(np.expm1(X_original[col].mean()))  # Convert from log scale
            user_input[col] = st.number_input(
                f"{col}",
                min_value=0,
                max_value=original_max,  # Now shows realistic numbers
                value=original_mean,     # Now shows realistic numbers
                step=1,
                format="%d"
            )
        else:
            user_input[col] = st.selectbox(f"{col}", sorted(X_original[col].unique()))

    if st.button("Predict Sleep Disorder"):
        # Create DataFrame with same column order as training
        input_df = pd.DataFrame([user_input])[X_original.columns]
        
        # Transform user input to log scale (since model expects log-transformed data)
        for col in input_df.columns:
            if col not in categorical_cols and np.issubdtype(input_df[col].dtype, np.number):
                input_df[col] = np.log1p(input_df[col])  # Transform user input to log scale
            if not np.issubdtype(input_df[col].dtype, np.number):
                input_df[col] = input_df[col].astype(X_original[col].dtype)
        
        try:
            pred = dt_classifier.predict(input_df)[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f"Predicted Sleep Disorder: {pred_label}")
                    
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("Upload your cleaned_data_v2.csv file in the sidebar to begin.")
