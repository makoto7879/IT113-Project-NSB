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
    X_original = df.drop(target_col, axis=1)  # Keep original data for reference
    y = df[target_col]

    # Define categorical columns that should NOT be log-transformed
    categorical_cols = ['Gender_Numeric', 'BMI_category_numeric', 'Occupation_Numeric']
    
    # Apply log transformation to training data (same as your original training)
    X = X_original.copy()
    for col in X.columns:
        if col not in categorical_cols and np.issubdtype(X[col].dtype, np.number):
            X[col] = np.log1p(X[col])  # Apply log transform to training data

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
    
    # Debug section - show first few training examples for verification
    st.sidebar.write("Debug: First training example")
    debug_row = X_train.iloc[0:1]
    debug_pred = dt_classifier.predict(debug_row)[0]
    debug_actual = y_train[0]
    st.sidebar.write(f"Prediction: {debug_pred}, Actual: {debug_actual}, Match: {debug_pred == debug_actual}")

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
        elif np.issubdtype(X_original[col].dtype, np.number):
            # Use ORIGINAL data for min/max/mean (before log transform)
            user_input[col] = st.number_input(
                f"{col}",
                min_value=0,
                max_value=int(X_original[col].max()),  # Use original data max
                value=int(X_original[col].mean()),     # Use original data mean  
                step=1,
                format="%d"
            )
        else:
            user_input[col] = st.selectbox(f"{col}", sorted(X_original[col].unique()))

    if st.button("Predict Sleep Disorder"):
        # Create DataFrame with same column order as training
        input_df = pd.DataFrame([user_input])[X_original.columns]
        
        # Apply the SAME log transformation as training data
        for col in input_df.columns:
            if col not in categorical_cols and np.issubdtype(input_df[col].dtype, np.number):
                input_df[col] = np.log1p(input_df[col])  # Apply log transform
            if not np.issubdtype(input_df[col].dtype, np.number):
                input_df[col] = input_df[col].astype(X_original[col].dtype)
        
        # Show what's being fed to the model for debugging
        st.write("Debug - Transformed input features:")
        st.write(input_df)
        
        try:
            pred = dt_classifier.predict(input_df)[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f"Predicted Sleep Disorder: {pred_label}")
            
            # Show prediction confidence/probability if available
            if hasattr(dt_classifier, "predict_proba"):
                proba = dt_classifier.predict_proba(input_df)[0]
                st.write("Prediction probabilities:")
                for i, class_name in enumerate(le.classes_):
                    st.write(f"- {class_name}: {proba[i]:.2%}")
                    
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("Upload your cleaned_data_v2.csv file in the sidebar to begin.")
