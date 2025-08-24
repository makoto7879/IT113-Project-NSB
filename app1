import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Sleep Disorder Interactive Prediction")

# --- Load Data for Feature Choices and Model Training ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data_v2.csv")
    label_encoder = LabelEncoder()
    df["Sleep Disorder Encoded"] = label_encoder.fit_transform(df["Sleep Disorder"])
    X = df.drop(columns=["Sleep Disorder", "Sleep Disorder Encoded"])
    y = df["Sleep Disorder Encoded"]
    return X, y, label_encoder, df

try:
    X, y, le, df = load_data()
    feature_names = X.columns.tolist()
except Exception as e:
    st.error("Error loading data. Please make sure cleaned_data_v2.csv is in the directory.")
    st.stop()

# --- Train Model (or load if you've saved one) ---
@st.cache_data
def train_model(X, y):
    model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=10, criterion='gini')
    model.fit(X, y)
    return model

model = train_model(X, y)

# --- Interactive Feature Input ---
st.header("Enter Features for Prediction")

user_input = {}
for feature in feature_names:
    if df[feature].dtype == 'object':
        choices = sorted(df[feature].dropna().unique())
        user_input[feature] = st.selectbox(f"{feature}:", choices)
    elif np.issubdtype(df[feature].dtype, np.integer) or np.issubdtype(df[feature].dtype, np.floating):
        min_val = int(df[feature].min())
        max_val = int(df[feature].max())
        user_input[feature] = st.number_input(f"{feature}:", min_value=min_val, max_value=max_val, value=min_val)
    else:
        user_input[feature] = st.text_input(f"{feature}:")

input_df = pd.DataFrame([user_input])

# --- Prediction ---
if st.button("Predict Sleep Disorder"):
    pred = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred])[0]
    st.success(f"Predicted Sleep Disorder: {pred_label}")

    st.write("Feature values entered:")
    st.write(input_df)
