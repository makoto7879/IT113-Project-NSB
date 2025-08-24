import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# App title
st.title("Sleep Disorder Decision Tree Analysis (Interactive User Input)")

# Sidebar for data loading
st.sidebar.header("Load cleaned_data_v2.csv")
csv_path = "cleaned_data_v2.csv"
df = None

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success(f"Loaded {csv_path} from repository!")
    st.write("### Data Sample", df.head())
else:
    st.error(f"File {csv_path} not found in repository. Please upload it manually.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Loaded your uploaded file!")
        st.write("### Data Sample", df.head())

# Data processing and model training
if df is not None:
    # Dataset diagnostics
    st.write("#### Dataset Information")
    st.write(f"Dataset Size: {df.shape}")
    st.write(f"Features: {list(df.columns)}")
    st.write("Class Distribution in Target ('Sleep Disorder'):")
    st.write(pd.Series(df['Sleep Disorder']).value_counts(normalize=True))

    target_col = 'Sleep Disorder'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Data split diagnostics
    st.write("#### Data Split Information")
    st.write(f"Training Set Size: {X_train.shape}")
    st.write(f"Test Set Size: {X_test.shape}")
    st.write(f"Validation Set Size: {X_val.shape}")
    st.write("Class Distribution in Training Set:")
    st.write(pd.Series(y_train).value_counts(normalize=True))

    # Define models
    st.subheader("Model Definitions")
    initial_model = DecisionTreeClassifier(
        random_state=42, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, criterion='gini'
    )
    # Hardcode Tuned Decision Tree parameters from Colab
    tuned_model = DecisionTreeClassifier(
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

    st.write("#### Tuned Decision Tree Parameters (Hardcoded from Colab):")
    st.write("- splitter: best")
    st.write("- min_samples_split: 30")
    st.write("- min_samples_leaf: 10")
    st.write("- min_impurity_decrease: 0.001")
    st.write("- max_features: 0.5")
    st.write("- max_depth: 3")
    st.write("- criterion: entropy")
    st.write("- ccp_alpha: 0.01")

    models = {
        'Initial Decision Tree': initial_model,
        'Tuned Decision Tree': tuned_model
    }

    # Train and evaluate models
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        val_acc = accuracy_score(y_val, val_pred)

        model_results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'model': model
        }

    # Model comparison
    st.subheader("Model Comparison")
    model_comp = pd.DataFrame({
        "Train Accuracy": {k: v['train_acc'] for k, v in model_results.items()},
        "Test Accuracy": {k: v['test_acc'] for k, v in model_results.items()},
        "Validation Accuracy": {k: v['val_acc'] for k, v in model_results.items()}
    })

    st.write("#### Model Performance Metrics")
    st.dataframe(model_comp.style.format("{:.4f}").highlight_max(subset=['Test Accuracy'], color='lightgreen'))

    # Test accuracy comparison
    st.write("#### Test Accuracy Comparison")
    try:
        initial_test_acc = model_results['Initial Decision Tree']['test_acc']
        tuned_test_acc = model_results['Tuned Decision Tree']['test_acc']
        st.write(f"- **Initial Decision Tree**: Test Accuracy = {initial_test_acc:.4f}")
        st.write(f"- **Tuned Decision Tree**: Test Accuracy = {tuned_test_acc:.4f}")
        st.write(f"Difference (Tuned - Initial): {tuned_test_acc - initial_test_acc:.4f}")

        # Diagnostic for test accuracies
        st.write("#### Diagnostic Check")
        st.write(f"Raw Initial Test Accuracy: {initial_test_acc}")
        st.write(f"Raw Tuned Test Accuracy: {tuned_test_acc}")
        if abs(tuned_test_acc - initial_test_acc) < 1e-6:
            st.warning("The Initial and Tuned Decision Tree models have identical test accuracies. "
                       "This is unexpected since Colab shows Tuned Test Accuracy ~0.9067. "
                       "Possible reasons: dataset differences, preprocessing issues, or split inconsistencies.")
        elif tuned_test_acc > initial_test_acc and abs(tuned_test_acc - 0.9067) < 0.01:
            st.success("Tuned Decision Tree has higher test accuracy (~0.9067), matching Colab results.")
            if abs(model_results['Tuned Decision Tree']['train_acc'] - 0.8884) > 0.01 or \
               abs(model_results['Tuned Decision Tree']['val_acc'] - 0.9333) > 0.01:
                st.warning("Tuned model test accuracy matches Colab, but training or validation accuracies differ. "
                           "Check dataset or preprocessing.")
        else:
            st.error("Unexpected: Tuned Decision Tree test accuracy does not match Colab (0.9067). "
                     "Check dataset consistency, preprocessing, or library versions.")

    except KeyError as e:
        st.error(f"Error accessing model results: {e}. Please ensure models are trained correctly.")

    # Parameter comparison
    st.write("#### Parameter Comparison")
    st.write(f"Initial Decision Tree Parameters: max_depth=10, min_samples_split=5, min_samples_leaf=2, criterion='gini'")
    st.write("Tuned Decision Tree Parameters (from Colab):")
    st.write("- splitter: best")
    st.write("- min_samples_split: 30")
    st.write("- min_samples_leaf: 10")
    st.write("- min_impurity_decrease: 0.001")
    st.write("- max_features: 0.5")
    st.write("- max_depth: 3")
    st.write("- criterion: entropy")
    st.write("- ccp_alpha: 0.01")

    # Bar chart for test accuracy comparison
    st.write("#### Test Accuracy Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    models_names = ['Initial Decision Tree', 'Tuned Decision Tree']
    test_accuracies = [initial_test_acc, tuned_test_acc]
    bars = ax.bar(models_names, test_accuracies, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Test Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)

    # Select the best model based on test accuracy
    best_test_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_acc'])
    best_final_model = model_results[best_test_model_name]['model']
    st.success(f"Best model based on test accuracy: **{best_test_model_name}** (Test Accuracy: {model_results[best_test_model_name]['test_acc']:.4f})")

    # Evaluate the best model
    dt_classifier = best_final_model
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    y_val_pred = dt_classifier.predict(X_val)

    st.subheader("Performance Metrics")
    st.write(f"**Training Accuracy:** {accuracy_score(y_train, y_train_pred):.4f}")
    st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_test_pred):.4f}")
    st.write(f"**Validation Accuracy:** {accuracy_score(y_val, y_val_pred):.4f}")

    st.markdown("**Classification Report (Test Set):**")
    st.text(classification_report(y_test, y_test_pred, target_names=le.classes_))

    st.markdown("**Classification Report (Validation Set):**")
    st.text(classification_report(y_val, y_val_pred, target_names=le.classes_))

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title('Test Set')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
    ax2.set_title('Validation Set')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)

    # Decision tree visualization
    st.subheader("Decision Tree Visualization (First 3 Levels)")
    fig2 = plt.figure(figsize=(20, 10))
    tree.plot_tree(dt_classifier,
                   feature_names=X.columns,
                   class_names=le.classes_,
                   filled=True,
                   max_depth=3,
                   fontsize=10)
    plt.title(f'Decision Tree Visualization (First 3 Levels) - {best_test_model_name}')
    plt.tight_layout()
    st.pyplot(fig2)

    # Interactive user input for prediction
    st.header("Predict Sleep Disorder for New User")
    st.write("Input values for each feature below and click 'Predict Sleep Disorder'.")

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
    for col in X.columns:
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
        elif np.issubdtype(df[col].dtype, np.number):
            user_input[col] = st.number_input(
                f"{col}", 
                float(df[col].min()), float(df[col].max()), float(df[col].mean())
            )
        else:
            user_input[col] = st.selectbox(f"{col}", sorted(df[col].unique()))

    if st.button("Predict Sleep Disorder"):
        input_df = pd.DataFrame([user_input])
        # Ensure types match training data
        for col in input_df.columns:
            if not np.issubdtype(input_df[col].dtype, np.number):
                input_df[col] = input_df[col].astype(df[col].dtype)
        try:
            pred = dt_classifier.predict(input_df)[0]
            st.success(f"Predicted Sleep Disorder: {le.inverse_transform([pred])[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.info("Upload your cleaned_data_v2.csv file in the sidebar to begin.")
