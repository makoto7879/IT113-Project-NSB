import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

st.title("Sleep Disorder Decision Tree Analysis (Interactive User Input)")

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

if df is not None:
    # ... rest of your app code that uses df
    pass
else:
    st.info("Upload your cleaned_data_v2.csv file in the sidebar to begin.")

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

    # Hyperparameter search
    st.subheader("Randomized Hyperparameter Search (Decision Tree)")
    param_dist = {
        'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
        'splitter': ['best', 'random'],
        'min_impurity_decrease': [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    }

    base_dt = DecisionTreeClassifier(random_state=42)
    st.write("Running RandomizedSearchCV, please wait...")
    with st.spinner("Tuning hyperparameters..."):
        random_search = RandomizedSearchCV(
            base_dt,
            param_dist,
            n_iter=50,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        random_search.fit(X_train, y_train)

    st.write("#### Best Hyperparameters Found:")
    for param, value in random_search.best_params_.items():
        st.write(f"- **{param}**: {value}")

    best_tuned_model = random_search.best_estimator_

    models = {
        'Initial Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, criterion='gini'
        ),
        'Tuned Decision Tree': best_tuned_model
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        val_acc = accuracy_score(y_val, val_pred)
        overfitting = train_acc - val_acc

        model_results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'overfitting': overfitting,
            'model': model
        }

    st.subheader("Model Comparison")
    model_comp = pd.DataFrame({
        "Train Accuracy": {k: v['train_acc'] for k, v in model_results.items()},
        "Test Accuracy": {k: v['test_acc'] for k, v in model_results.items()},
        "Validation Accuracy": {k: v['val_acc'] for k, v in model_results.items()},
        "Overfitting Metric": {k: v['overfitting'] for k, v in model_results.items()}
    })
    st.dataframe(model_comp.style.format("{:.4f}"))

    best_val_model_name = max(model_results.keys(), key=lambda x: model_results[x]['val_acc'])
    best_final_model = model_results[best_val_model_name]['model']
    st.success(f"Best model based on validation accuracy: **{best_val_model_name}** (Validation Accuracy: {model_results[best_val_model_name]['val_acc']:.4f})")

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

    st.subheader("Decision Tree Visualization (First 3 Levels)")
    fig2 = plt.figure(figsize=(20, 10))
    tree.plot_tree(dt_classifier,
                   feature_names=X.columns,
                   class_names=le.classes_,
                   filled=True,
                   max_depth=3,
                   fontsize=10)
    plt.title('Decision Tree Visualization (First 3 Levels)')
    plt.tight_layout()
    st.pyplot(fig2)

    # Interactive User Input for Prediction
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
