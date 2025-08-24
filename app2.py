import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Sleep Disorder Decision Tree Analysis (Interactive Feature Selection)")

st.sidebar.header("1. Upload your cleaned_data_v2.csv")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Sample", df.head())

    target_col = 'Sleep Disorder'
    all_features = [col for col in df.columns if col != target_col]

    st.sidebar.header("2. Select Features for Modeling")
    selected_features = st.sidebar.multiselect(
        "Choose input features for the model (minimum 2):",
        options=all_features,
        default=all_features
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two features.")
        st.stop()

    X = df[selected_features]
    y = df[target_col]

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into train, test, and validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH
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

    st.write("Running RandomizedSearchCV (may take ~1 minute)...")
    with st.spinner("Tuning hyperparameters..."):
        base_dt = DecisionTreeClassifier(random_state=42)
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

    # Models for comparison
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

    # Model Comparison Table
    st.subheader("Model Comparison")
    model_comp = pd.DataFrame({
        "Train Accuracy": {k: v['train_acc'] for k, v in model_results.items()},
        "Test Accuracy": {k: v['test_acc'] for k, v in model_results.items()},
        "Validation Accuracy": {k: v['val_acc'] for k, v in model_results.items()},
        "Overfitting Metric": {k: v['overfitting'] for k, v in model_results.items()}
    })
    st.dataframe(model_comp.style.format("{:.4f}"))

    # Final Model Selection
    best_val_model_name = max(model_results.keys(), key=lambda x: model_results[x]['val_acc'])
    best_final_model = model_results[best_val_model_name]['model']
    st.success(f"Best model based on validation accuracy: **{best_val_model_name}** (Validation Accuracy: {model_results[best_val_model_name]['val_acc']:.4f})")

    # Final Evaluation
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
                   feature_names=selected_features,
                   class_names=le.classes_,
                   filled=True,
                   max_depth=3,
                   fontsize=10)
    plt.title('Decision Tree Visualization (First 3 Levels)')
    plt.tight_layout()
    st.pyplot(fig2)

    st.info("You can change the selected features in the sidebar and rerun for different analysis!")

else:
    st.info("Upload your cleaned_data_v2.csv file in the sidebar to begin.")