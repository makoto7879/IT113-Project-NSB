import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

st.title("Sleep Disorder Decision Tree Classifier Demo")

# Upload CSV
uploaded_file = st.file_uploader("Upload your cleaned_data_v2.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    param_dist = {
        'max_depth': [3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'sqrt', 'log2', 0.5, 0.7],
        'splitter': ['best', 'random'],
        'min_impurity_decrease': [0.0, 0.001, 0.01, 0.05],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.05],
        'class_weight': [None, 'balanced']
    }

    base_dt = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        base_dt,
        param_dist,
        n_iter=50,      # Lower for demo speed
        cv=3,           # Lower for demo speed
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    random_search.fit(X_train, y_train)
    best_tuned_model = random_search.best_estimator_

    feature_importances = pd.Series(best_tuned_model.feature_importances_, index=X.columns)
    st.subheader("Feature Importances")
    st.write(feature_importances.sort_values(ascending=False))

    models = {
        'Initial Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=5, min_samples_split=10,
            min_samples_leaf=10, criterion='gini'
        ),
        'Optimized Decision Tree': best_tuned_model
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        overfitting = train_acc - test_acc
        model_results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting': overfitting,
            'model': model
        }
        st.write(f"**{name}:** Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Overfitting: {overfitting:.4f}")

    best_test_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_acc'])
    best_final_model = model_results[best_test_model_name]['model']
    st.success(f"Best model: {best_test_model_name} (Test Acc: {model_results[best_test_model_name]['test_acc']:.4f})")

    y_test_pred = best_final_model.predict(X_test)
    st.subheader("Classification Report (Test Set)")
    st.text(classification_report(y_test, y_test_pred, target_names=le.classes_))

    cm_test = confusion_matrix(y_test, y_test_pred)
    st.subheader("Confusion Matrix (Test Set)")
    fig, ax = plt.subplots()
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    st.pyplot(fig)

    st.subheader("Decision Tree Visualization (First 3 Levels)")
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    tree.plot_tree(best_final_model,
                   feature_names=X.columns,
                   class_names=le.classes_,
                   filled=True,
                   max_depth=3,
                   fontsize=10,
                   ax=ax2)
    st.pyplot(fig2)
else:
    st.info("Upload your CSV file to run the demo.")