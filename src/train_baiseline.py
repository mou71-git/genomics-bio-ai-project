import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def train_models():
    print("--- Starting Baseline Model Training ---")

    # 1. Load Preprocessed Data
    X = np.load("data/processed/X_processed.npy")
    y = np.load("data/processed/y_labels.npy")

    # 2. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set:  {X_test.shape[0]} samples")

    # 3. Model 1: Logistic Regression (L2 Regularized Linear Model)
    log_reg = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

    print("\n=================== Logistic Regression ===================")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=["Normal", "Tumor"]))

    # 4. Model 2: Random Forest Classifier (Non-linear Ensemble Model)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    print("\n===================== Random Forest =====================")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Tumor"]))

    # 5. Save Model Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(log_reg, "models/logistic_regression_model.pkl")
    joblib.dump(rf, "models/random_forest_model.pkl")

    print("\nModels successfully trained and saved to models/")


if __name__ == "__main__":
    train_models()