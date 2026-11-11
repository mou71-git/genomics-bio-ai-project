import joblib
import numpy as np
import pandas as pd


def analyze_model_features():
    print("--- Running Biological Feature Importance Analysis ---")

    # 1. Load saved artifacts
    log_reg = joblib.load("models/logistic_regression_model.pkl")
    rf = joblib.load("models/random_forest_model.pkl")
    selected_genes = np.load(
        "data/processed/selected_genes.npy", allow_pickle=True
    )

    # 2. Extract Logistic Regression Coefficients
    # Positive weights = indicative of Tumor | Negative weights = indicative of Normal
    lr_weights = log_reg.coef_[0]

    lr_df = pd.DataFrame(
        {"Gene": selected_genes, "Weight": lr_weights, "Abs_Weight": np.abs(lr_weights)}
    ).sort_values(by="Abs_Weight", ascending=False)

    print("\nTop 10 Most Important Genes (Logistic Regression):")
    print(lr_df[["Gene", "Weight"]].head(10).to_string(index=False))

    # 3. Extract Random Forest Feature Importances
    rf_importances = rf.feature_importances_

    rf_df = pd.DataFrame(
        {"Gene": selected_genes, "Importance": rf_importances}
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Most Important Genes (Random Forest):")
    print(rf_df.head(10).to_string(index=False))


if __name__ == "__main__":
    analyze_model_features()