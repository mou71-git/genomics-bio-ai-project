import joblib
import numpy as np
import pandas as pd


class RNACancerPredictor:
    """End-to-end deployment pipeline for RNA-seq breast cancer classification."""

    def __init__(
        self,
        feature_selector_path="models/feature_selector.pkl",
        scaler_path="models/scaler.pkl",
        model_path="models/logistic_regression_model.pkl",
        selected_genes_path="data/processed/selected_genes.npy",
    ):
        # Load saved preprocessors and trained model artifacts
        self.selector = joblib.load(feature_selector_path)
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        self.selected_genes = np.load(selected_genes_path, allow_pickle=True)

    def preprocess_raw_sample(self, raw_counts_df):
        """Applies CPM normalization, Log2 transform, Feature Selection,

        and Scaling to raw RNA-seq count input.
        """
        X_raw = raw_counts_df.values  # Shape: (N_samples, P_genes)

        # 1. CPM Normalization
        library_sizes = X_raw.sum(axis=1, keepdims=True)
        library_sizes[library_sizes == 0] = 1.0
        cpm = (X_raw / library_sizes) * 1e6

        # 2. Log2 Transformation
        X_log = np.log2(cpm + 1.0)

        # 3. Feature Selection using saved ANOVA selector
        X_selected = self.selector.transform(X_log)

        # 4. Feature Scaling using saved StandardScaler
        X_scaled = self.scaler.transform(X_selected)

        return X_scaled

    def predict(self, raw_counts_df):
        """Predicts binary outcome (0: Normal, 1: Tumor) and probabilities."""
        X_processed = self.preprocess_raw_sample(raw_counts_df)

        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        results = []
        for pred, proba in zip(predictions, probabilities):
            label = "Tumor" if pred == 1 else "Normal"
            confidence = proba[1] if pred == 1 else proba[0]
            results.append({"prediction": label, "confidence": float(confidence)})

        return results


if __name__ == "__main__":
    # Test script with raw data sample
    raw_df = pd.read_csv("data/raw_gene_matrix.csv", index_col=0).T
    predictor = RNACancerPredictor()

    # Predict on the first 3 samples as a test
    test_sample = raw_df.iloc[:3]
    predictions = predictor.predict(test_sample)

    for i, res in enumerate(predictions):
        print(
            f"Sample {i+1} ({test_sample.index[i]}): {res['prediction']} (Confidence: {res['confidence']:.2%})"
        )