import os
import joblib # for saving the model with all the hyperparameters
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif#ANOVA F-test for feature selection which is a statistical test to select the best features 
from sklearn.preprocessing import StandardScaler#Standardization of the data to have a mean of 0 and a standard deviation of 1 which is important for the model to converge faster and avoid overfitting by giving more weight to the most important features so we do division by the standard deviation and subtraction of the mean


def preprocess_data(
    raw_matrix_path="data/raw_gene_matrix.csv",
    metadata_path="data/samples_metadata.csv",
    k_features=500,
):
    print("--- Starting RNA-seq Preprocessing Pipeline ---")

    # 1. Load Data
    raw_df = pd.read_csv(raw_matrix_path, index_col=0)
    meta_df = pd.read_csv(metadata_path, index_col=0)

    # Align samples between matrix and metadata
    common_samples = raw_df.columns.intersection(meta_df.index)
    raw_df = raw_df[common_samples]
    meta_df = meta_df.loc[common_samples]

    # 2. Transpose Matrix: (Genes x Samples) -> (Samples x Genes)
    X_raw = raw_df.T.values  # Shape: (N_samples, P_genes)
    gene_names = raw_df.index.values

    # Map target labels ('Normal': 0, 'Tumor': 1)
    label_map = {"Normal": 0, "Tumor": 1}
    y = meta_df["condition"].map(label_map).values#values give a numpy array of the labels

    print(
        f"Initial Matrix Shape: {X_raw.shape[0]} samples, {X_raw.shape[1]} genes"
    )

    # 3. CPM Normalization: (Counts / Total Read Depth) * 1e6
    library_sizes = X_raw.sum(axis=1, keepdims=True)#For any given sample, the total read count (sequencing depth) is the sum of raw reads across all genes in that single sample.
    # Avoid division by zero if any sample has 0 total reads
    library_sizes[library_sizes == 0] = 1.0
    cpm = (X_raw / library_sizes) * 1e6

    # 4. Log2 Transformation: log2(CPM + 1)
    X_log = np.log2(cpm + 1.0)

    # 5. Feature Selection: ANOVA F-test (SelectKBest)
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X_log, y)
    selected_genes = gene_names[selector.get_support()]#get support gives a boolean array of the selected genes

    print(f"Selected Top {k_features} Marker Genes using ANOVA F-test.")

    # 6. Feature Scaling: StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 7. Save Processed Artifacts & Transformers
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_processed.npy", X_scaled)
    np.save("data/processed/y_labels.npy", y)
    np.save("data/processed/selected_genes.npy", selected_genes)

    joblib.dump(selector, "models/feature_selector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Preprocessing complete! Saved arrays to data/processed/")
    print(f"Final Feature Matrix Shape: {X_scaled.shape}")


if __name__ == "__main__":
    preprocess_data()