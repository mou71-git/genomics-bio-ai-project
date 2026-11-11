import pandas as pd

# 1. Read expression file
raw_df = pd.read_csv("data/GSE183947_fpkm.csv", index_col=0)
raw_df.index.name = "gene_id"

sample_ids = list(raw_df.columns)

# 2. Corrected label assignment based on sample prefix
conditions = []
for col in sample_ids:
    if col.startswith("CA."):
        conditions.append("Tumor")
    elif col.startswith("CAP."):
        conditions.append("Normal")
    else:
        # Fallback if names vary slightly
        conditions.append("Tumor" if "CAP" not in col else "Normal")

# 3. Rebuild metadata table
metadata_df = pd.DataFrame({"condition": conditions}, index=sample_ids)
metadata_df.index.name = "sample_id"

# 4. Export clean CSVs
raw_df.to_csv("data/raw_gene_matrix.csv")
metadata_df.to_csv("data/samples_metadata.csv")

print("Updated Metadata successfully!")
print(metadata_df["condition"].value_counts())