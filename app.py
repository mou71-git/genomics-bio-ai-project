import io
from pathlib import Path
import traceback
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ==============================================================================
# SECTION 1: PYDANTIC SCHEMAS (Data Validation & JSON Serialization)
# ==============================================================================


class PredictionResult(BaseModel):
    """Schema for an individual sample's prediction result."""

    sample_id: str
    predicted_class: str
    tumor_probability: float = Field(..., ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Schema for the top-level API response payload."""

    total_samples: int
    tumor_count: int
    normal_count: int
    predictions: list[PredictionResult]


# ==============================================================================
# SECTION 2: APPLICATION INIT & MODEL ARTIFACT LOADING
# ==============================================================================

app = FastAPI(
    title="RNA-seq Biomarker Engine",
    description="Inference server using pre-trained ML models for cancer diagnostics.",
    version="1.0.0",
)

# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "staticfront"

# Mount static files to serve the index.html UI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load artifacts into RAM at startup
try:
    selector = joblib.load(MODELS_DIR / "feature_selector.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")
    rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    print(" Successfully loaded all pipeline artifacts into RAM.")
except Exception as err:
    print(f" Warning: Could not load pipeline artifacts: {err}")
    selector, scaler, lr_model, rf_model = None, None, None, None


# ==============================================================================
# SECTION 3: API CONTROLLERS & ENDPOINTS
# ==============================================================================


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the static HTML user interface."""
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(
            status_code=404, detail="static/index.html not found."
        )
    return HTMLResponse(content=html_file.read_text(), status_code=200)


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint monitored by the UI status indicator."""
    is_ready = all(
        item is not None for item in [selector, scaler, lr_model, rf_model]
    )
    return {
        "status": "healthy" if is_ready else "degraded",
        "artifacts_loaded": is_ready,
    }


@app.post("/api/v1/predict", response_model=BatchPredictionResponse)
async def predict_rna_seq(
    file: UploadFile = File(...),
    model_type: str = Form("logistic_regression"),
):
    """Receives a raw count CSV matrix, executes preprocessing, and returns predictions."""

    # 1. Ensure model artifacts are available in RAM
    if not all([selector, scaler, lr_model, rf_model]):
        raise HTTPException(
            status_code=500,
            detail="Pipeline models are not loaded on the server.",
        )

    # 2. File type validation
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a CSV."
        )

    try:
        # 3. Parse CSV directly from upload stream in memory
        file_bytes = await file.read()
        raw_df = pd.read_csv(io.BytesIO(file_bytes), index_col=0)

        if raw_df.empty:
            raise HTTPException(
                status_code=400, detail="Uploaded CSV file is empty."
            )

        sample_names = list(raw_df.columns)
        X_raw = raw_df.T.values  # Transpose to shape (N_samples, N_genes)

        # 4. Pipeline Step 1: Counts Per Million (CPM) Normalization
        library_sizes = X_raw.sum(axis=1, keepdims=True)
        library_sizes[library_sizes == 0] = 1
        X_cpm = (X_raw / library_sizes) * 1e6

        # 5. Pipeline Step 2: Log2 Transformation
        X_log = np.log2(X_cpm + 1)

        # 6. Pipeline Step 3: Apply Saved SelectKBest Transformer (.transform ONLY)
        X_selected = selector.transform(X_log)

        # 7. Pipeline Step 4: Apply Saved StandardScaler Transformer (.transform ONLY)
        X_scaled = scaler.transform(X_selected)

        # 8. Select Classifier Architecture
        chosen_model = (
            rf_model if model_type == "random_forest" else lr_model
        )

        # 9. Execute Model Inference
        raw_preds = chosen_model.predict(X_scaled)
        probabilities = chosen_model.predict_proba(X_scaled)[:, 1]

        # 10. Format predictions into structured response objects
        prediction_records = []
        tumor_count = 0
        normal_count = 0

        for name, pred, proba in zip(sample_names, raw_preds, probabilities):
            is_tumor = bool(pred == 1)
            if is_tumor:
                tumor_count += 1
            else:
                normal_count += 1

            prediction_records.append(
                PredictionResult(
                    sample_id=str(name),
                    predicted_class="Tumor" if is_tumor else "Normal",
                    tumor_probability=float(proba),
                )
            )

        # Return validated Pydantic model
        return BatchPredictionResponse(
            total_samples=len(prediction_records),
            tumor_count=tumor_count,
            normal_count=normal_count,
            predictions=prediction_records,
        )

    except Exception as err:
        traceback.print_exc()
        raise HTTPException(    
            status_code=500, detail=f"Inference error: {str(err)}"
        )