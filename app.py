from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from evaluation import evaluate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

features = metadata["feature_names"]
target = metadata["target_column"]
model_files = metadata["model_files"]

st.set_page_config(page_title="WDBC Model Comparison")
st.title("Breast mass classification")
st.write(
    "Compare five trained classifiers on labeled test records. "
    "The target is 0 for benign and 1 for malignant."
)

st.download_button(
    "Download sample test CSV",
    data=(ROOT / "test_data.csv").read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload a labeled test CSV", type=["csv"])
selected = st.selectbox("Choose a classifier", list(model_files))

if uploaded is None:
    st.info("Download the sample CSV above, then upload it to see the results.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except (ValueError, UnicodeError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

if data.empty:
    st.error("The CSV must contain at least one test record.")
    st.stop()

missing = [column for column in [*features, target] if column not in data]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

try:
    # Select features by their saved names and training order.
    X_upload = data[features].apply(pd.to_numeric, errors="raise")
    y_upload = pd.to_numeric(data[target], errors="raise")
except (ValueError, TypeError):
    st.error("The feature columns and target must contain numeric values.")
    st.stop()

if not np.isfinite(X_upload.to_numpy()).all():
    st.error("The features contain missing or infinite values.")
    st.stop()

if not y_upload.isin([0, 1]).all():
    st.error("Every target must be 0 (benign) or 1 (malignant).")
    st.stop()

# Load the fitted model and evaluate the uploaded rows.
model = joblib.load(MODEL_DIR / model_files[selected])
metrics, predictions = evaluate(model, X_upload, y_upload)

st.subheader(f"{selected} results")
st.write(f"Test records evaluated: {len(data)}")

metric_table = pd.DataFrame({
    "Metric": list(metrics),
    "Value": [
        "Unavailable (one class)" if pd.isna(value) else f"{value:.4f}"
        for value in metrics.values()
    ],
})
st.table(metric_table.set_index("Metric"))

matrix = confusion_matrix(y_upload, predictions, labels=[0, 1])
st.subheader("Confusion matrix")
st.table(pd.DataFrame(
    matrix,
    index=["Actual benign", "Actual malignant"],
    columns=["Predicted benign", "Predicted malignant"],
))