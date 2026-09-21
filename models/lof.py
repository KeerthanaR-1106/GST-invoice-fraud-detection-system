import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD ML FEATURES
# ============================================================

df = pd.read_csv(
    "data/ml_features.csv"
)

invoice_info = pd.read_csv(
    "data/gst_invoices.csv",
    usecols=["invoice_no"]
)

print("ML dataset loaded.")
print("Dataset shape:", df.shape)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(df)


# ============================================================
# CREATE LOF MODEL
# ============================================================

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)


# ============================================================
# FIT AND PREDICT
# ============================================================

predictions = model.fit_predict(X)


# ============================================================
# CREATE RESULTS
# ============================================================

results = pd.DataFrame()

results["invoice_no"] = (
    invoice_info["invoice_no"]
)

results["lof_prediction"] = predictions

results["lof_anomaly"] = (
    predictions == -1
).astype(int)


# ============================================================
# LOF ANOMALY SCORE
# ============================================================

results["lof_score"] = (
    -model.negative_outlier_factor_
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "data/lof_results.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nLOF completed."
)

print(
    "Normal records:",
    (
        results["lof_anomaly"] == 0
    ).sum()
)

print(
    "Anomalous records:",
    (
        results["lof_anomaly"] == 1
    ).sum()
)

print(
    "\nResults saved to:"
    " data/lof_results.csv"
)