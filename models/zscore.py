import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
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
# FEATURES FOR Z-SCORE
# ============================================================

zscore_features = [
    "quantity",
    "unit_price",
    "discount_percentage",
    "taxable_value",
    "gst_rate",
    "cgst",
    "sgst",
    "igst",
    "cess_amount",
    "total_tax",
    "total_amount"
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in zscore_features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        + str(missing_features)
    )


# ============================================================
# CALCULATE Z-SCORES
# ============================================================

z_scores = pd.DataFrame(
    index=df.index
)


for feature in zscore_features:

    mean = df[feature].mean()

    std = df[feature].std()

    if std == 0:

        z_scores[feature] = 0

    else:

        z_scores[feature] = (
            (df[feature] - mean)
            / std
        )


# ============================================================
# FIND MAXIMUM ABSOLUTE Z-SCORE
# ============================================================

absolute_z_scores = (
    z_scores.abs()
)

max_z_score = (
    absolute_z_scores.max(
        axis=1
    )
)


# ============================================================
# COUNT EXTREME FEATURES
# ============================================================

extreme_feature_count = (
    absolute_z_scores >= 3
).sum(
    axis=1
)


# ============================================================
# Z-SCORE ANOMALY RULE
# ============================================================

# Baseline threshold:
# |Z| >= 3 for at least one feature

zscore_anomaly = (
    max_z_score >= 3
).astype(int)


# ============================================================
# CREATE RESULTS
# ============================================================

results = pd.DataFrame()

results["invoice_no"] = (
    invoice_info["invoice_no"]
)

results["zscore_anomaly"] = (
    zscore_anomaly
)

results["max_abs_zscore"] = (
    max_z_score
)

results["extreme_feature_count"] = (
    extreme_feature_count
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "data/zscore_results.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nZ-score detection completed."
)

print(
    "Threshold: |Z| >= 3"
)

print(
    "Normal records:",
    (
        results["zscore_anomaly"] == 0
    ).sum()
)

print(
    "Anomalous records:",
    (
        results["zscore_anomaly"] == 1
    ).sum()
)

print(
    "\nResults saved to:"
    " data/zscore_results.csv"
)