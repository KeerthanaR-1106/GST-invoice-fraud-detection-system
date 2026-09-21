import pandas as pd


# ==========================================
# 1. LOAD MODEL RESULTS
# ==========================================

results = pd.read_csv(
    "data/isolation_forest_results.csv"
)


# ==========================================
# 2. LOAD GROUND TRUTH
# ==========================================

ground_truth = pd.read_csv(
    "data/ground_truth.csv"
)


# ==========================================
# 3. MERGE DATA
# ==========================================

evaluation = results.merge(
    ground_truth,
    on="invoice_no",
    how="inner"
)


# ==========================================
# 4. ANALYZE ANOMALY TYPES
# ==========================================

anomalies = evaluation[
    evaluation["actual_anomaly"] == 1
]


summary = anomalies.groupby(
    "anomaly_type"
)["isolation_anomaly"].agg(
    total="count",
    detected="sum"
)


# ==========================================
# 5. CALCULATE DETECTION RATE
# ==========================================

summary["detection_rate"] = (
    summary["detected"] / summary["total"] * 100
)


# ==========================================
# 6. DISPLAY
# ==========================================

print("\nDetection by anomaly type:")
print(summary)