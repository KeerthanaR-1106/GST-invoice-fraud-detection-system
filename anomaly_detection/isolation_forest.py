"""Isolation Forest scoring and result writer.

This module runs scikit-learn's IsolationForest on the ML features CSV
and writes a results CSV with the following columns:

invoice_no,isolation_prediction,isolation_anomaly,isolation_score

- isolation_prediction: 1 (inlier) or -1 (outlier) from the estimator
- isolation_anomaly: binary flag (1 if prediction == -1 else 0)
- isolation_score: normalized anomaly score in [0,1] where larger means more anomalous

The implementation uses a reasonable default contamination value but accepts
parameters when called programmatically or from the command line.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def run_isolation_forest(
    input_csv: str = "data/ml_features.csv",
    output_csv: str = "data/isolation_forest_results.csv",
    random_state: int = 42,
    n_estimators: int = 100,
    contamination: float | str = "auto",
):
    """Run IsolationForest and save results to output_csv.

    Returns the DataFrame written.
    """

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("ML features dataframe is empty")

    X = df.select_dtypes(include=[np.number]).values

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )

    # Fit and predict
    clf.fit(X)
    preds = clf.predict(X)  # 1 (inlier) or -1 (outlier)

    # decision_function: higher => more normal (sklearn). Convert to anomaly score:
    scores = -clf.decision_function(X)

    # Normalize scores to [0,1]
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))

    if max_s - min_s > 0:
        norm_scores = (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(scores)

    out = pd.DataFrame({
        "invoice_no": df.index.to_series().apply(lambda i: df.index.name and df.index[i] or None)
    })

    # If original invoices have an invoice_no column in the upstream CSV, preserve it
    if "invoice_no" in df.columns:
        out["invoice_no"] = df["invoice_no"].astype(str)
    else:
        # fallback to index-based invoice ids
        out["invoice_no"] = df.index.astype(str)

    out["isolation_prediction"] = preds
    out["isolation_anomaly"] = (preds == -1).astype(int)
    out["isolation_score"] = norm_scores

    out.to_csv(output_csv, index=False)

    return out


if __name__ == "__main__":
    # Run with defaults when invoked as a script
    import sys

    df_out = run_isolation_forest()
    print(f"Isolation Forest: wrote {len(df_out)} rows to data/isolation_forest_results.csv")
