"""Local Outlier Factor scoring and result writer.

This module runs scikit-learn's LocalOutlierFactor (novelty=True) so it can be
used to compute outlier scores for the ML dataset. It writes a CSV with columns:

invoice_no,lof_prediction,lof_anomaly,lof_score

- lof_prediction: 1 (inlier) or -1 (outlier)
- lof_anomaly: binary flag (1 if prediction == -1 else 0)
- lof_score: normalized anomaly score in [0,1]
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def run_lof(
    input_csv: str = "data/ml_features.csv",
    output_csv: str = "data/lof_results.csv",
    n_neighbors: int = 20,
    novelty: bool = True,
):
    """Run LOF and save results to output_csv.

    Returns the DataFrame written.
    """

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("ML features dataframe is empty")

    X = df.select_dtypes(include=[np.number]).values

    # Use novelty=True so the estimator exposes decision_function for scoring
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=novelty)

    # When novelty=True, we must call fit and then predict/decision_function
    lof.fit(X)
    preds = lof.predict(X)  # 1 or -1

    # decision_function: higher => more normal; convert to anomaly score
    scores = -lof.decision_function(X)

    # Normalize scores to [0,1]
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))

    if max_s - min_s > 0:
        norm_scores = (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(scores)

    out = pd.DataFrame()

    if "invoice_no" in df.columns:
        out["invoice_no"] = df["invoice_no"].astype(str)
    else:
        out["invoice_no"] = df.index.astype(str)

    out["lof_prediction"] = preds
    out["lof_anomaly"] = (preds == -1).astype(int)
    out["lof_score"] = norm_scores

    out.to_csv(output_csv, index=False)

    return out


if __name__ == "__main__":
    df_out = run_lof()
    print(f"LOF: wrote {len(df_out)} rows to data/lof_results.csv")