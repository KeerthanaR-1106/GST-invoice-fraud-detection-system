"""Z-score anomaly detection and result writer.

Computes per-feature z-scores and records per-invoice:
- max_abs_zscore: maximum absolute z-score across features
- extreme_feature_count: number of features with abs(zscore) > 3
- zscore_anomaly: binary flag (1 if max_abs_zscore > 3 else 0)

Writes CSV with columns:
invoice_no,zscore_anomaly,max_abs_zscore,extreme_feature_count
"""

import pandas as pd
import numpy as np


def run_zscore(
    input_csv: str = "data/ml_features.csv",
    output_csv: str = "data/zscore_results.csv",
    threshold: float = 3.0,
):
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("ML features dataframe is empty")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Compute mean/std per feature
    means = numeric_df.mean(axis=0)
    stds = numeric_df.std(axis=0, ddof=0)

    # Avoid division by zero
    stds_replaced = stds.replace(0, np.nan).fillna(1.0)

    z = (numeric_df - means) / stds_replaced

    abs_z = z.abs()

    max_abs_zscore = abs_z.max(axis=1)
    extreme_feature_count = (abs_z > threshold).sum(axis=1)

    zscore_anomaly = (max_abs_zscore > threshold).astype(int)

    out = pd.DataFrame()

    if "invoice_no" in df.columns:
        out["invoice_no"] = df["invoice_no"].astype(str)
    else:
        out["invoice_no"] = df.index.astype(str)

    out["zscore_anomaly"] = zscore_anomaly
    out["max_abs_zscore"] = max_abs_zscore
    out["extreme_feature_count"] = extreme_feature_count

    out.to_csv(output_csv, index=False)

    return out


if __name__ == "__main__":
    out_df = run_zscore()
    print(f"Z-score: wrote {len(out_df)} rows to data/zscore_results.csv")