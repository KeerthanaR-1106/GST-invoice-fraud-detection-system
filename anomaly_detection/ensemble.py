"""Ensemble scoring for Phase 3.

This module reads the three per-method result CSVs (Isolation Forest, LOF, Z-score),
normalizes the z-score method into a 0-1 range, and computes an ensemble score as

    ensemble_score = (isolation_score + lof_score + zscore_score) / 3

Normalization details:
- Isolation Forest: uses isolation_score as-is (already normalized to [0,1] by the IF module).
- LOF: uses lof_score as-is (already normalized to [0,1] by the LOF module).
- Z-score: max_abs_zscore is positive and unbounded. To map it to [0,1], this script
  computes the 95th percentile (p95) of max_abs_zscore and caps values at p95; the
  normalized zscore_score = clipped_z / p95. This maps typical z-scores to [0,1] and
  saturates extreme outliers at 1.0. If p95 == 0, the zscore_score is set to 0.

Ensemble anomaly indicator (temporary/simple): ensemble_anomaly = 1 when ensemble_score > 0.5.
This threshold is deliberately simple and may be refined in Phase 4.

Output CSV: data/ensemble_results.csv with columns:
- invoice_no
- isolation_score
- lof_score
- zscore_score
- ensemble_score
- ensemble_anomaly

"""

import pandas as pd
import numpy as np


def run_ensemble(
    isolation_csv: str = "data/isolation_forest_results.csv",
    lof_csv: str = "data/lof_results.csv",
    zscore_csv: str = "data/zscore_results.csv",
    output_csv: str = "data/ensemble_results.csv",
    zscore_pctl: float = 95.0,
    anomaly_threshold: float = 0.5,
):
    # Load
    if_df = pd.read_csv(isolation_csv)
    lof_df = pd.read_csv(lof_csv)
    z_df = pd.read_csv(zscore_csv)

    # Inspect columns
    required_if = ["invoice_no", "isolation_score"]
    required_lof = ["invoice_no", "lof_score"]
    required_z = ["invoice_no", "max_abs_zscore"]

    for col in required_if:
        if col not in if_df.columns:
            raise ValueError(f"Missing column in isolation CSV: {col}")
    for col in required_lof:
        if col not in lof_df.columns:
            raise ValueError(f"Missing column in lof CSV: {col}")
    for col in required_z:
        if col not in z_df.columns:
            raise ValueError(f"Missing column in zscore CSV: {col}")

    # Keep only needed columns and rename to standard names
    if_sub = if_df[["invoice_no", "isolation_score"]].copy()
    lof_sub = lof_df[["invoice_no", "lof_score"]].copy()
    z_sub = z_df[["invoice_no", "max_abs_zscore"]].copy()

    # Merge on invoice_no
    merged = if_sub.merge(lof_sub, on="invoice_no", how="outer", validate="one_to_one")
    merged = merged.merge(z_sub, on="invoice_no", how="outer", validate="one_to_one")

    # Check for missing or duplicate invoice IDs
    total_invoices = len(merged)
    missing_if = merged["isolation_score"].isna().sum()
    missing_lof = merged["lof_score"].isna().sum()
    missing_z = merged["max_abs_zscore"].isna().sum()

    dup_count = merged[merged.duplicated(subset=["invoice_no"], keep=False)].shape[0]

    # Normalize z-score: cap at pctl and divide
    p95 = np.percentile(merged["max_abs_zscore"].fillna(0).values, zscore_pctl)
    if p95 <= 0:
        # avoid div by zero
        merged["zscore_score"] = 0.0
    else:
        clipped = merged["max_abs_zscore"].clip(upper=p95)
        merged["zscore_score"] = (clipped / p95).astype(float)

    # Ensure isolation_score and lof_score are numeric and within [0,1]
    merged["isolation_score"] = pd.to_numeric(merged["isolation_score"], errors="coerce").fillna(0.0)
    merged["lof_score"] = pd.to_numeric(merged["lof_score"], errors="coerce").fillna(0.0)

    # Clip to [0,1] just in case
    merged["isolation_score"] = merged["isolation_score"].clip(0.0, 1.0)
    merged["lof_score"] = merged["lof_score"].clip(0.0, 1.0)
    merged["zscore_score"] = merged["zscore_score"].clip(0.0, 1.0)

    # Compute ensemble
    merged["ensemble_score"] = (
        merged["isolation_score"] + merged["lof_score"] + merged["zscore_score"]
    ) / 3.0

    merged["ensemble_anomaly"] = (merged["ensemble_score"] > anomaly_threshold).astype(int)

    # Save selected columns
    out_cols = [
        "invoice_no",
        "isolation_score",
        "lof_score",
        "zscore_score",
        "ensemble_score",
        "ensemble_anomaly",
    ]

    merged[out_cols].to_csv(output_csv, index=False)

    # Return diagnostics and df
    diagnostics = {
        "total_invoices": total_invoices,
        "missing_if": int(missing_if),
        "missing_lof": int(missing_lof),
        "missing_z": int(missing_z),
        "duplicates": int(dup_count),
        "p95_max_abs_zscore": float(p95),
        "anomaly_threshold_used": anomaly_threshold,
    }

    return merged[out_cols], diagnostics


if __name__ == "__main__":
    df_out, diag = run_ensemble()
    print("Wrote", len(df_out), "rows to data/ensemble_results.csv")
    print("Diagnostics:", diag)
