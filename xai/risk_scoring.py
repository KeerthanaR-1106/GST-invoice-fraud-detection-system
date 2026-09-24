"""Phase 4: Risk scoring and simple XAI explanations.

Produces data/risk_results.csv with columns:
- invoice_no
- isolation_score
- lof_score
- zscore_score
- ensemble_score
- final_risk_score
- risk_level
- explanation

Notes:
- final_risk_score = ensemble_score (keeps between 0 and 1)
- risk_level thresholds (documented):
    0.00 - 0.33 -> Low
    0.34 - 0.66 -> Medium
    0.67 - 1.00 -> High
  Implementation uses: <=0.33 Low; <=0.66 Medium; else High.
- Explanations combine signals from isolation forest, LOF, z-score and simple
  per-feature z-score checks (abs(z) > 3) as well as flags like duplicate or future_date.

This script is intentionally conservative in language ("anomaly detected", "requires review").
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(".")
DATA = ROOT / os.environ.get("GST_DATA_DIR", "data")

# Input files
ENSEMBLE_CSV = DATA / os.environ.get("GST_ENSEMBLE_CSV", "ensemble_results.csv")
IF_CSV = DATA / os.environ.get("GST_IF_CSV", "isolation_forest_results.csv")
LOF_CSV = DATA / os.environ.get("GST_LOF_CSV", "lof_results.csv")
Z_CSV = DATA / os.environ.get("GST_Z_CSV", "zscore_results.csv")
ML_FEATURES_CSV = DATA / os.environ.get("GST_ML_FEATURES_CSV", "ml_features.csv")

OUT_CSV = DATA / os.environ.get("GST_RISK_OUT_CSV", "risk_results.csv")


def load_sources():
    ensemble = pd.read_csv(ENSEMBLE_CSV)
    if_df = pd.read_csv(IF_CSV)
    lof_df = pd.read_csv(LOF_CSV)
    z_df = pd.read_csv(Z_CSV)
    ml_df = pd.read_csv(ML_FEATURES_CSV)
    return ensemble, if_df, lof_df, z_df, ml_df


def compute_feature_extremes(ml_df, z_threshold=3.0):
    """Compute per-row feature z-scores and return a DataFrame of booleans where abs(z) > threshold."""
    numeric = ml_df.select_dtypes(include=[np.number]).copy()
    # compute population std (ddof=0) to match earlier usage
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0)
    stds_replaced = stds.replace(0, np.nan).fillna(1.0)
    z = (numeric - means) / stds_replaced
    abs_z = z.abs()
    extremes = abs_z > z_threshold
    return extremes, z


# Friendly feature name mapping for explanations
FEATURE_MESSAGES = {
    "quantity": "Unusual quantity detected",
    "unit_price": "Unusual unit price detected",
    "discount_amount": "Unusual discount amount detected",
    "discount_percentage": "Unusual discount percentage detected",
    "taxable_value": "Unusual taxable value detected",
    "gst_difference": "GST calculation difference detected",
    "total_tax_difference": "Total tax mismatch detected",
    "total_amount_difference": "Total amount mismatch detected",
    "cgst_sgst_split_difference": "CGST/SGST split inconsistency detected",
    "duplicate_transaction_flag": "Duplicate transaction pattern detected",
    "supplier_frequency": "Unusual supplier activity detected",
    "buyer_frequency": "Unusual buyer activity detected",
}


def explanation_for_row(row, feature_extremes_row, ml_row, zscore_row):
    """Assemble explanation strings for a single invoice.

    row: merged ensemble/flags row containing invoice_no, isolation/lof anomaly flags
    feature_extremes_row: boolean series for which numeric features exceed z-threshold
    ml_row: original ML features row (to inspect flags like duplicate_transaction_flag)
    zscore_row: row from zscore_results (contains extreme_feature_count etc.)
    """
    notes = []

    # Method-level signals
    if row.get("isolation_anomaly", 0) == 1:
        notes.append("Isolation Forest detected an anomalous pattern (requires review)")

    if row.get("lof_anomaly", 0) == 1:
        notes.append("Local Outlier Factor detected a local anomaly (requires review)")

    # zscore anomaly flag
    if zscore_row.get("zscore_anomaly", 0) == 1:
        notes.append("Z-score analysis detected extreme feature values")

    # Feature-level signals from per-feature z-scores
    # Map feature names to messages where available
    extreme_features = feature_extremes_row[feature_extremes_row].index.tolist()

    for feat in extreme_features:
        msg = FEATURE_MESSAGES.get(feat, None)
        if msg:
            notes.append(msg + " (high z-score)")
        else:
            # fallback generic message
            notes.append(f"Anomalous feature detected: {feat}")

    # Binary flags that may not be in numeric extremes (ensure these are checked)
    if "duplicate_transaction_flag" in ml_row and int(ml_row.get("duplicate_transaction_flag", 0)) == 1:
        # avoid duplication of message if already added
        if not any("Duplicate transaction" in n for n in notes):
            notes.append("Duplicate transaction pattern detected (requires review)")

    # If no signals were appended, provide a conservative default explanation
    if len(notes) == 0:
        notes.append("No strong method-level or feature-level anomaly flags, but manual review may be useful")

    # Concatenate into a human-readable explanation (semi-colon separated)
    explanation = "; ".join(notes)
    return explanation


def assign_risk_level(score: float) -> str:
    """Assigns risk level based on score with documented thresholds.

    Thresholds (initial, not validated):
      - 0.00 <= score <= 0.33 : Low
      - 0.34 <= score <= 0.66 : Medium
      - 0.67 <= score <= 1.00 : High

    Implementation: <=0.33 Low; <=0.66 Medium; else High.
    """
    if score <= 0.33:
        return "Low"
    elif score <= 0.66:
        return "Medium"
    else:
        return "High"


def run_phase4(out_csv=OUT_CSV):
    ensemble, if_df, lof_df, z_df, ml_df = load_sources()

    # Merge required info (use ensemble as base which has scores)
    merged = ensemble.merge(if_df[["invoice_no", "isolation_prediction", "isolation_anomaly"]], on="invoice_no", how="left")
    merged = merged.merge(lof_df[["invoice_no", "lof_prediction", "lof_anomaly"]], on="invoice_no", how="left")
    merged = merged.merge(z_df[["invoice_no", "zscore_anomaly", "max_abs_zscore", "extreme_feature_count"]], on="invoice_no", how="left")

    # Fill NaNs for anomaly flags (treat as not anomalous)
    for col in ["isolation_anomaly", "lof_anomaly", "zscore_anomaly"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # Prepare per-feature extremes
    extremes_df, z_scores_df = compute_feature_extremes(ml_df, z_threshold=3.0)

    # Align indices: ml_df and merged should align by invoice order, but to be safe, merge ml by invoice_no
    # ml_df contains invoice_no as first column per Phase 2 fix
    if "invoice_no" in ml_df.columns:
        ml_by_invoice = ml_df.set_index("invoice_no")
    else:
        ml_by_invoice = ml_df.set_index(ml_df.index.astype(str))

    # Build explanation for every invoice in merged
    explanations = []
    final_scores = []
    risk_levels = []

    # For performance, create a mapping of extremes by invoice_no
    # extremes_df has same index as ml_df; ensure mapping via ml_by_invoice.index order
    extremes_df_indexed = extremes_df.copy()
    extremes_df_indexed.index = ml_by_invoice.index

    # Iterate over merged rows
    for idx, row in merged.iterrows():
        inv = row["invoice_no"]
        # feature extremes row: if invoice exists in extremes_df_indexed
        if inv in extremes_df_indexed.index:
            feat_ext_row = extremes_df_indexed.loc[inv]
            # get boolean series where True
            feat_ext_true = feat_ext_row[feat_ext_row == True]
        else:
            feat_ext_true = pd.Series(dtype=bool)

        # ml_row for flags
        if inv in ml_by_invoice.index:
            ml_row = ml_by_invoice.loc[inv]
        else:
            ml_row = {}

        # corresponding zscore row
        z_row = z_df[z_df["invoice_no"] == inv]
        z_row = z_row.iloc[0].to_dict() if not z_row.empty else {}

        explanation = explanation_for_row(row, feat_ext_true, ml_row, z_row)

        # final risk score = ensemble_score (already between 0 and 1)
        final_score = float(row.get("ensemble_score", 0.0))
        # Clamp to [0,1]
        final_score = max(0.0, min(1.0, final_score))

        risk_level = assign_risk_level(final_score)

        explanations.append(explanation)
        final_scores.append(final_score)
        risk_levels.append(risk_level)

    # Attach
    merged["final_risk_score"] = final_scores
    merged["risk_level"] = risk_levels
    merged["explanation"] = explanations

    # Select and save required columns
    out_cols = [
        "invoice_no",
        "isolation_score",
        "lof_score",
        "zscore_score",
        "ensemble_score",
        "final_risk_score",
        "risk_level",
        "explanation",
    ]

    merged[out_cols].to_csv(out_csv, index=False)

    # Verification info
    total = len(merged)
    missing_invoice = merged["invoice_no"].isna().any()
    dup_count = merged["invoice_no"].duplicated().sum()

    # counts per risk level
    counts = merged["risk_level"].value_counts().to_dict()

    return out_csv, {
        "total_rows": int(total),
        "missing_invoice_no": bool(missing_invoice),
        "duplicate_invoice_no_count": int(dup_count),
        "risk_level_counts": counts,
    }


if __name__ == "__main__":
    out_csv, info = run_phase4()
    print("Wrote:", out_csv)
    print("Info:", info)
