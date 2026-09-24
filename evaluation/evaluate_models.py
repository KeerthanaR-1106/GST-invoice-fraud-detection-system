from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_csv(path):
    return pd.read_csv(path)


def coerce_binary_label(series, name):
    cleaned = series.copy()
    cleaned = cleaned.replace({True: 1, False: 0})
    cleaned = cleaned.astype(str).str.strip()
    valid = {"0", "1", "0.0", "1.0", "true", "false"}
    if not cleaned.isin(valid).all():
        invalid = cleaned[~cleaned.isin(valid)]
        raise ValueError(f"{name} contains non-binary values: {invalid.head().tolist()}")
    cleaned = cleaned.str.lower().replace({"true": "1", "false": "0", "0.0": "0", "1.0": "1"})
    return cleaned.astype(int)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def normalize_zscore_scores(z_df):
    scores = z_df["max_abs_zscore"].copy()
    clipped = scores.clip(lower=0)
    p95 = np.percentile(clipped, 95) if not clipped.empty else 0.0
    if p95 > 0:
        normalized = np.clip(clipped, 0, p95) / p95
    else:
        normalized = np.zeros(len(clipped), dtype=float)
    return pd.Series(normalized, index=z_df.index, name="zscore_score")


def compute_confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def evaluate_method(method_name, gt_df, result_df, prediction_col, score_col):
    merged = gt_df[["invoice_no", "actual_anomaly"]].merge(
        result_df[["invoice_no", prediction_col, score_col]],
        on="invoice_no",
        how="inner",
    )

    if len(merged) != len(gt_df):
        raise ValueError(
            f"{method_name}: expected {len(gt_df)} rows after merge but got {len(merged)}. "
            f"Check invoice_no alignment."
        )

    y_true = coerce_binary_label(merged["actual_anomaly"], "ground_truth.actual_anomaly")
    y_pred = coerce_binary_label(merged[prediction_col], f"{method_name}.{prediction_col}")
    y_score = pd.to_numeric(merged[score_col], errors="coerce").fillna(0.0)

    if y_true.nunique() < 2:
        raise ValueError(f"{method_name}: ground truth label has only one class; ROC-AUC is not defined.")

    cm = compute_confusion_counts(y_true.to_numpy(), y_pred.to_numpy())
    metrics = {
        "method": method_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    metrics.update(cm)
    return merged, metrics, cm


def main():
    warnings = []

    gt_df = load_csv(DATA_DIR / "ground_truth.csv")
    if "invoice_no" not in gt_df.columns:
        raise ValueError("ground_truth.csv is missing invoice_no")
    if "actual_anomaly" not in gt_df.columns:
        raise ValueError("ground_truth.csv is missing the anomaly label column")

    gt_df = gt_df.copy()
    gt_df["invoice_no"] = gt_df["invoice_no"].astype(str)
    gt_df["actual_anomaly"] = gt_df["actual_anomaly"].astype(str).str.strip()

    if gt_df["invoice_no"].isnull().any() or (gt_df["invoice_no"] == "").any():
        warnings.append("ground_truth.csv contains blank invoice_no values")
    if gt_df["actual_anomaly"].isnull().any() or (gt_df["actual_anomaly"] == "").any():
        warnings.append("ground_truth.csv contains blank actual_anomaly values")

    if len(gt_df) != 20000:
        warnings.append(f"Expected 20,000 ground-truth records; found {len(gt_df)}")

    gt_label_distribution = gt_df["actual_anomaly"].value_counts(dropna=False)
    print("Ground-truth label distribution:")
    print(gt_label_distribution.to_string())

    if_gt = load_csv(DATA_DIR / "isolation_forest_results.csv")
    lof_df = load_csv(DATA_DIR / "lof_results.csv")
    z_df = load_csv(DATA_DIR / "zscore_results.csv")
    ensemble_df = load_csv(DATA_DIR / "ensemble_results.csv")

    method_results = []
    confusion_rows = []

    # Isolation Forest
    if_df = if_gt.copy()
    if_df["invoice_no"] = if_df["invoice_no"].astype(str)
    merged_if, metrics_if, cm_if = evaluate_method(
        "Isolation Forest",
        gt_df,
        if_df,
        prediction_col="isolation_anomaly",
        score_col="isolation_score",
    )
    method_results.append(metrics_if)
    confusion_rows.append({"method": "Isolation Forest", **cm_if})

    # LOF
    lof_df = lof_df.copy()
    lof_df["invoice_no"] = lof_df["invoice_no"].astype(str)
    merged_lof, metrics_lof, cm_lof = evaluate_method(
        "LOF",
        gt_df,
        lof_df,
        prediction_col="lof_anomaly",
        score_col="lof_score",
    )
    method_results.append(metrics_lof)
    confusion_rows.append({"method": "LOF", **cm_lof})

    # Z-score
    z_df = z_df.copy()
    z_df["invoice_no"] = z_df["invoice_no"].astype(str)
    z_df["zscore_score"] = normalize_zscore_scores(z_df)
    merged_z, metrics_z, cm_z = evaluate_method(
        "Z-score",
        gt_df,
        z_df,
        prediction_col="zscore_anomaly",
        score_col="zscore_score",
    )
    method_results.append(metrics_z)
    confusion_rows.append({"method": "Z-score", **cm_z})

    # Ensemble
    ensemble_df = ensemble_df.copy()
    ensemble_df["invoice_no"] = ensemble_df["invoice_no"].astype(str)
    merged_ensemble, metrics_ensemble, cm_ensemble = evaluate_method(
        "Ensemble",
        gt_df,
        ensemble_df,
        prediction_col="ensemble_anomaly",
        score_col="ensemble_score",
    )
    method_results.append(metrics_ensemble)
    confusion_rows.append({"method": "Ensemble", **cm_ensemble})

    eval_df = pd.DataFrame(method_results)
    eval_df = eval_df[
        [
            "method",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ]
    ]

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df = confusion_df[
        [
            "method",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ]
    ]

    eval_df.to_csv(DATA_DIR / "model_evaluation.csv", index=False)
    confusion_df.to_csv(DATA_DIR / "confusion_matrix.csv", index=False)

    print("\nEvaluation summary:")
    print(eval_df.to_string(index=False))

    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))

    print("\nFirst few rows of model_evaluation.csv:")
    print(eval_df.head().to_string(index=False))

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if len(gt_df) != len(gt_df["invoice_no"].drop_duplicates()):
        warnings.append("ground_truth.csv contains duplicate invoice_no values")

    if gt_df["invoice_no"].isna().any():
        warnings.append("ground_truth.csv contains missing invoice_no values")

    if not warnings:
        print("\nNo warnings or data issues detected.")

    print(f"\nEvaluated invoice count: {len(gt_df)}")


if __name__ == "__main__":
    main()
