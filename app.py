"""Flask backend for the GST Invoice Fraud Detection system.

This phase adds a safe CSV upload pipeline while preserving the original
ML algorithm implementations and project data files.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
REQUIRED_INVOICE_COLUMNS = [
    "invoice_no",
    "invoice_date",
    "supplier_id",
    "supplier_gstin",
    "buyer_id",
    "buyer_gstin",
    "product_category",
    "hsn_code",
    "quantity",
    "unit_price",
    "discount_percentage",
    "discount_amount",
    "taxable_value",
    "gst_rate",
    "cgst",
    "sgst",
    "igst",
    "cess_rate",
    "cess_amount",
    "total_tax",
    "total_amount",
    "payment_method",
]


def read_csv_safely(csv_name):
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        return None, f"CSV file not found: {csv_path.name}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return None, f"Could not read {csv_path.name}: {exc}"

    if df.empty:
        return None, f"CSV file is empty: {csv_path.name}"

    return df, None


def load_risk_results():
    df, error = read_csv_safely("risk_results.csv")
    if df is None:
        return None, error

    required_columns = ["invoice_no", "final_risk_score", "risk_level", "explanation"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return None, f"Missing required columns in risk_results.csv: {missing}"

    df = df.copy()
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["final_risk_score"] = pd.to_numeric(df["final_risk_score"], errors="coerce")
    df["risk_level"] = df["risk_level"].fillna("Unknown").astype(str)
    return df, None


def load_invoices():
    df, error = read_csv_safely("gst_invoices.csv")
    if df is None:
        return None, error

    required_columns = [
        "invoice_no",
        "invoice_date",
        "supplier_id",
        "supplier_gstin",
        "buyer_id",
        "buyer_gstin",
        "product_category",
        "hsn_code",
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
        "total_amount",
        "payment_method",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return None, f"Missing required columns in gst_invoices.csv: {missing}"

    df = df.copy()
    df["invoice_no"] = df["invoice_no"].astype(str)
    return df, None


def load_model_evaluation():
    df, error = read_csv_safely("model_evaluation.csv")
    if df is None:
        return None, error
    return df, None


def load_confusion_matrix():
    df, error = read_csv_safely("confusion_matrix.csv")
    if df is None:
        return None, error
    return df, None


def validate_uploaded_csv(file_storage):
    if file_storage is None or file_storage.filename in (None, ""):
        return None, "Please choose a CSV file to upload."

    filename = secure_filename(file_storage.filename)
    if not filename or not filename.lower().endswith(".csv"):
        return None, "Only CSV files are allowed."

    if file_storage.content_length and file_storage.content_length > MAX_UPLOAD_SIZE:
        return None, "Uploaded file is too large. Please upload a CSV under 10 MB."

    safe_name = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}.csv"
    save_path = UPLOAD_DIR / safe_name

    try:
        file_storage.save(str(save_path))
    except Exception as exc:
        return None, f"Could not save uploaded file: {exc}"

    try:
        df = pd.read_csv(save_path)
    except Exception as exc:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return None, f"Could not read the uploaded CSV: {exc}"

    if df.empty:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return None, "The uploaded CSV is empty."

    missing = [column for column in REQUIRED_INVOICE_COLUMNS if column not in df.columns]
    if missing:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return None, f"Missing required columns: {', '.join(missing)}"

    return df, save_path


def run_uploaded_pipeline(upload_csv_path):
    upload_name = Path(upload_csv_path).stem
    ml_output = UPLOAD_DIR / f"{upload_name}_ml_features.csv"
    if_output = UPLOAD_DIR / f"{upload_name}_isolation_forest_results.csv"
    lof_output = UPLOAD_DIR / f"{upload_name}_lof_results.csv"
    z_output = UPLOAD_DIR / f"{upload_name}_zscore_results.csv"
    ensemble_output = UPLOAD_DIR / f"{upload_name}_ensemble_results.csv"
    risk_output = UPLOAD_DIR / f"{upload_name}_risk_results.csv"

    env_preprocess = os.environ.copy()
    env_preprocess["GST_PREPROCESS_INPUT"] = str(upload_csv_path)
    env_preprocess["GST_PREPROCESS_OUTPUT"] = str(ml_output)

    preprocess_result = subprocess.run(
        [sys.executable, str(BASE_DIR / "preprocessing" / "preprocess.py")],
        cwd=str(BASE_DIR),
        env=env_preprocess,
        capture_output=True,
        text=True,
    )
    if preprocess_result.returncode != 0:
        raise RuntimeError(preprocess_result.stderr.strip() or "Preprocessing failed.")

    from anomaly_detection.isolation_forest import run_isolation_forest
    from anomaly_detection.lof import run_lof
    from anomaly_detection.zscore import run_zscore
    from anomaly_detection.ensemble import run_ensemble

    run_isolation_forest(str(ml_output), str(if_output))
    run_lof(str(ml_output), str(lof_output))
    run_zscore(str(ml_output), str(z_output))
    run_ensemble(str(if_output), str(lof_output), str(z_output), str(ensemble_output))

    env_risk = os.environ.copy()
    env_risk["GST_DATA_DIR"] = "data/uploads"
    env_risk["GST_ENSEMBLE_CSV"] = f"{upload_name}_ensemble_results.csv"
    env_risk["GST_IF_CSV"] = f"{upload_name}_isolation_forest_results.csv"
    env_risk["GST_LOF_CSV"] = f"{upload_name}_lof_results.csv"
    env_risk["GST_Z_CSV"] = f"{upload_name}_zscore_results.csv"
    env_risk["GST_ML_FEATURES_CSV"] = f"{upload_name}_ml_features.csv"
    env_risk["GST_RISK_OUT_CSV"] = f"{upload_name}_risk_results.csv"

    risk_result = subprocess.run(
        [sys.executable, str(BASE_DIR / "xai" / "risk_scoring.py")],
        cwd=str(BASE_DIR),
        env=env_risk,
        capture_output=True,
        text=True,
    )
    if risk_result.returncode != 0:
        raise RuntimeError(risk_result.stderr.strip() or "Risk scoring failed.")

    return risk_output


def load_uploaded_risk_results(filename):
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return None, "Invalid uploaded result name."

    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return None, "Uploaded result file not found."

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return None, f"Could not read uploaded result: {exc}"

    if df.empty:
        return None, "Uploaded result file is empty."

    required = ["invoice_no", "final_risk_score", "risk_level", "explanation"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None, f"Uploaded result file is missing columns: {missing}"

    df = df.copy()
    df["final_risk_score"] = pd.to_numeric(df["final_risk_score"], errors="coerce")
    df = df.sort_values("final_risk_score", ascending=False)
    return df, None


def build_dashboard_summary(risk_df):
    if risk_df is None or risk_df.empty:
        return {
            "total_invoices": 0,
            "low_risk_count": 0,
            "medium_risk_count": 0,
            "high_risk_count": 0,
            "total_anomaly_count": 0,
            "average_risk_score": 0.0,
            "invoices_requiring_review": 0,
        }

    total_invoices = int(len(risk_df))
    low_risk_count = int((risk_df["risk_level"] == "Low").sum())
    medium_risk_count = int((risk_df["risk_level"] == "Medium").sum())
    high_risk_count = int((risk_df["risk_level"] == "High").sum())
    total_anomaly_count = int((risk_df["risk_level"].isin(["Medium", "High"])).sum())
    average_risk_score = float(risk_df["final_risk_score"].mean()) if total_invoices else 0.0
    return {
        "total_invoices": total_invoices,
        "low_risk_count": low_risk_count,
        "medium_risk_count": medium_risk_count,
        "high_risk_count": high_risk_count,
        "total_anomaly_count": total_anomaly_count,
        "average_risk_score": average_risk_score,
        "invoices_requiring_review": total_anomaly_count,
    }


def build_risk_distribution(summary):
    return {
        "labels": ["Low", "Medium", "High"],
        "values": [
            summary["low_risk_count"],
            summary["medium_risk_count"],
            summary["high_risk_count"],
        ],
    }


def build_score_distribution(risk_df):
    if risk_df is None or risk_df.empty:
        return {"labels": [], "values": []}

    values = pd.to_numeric(risk_df["final_risk_score"], errors="coerce").dropna().tolist()
    if not values:
        return {"labels": [], "values": []}

    hist, edges = np.histogram(values, bins=10, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": hist.astype(int).tolist()}


def build_model_chart(metrics_df):
    if metrics_df is None or metrics_df.empty:
        return {"labels": [], "datasets": []}

    return {
        "labels": metrics_df["method"].astype(str).tolist(),
        "datasets": [
            {
                "label": "Accuracy",
                "data": metrics_df["accuracy"].astype(float).tolist(),
                "backgroundColor": "rgba(13, 110, 253, 0.6)",
            },
            {
                "label": "Precision",
                "data": metrics_df["precision"].astype(float).tolist(),
                "backgroundColor": "rgba(40, 167, 69, 0.6)",
            },
            {
                "label": "Recall",
                "data": metrics_df["recall"].astype(float).tolist(),
                "backgroundColor": "rgba(255, 193, 7, 0.6)",
            },
            {
                "label": "F1 Score",
                "data": metrics_df["f1_score"].astype(float).tolist(),
                "backgroundColor": "rgba(220, 53, 69, 0.6)",
            },
            {
                "label": "ROC-AUC",
                "data": metrics_df["roc_auc"].astype(float).tolist(),
                "backgroundColor": "rgba(23, 162, 184, 0.6)",
            },
        ],
    }


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.errorhandler(404)
    def handle_not_found(error):
        return render_template("error.html", message="Page not found."), 404

    @app.route("/")
    def index():
        risk_df, error = load_risk_results()
        if risk_df is None:
            summary = build_dashboard_summary(None)
            return render_template(
                "index.html",
                summary=summary,
                risk_distribution={"labels": ["Low", "Medium", "High"], "values": [0, 0, 0]},
                score_distribution={"labels": [], "values": []},
                high_risk_rows=[],
                suspicious_rows=[],
                error=error,
            )

        summary = build_dashboard_summary(risk_df)
        risk_distribution = build_risk_distribution(summary)
        score_distribution = build_score_distribution(risk_df)

        high_risk_rows = (
            risk_df[risk_df["risk_level"] == "High"]
            .sort_values("final_risk_score", ascending=False)
            [["invoice_no", "final_risk_score", "risk_level", "explanation"]]
            .head(10)
            .to_dict("records")
        )

        suspicious_rows = (
            risk_df.sort_values("final_risk_score", ascending=False)
            [["invoice_no", "final_risk_score", "risk_level", "explanation"]]
            .head(10)
            .to_dict("records")
        )

        return render_template(
            "index.html",
            summary=summary,
            risk_distribution=risk_distribution,
            score_distribution=score_distribution,
            high_risk_rows=high_risk_rows,
            suspicious_rows=suspicious_rows,
            error=None,
        )

    @app.route("/results")
    def results():
        risk_df, error = load_risk_results()
        if risk_df is None:
            return render_template("results.html", rows=[], error=error, filters={"q": "", "risk_level": "all", "sort": "desc"})

        q = (request.args.get("q") or "").strip()
        selected_level = (request.args.get("risk_level") or "all").strip()
        sort_order = request.args.get("sort") or "desc"

        filtered = risk_df.copy()
        if q:
            filtered = filtered[filtered["invoice_no"].astype(str).str.contains(q, case=False, na=False)]
        if selected_level != "all":
            filtered = filtered[filtered["risk_level"].astype(str) == selected_level]
        filtered = filtered.sort_values("final_risk_score", ascending=(sort_order == "asc"))

        rows = filtered[["invoice_no", "final_risk_score", "risk_level", "explanation"]].to_dict("records")
        return render_template(
            "results.html",
            rows=rows,
            error=None,
            filters={"q": q, "risk_level": selected_level, "sort": sort_order},
        )

    @app.route("/invoice/<invoice_no>")
    def invoice_detail(invoice_no):
        invoice_no = str(invoice_no).strip()

        invoice_df, invoice_error = load_invoices()
        risk_df, risk_error = load_risk_results()
        if invoice_df is None or risk_df is None:
            return render_template(
                "invoice_details.html",
                error=(invoice_error or risk_error or "Unable to load invoice data."),
                invoice_no=invoice_no,
                invoice=None,
                risk_record=None,
            ), 404

        invoice_row = invoice_df[invoice_df["invoice_no"].astype(str) == invoice_no]
        risk_row = risk_df[risk_df["invoice_no"].astype(str) == invoice_no]

        if invoice_row.empty or risk_row.empty:
            return render_template(
                "invoice_details.html",
                error=f"Invoice not found: {invoice_no}",
                invoice_no=invoice_no,
                invoice=None,
                risk_record=None,
            ), 404

        invoice = invoice_row.iloc[0].to_dict()
        risk_record = risk_row.iloc[0].to_dict()
        risk_fields = [
            "isolation_score",
            "lof_score",
            "zscore_score",
            "ensemble_score",
            "final_risk_score",
            "risk_level",
            "explanation",
        ]
        risk_record = {key: risk_record.get(key) for key in risk_fields}

        return render_template(
            "invoice_details.html",
            invoice_no=invoice_no,
            invoice=invoice,
            risk_record=risk_record,
            error=None,
        )

    @app.route("/evaluation")
    def evaluation():
        metrics_df, metrics_error = load_model_evaluation()
        confusion_df, confusion_error = load_confusion_matrix()

        if metrics_df is None or confusion_df is None:
            return render_template(
                "evaluation.html",
                metrics=[],
                confusion=[],
                model_chart_data={"labels": [], "datasets": []},
                error=(metrics_error or confusion_error or "Unable to load evaluation data."),
            )

        metrics = metrics_df.to_dict("records")
        confusion = confusion_df.to_dict("records")
        model_chart_data = build_model_chart(metrics_df)
        return render_template(
            "evaluation.html",
            metrics=metrics,
            confusion=confusion,
            model_chart_data=model_chart_data,
            error=None,
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            file_storage = request.files.get("file")
            if not file_storage:
                return render_template("upload.html", message="No file selected."), 400

            df, validated = validate_uploaded_csv(file_storage)
            if df is None:
                return render_template("upload.html", message=validated), 400

            try:
                risk_output = run_uploaded_pipeline(validated)
            except Exception as exc:
                if Path(validated).exists():
                    Path(validated).unlink(missing_ok=True)
                return render_template(
                    "upload.html",
                    message=f"Upload processing failed: {exc}",
                ), 400

            filename = Path(risk_output).name
            return redirect(url_for("upload_results", filename=filename))

        return render_template("upload.html", message=None)

    @app.route("/upload-results/<filename>")
    def upload_results(filename):
        df, error = load_uploaded_risk_results(filename)
        if df is None:
            return render_template("uploaded_results.html", rows=[], summary={}, error=error), 404

        low_risk_count = int((df["risk_level"] == "Low").sum())
        medium_risk_count = int((df["risk_level"] == "Medium").sum())
        high_risk_count = int((df["risk_level"] == "High").sum())
        average_risk_score = float(df["final_risk_score"].mean()) if not df.empty else 0.0
        summary = {
            "uploaded_invoices": int(len(df)),
            "low_risk_count": low_risk_count,
            "medium_risk_count": medium_risk_count,
            "high_risk_count": high_risk_count,
            "average_risk_score": average_risk_score,
        }

        rows = df[["invoice_no", "final_risk_score", "risk_level", "explanation"]].to_dict("records")
        return render_template(
            "uploaded_results.html",
            rows=rows,
            summary=summary,
            error=None,
            filename=filename,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
