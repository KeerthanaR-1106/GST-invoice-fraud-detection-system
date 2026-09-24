import os

import pandas as pd
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = os.environ.get("GST_PREPROCESS_INPUT", "data/gst_invoices.csv")
OUTPUT_FILE = os.environ.get("GST_PREPROCESS_OUTPUT", "data/ml_features.csv")

# ==========================================
# LOAD RAW DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("Raw dataset loaded.")
print("Shape:", df.shape)


# ==========================================
# BASIC DATA INFORMATION
# ==========================================

print("\nRaw columns:")

for column in df.columns:
    print("-", column)


# ==========================================
# CONVERT DATE
# ==========================================

df["invoice_date"] = pd.to_datetime(
    df["invoice_date"],
    errors="coerce"
)


print(
    "\nInvalid invoice dates:",
    df["invoice_date"].isna().sum()
)


# ==========================================
# MISSING VALUE CHECK
# ==========================================

print("\nMissing values:")

missing_values = df.isnull().sum()

print(missing_values)


# ==========================================
# 1. CALCULATED GROSS VALUE
# ==========================================

# Gross value before discount:
#
# quantity × unit price

df["calculated_gross_value"] = (
    df["quantity"] *
    df["unit_price"]
)


# ==========================================
# 2. DISCOUNT CONSISTENCY
# ==========================================

df["calculated_discount"] = (
    df["calculated_gross_value"] *
    df["discount_percentage"] /
    100
)


df["discount_difference"] = abs(
    df["discount_amount"] -
    df["calculated_discount"]
)


# ==========================================
# 3. TAXABLE VALUE CONSISTENCY
# ==========================================

df["calculated_taxable_value"] = (
    df["calculated_gross_value"] -
    df["discount_amount"]
)


df["taxable_value_difference"] = abs(
    df["taxable_value"] -
    df["calculated_taxable_value"]
)


# ==========================================
# 4. EXPECTED GST
# ==========================================

df["expected_gst"] = (
    df["taxable_value"] *
    df["gst_rate"] /
    100
)


# ==========================================
# 5. CALCULATED GST
# ==========================================

df["calculated_gst"] = (
    df["cgst"] +
    df["sgst"] +
    df["igst"]
)


# ==========================================
# 6. GST DIFFERENCE
# ==========================================

df["gst_difference"] = abs(
    df["expected_gst"] -
    df["calculated_gst"]
)


# ==========================================
# 7. EXPECTED CGST
# ==========================================

df["expected_cgst"] = np.where(

    df["igst"] == 0,

    df["taxable_value"] *
    (df["gst_rate"] / 2) /
    100,

    0
)


# ==========================================
# 8. EXPECTED SGST
# ==========================================

df["expected_sgst"] = np.where(

    df["igst"] == 0,

    df["taxable_value"] *
    (df["gst_rate"] / 2) /
    100,

    0
)


# ==========================================
# 9. CGST / SGST SPLIT DIFFERENCE
# ==========================================

df["cgst_sgst_split_difference"] = (

    abs(
        df["cgst"] -
        df["expected_cgst"]
    )

    +

    abs(
        df["sgst"] -
        df["expected_sgst"]
    )
)


# ==========================================
# 10. IGST DIFFERENCE
# ==========================================

df["igst_difference"] = np.where(

    df["igst"] > 0,

    abs(
        df["igst"] -
        df["expected_gst"]
    ),

    0
)


# ==========================================
# 11. TOTAL TAX CONSISTENCY
# ==========================================

df["calculated_total_tax"] = (

    df["cgst"] +
    df["sgst"] +
    df["igst"] +
    df["cess_amount"]
)


df["total_tax_difference"] = abs(

    df["total_tax"] -
    df["calculated_total_tax"]
)


# ==========================================
# 12. TOTAL AMOUNT CONSISTENCY
# ==========================================

df["calculated_total_amount"] = (

    df["taxable_value"] +
    df["total_tax"]
)


df["total_amount_difference"] = abs(

    df["total_amount"] -
    df["calculated_total_amount"]
)


# Date values are retained as invoice attributes, but no fixed calendar
# range is imposed on the dataset.
df["future_date_flag"] = 0
df["days_beyond_allowed_end"] = 0


# ==========================================
# 15. SUPPLIER FREQUENCY
# ==========================================

supplier_frequency = (
    df["supplier_id"]
    .value_counts()
)


df["supplier_frequency"] = (
    df["supplier_id"]
    .map(
        supplier_frequency
    )
)


# ==========================================
# 16. BUYER FREQUENCY
# ==========================================

buyer_frequency = (
    df["buyer_id"]
    .value_counts()
)


df["buyer_frequency"] = (
    df["buyer_id"]
    .map(
        buyer_frequency
    )
)


# ==========================================
# 17. SUPPLIER-BUYER FREQUENCY
# ==========================================

supplier_buyer_frequency = (

    df.groupby(
        [
            "supplier_id",
            "buyer_id"
        ]
    )["invoice_no"]
    .transform("count")
)


df["supplier_buyer_frequency"] = (
    supplier_buyer_frequency
)


# ==========================================
# 18. SUPPLIER DAILY FREQUENCY
# ==========================================

supplier_daily_frequency = (

    df.groupby(
        [
            "supplier_id",
            "invoice_date"
        ]
    )["invoice_no"]
    .transform("count")
)


df["supplier_daily_frequency"] = (
    supplier_daily_frequency
)


# ==========================================
# 19. BUYER DAILY FREQUENCY
# ==========================================

buyer_daily_frequency = (

    df.groupby(
        [
            "buyer_id",
            "invoice_date"
        ]
    )["invoice_no"]
    .transform("count")
)


df["buyer_daily_frequency"] = (
    buyer_daily_frequency
)


# ==========================================
# 20. DAILY TRANSACTION FREQUENCY
# ==========================================

daily_frequency = (

    df.groupby(
        "invoice_date"
    )["invoice_no"]
    .transform("count")
)


df["daily_transaction_frequency"] = (
    daily_frequency
)


# ==========================================
# 21. DUPLICATE TRANSACTION FINGERPRINT
# ==========================================

# Invoice number is deliberately NOT included.
#
# Two invoices with the same transaction
# details can therefore be identified even
# when their invoice numbers are different.

transaction_columns = [

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

    "payment_method"
]


transaction_fingerprint = (

    df[
        transaction_columns
    ]
    .astype(str)
    .agg(
        "|".join,
        axis=1
    )
)


transaction_frequency = (
    transaction_fingerprint
    .value_counts()
)


df["duplicate_transaction_frequency"] = (

    transaction_fingerprint.map(
        transaction_frequency
    )
)


# ==========================================
# 22. DUPLICATE TRANSACTION FLAG
# ==========================================

df["duplicate_transaction_flag"] = (

    df[
        "duplicate_transaction_frequency"
    ] > 1

).astype(int)


# ==========================================
# 23. DATE FEATURES
# ==========================================

df["invoice_month"] = (
    df["invoice_date"].dt.month
)


# ==========================================
# 24. REMOVE INVALID NUMERIC VALUES
# ==========================================

numeric_columns = [

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

    "calculated_gross_value",
    "discount_difference",
    "taxable_value_difference",

    "expected_gst",
    "calculated_gst",
    "gst_difference",

    "expected_cgst",
    "expected_sgst",
    "cgst_sgst_split_difference",
    "igst_difference",

    "calculated_total_tax",
    "total_tax_difference",

    "calculated_total_amount",
    "total_amount_difference",

    "future_date_flag",
    "days_beyond_allowed_end",

    "supplier_frequency",
    "buyer_frequency",
    "supplier_buyer_frequency",

    "supplier_daily_frequency",
    "buyer_daily_frequency",
    "daily_transaction_frequency",

    "duplicate_transaction_frequency",
    "duplicate_transaction_flag",

    "invoice_month"
]


# Replace infinity with NaN

df[numeric_columns] = (

    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# Replace missing numeric values with zero

df[numeric_columns] = (

    df[numeric_columns]
    .fillna(0)
)


# ==========================================
# 25. SELECT FEATURES FOR MACHINE LEARNING
# ==========================================

ml_features = [

    # -------------------------------
    # Transaction / financial values
    # -------------------------------

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


    # -------------------------------
    # Mathematical consistency
    # -------------------------------

    "calculated_gross_value",

    "discount_difference",

    "taxable_value_difference",

    "gst_difference",

    "total_tax_difference",

    "total_amount_difference",

    "cgst_sgst_split_difference",

    "igst_difference",


    # -------------------------------
    # Duplicate behaviour
    # -------------------------------

    "duplicate_transaction_frequency",

    "duplicate_transaction_flag",


    # -------------------------------
    # Supplier / buyer behaviour
    # -------------------------------

    "supplier_frequency",

    "buyer_frequency",

    "supplier_buyer_frequency",

    "supplier_daily_frequency",

    "buyer_daily_frequency",

    "daily_transaction_frequency",


    # -------------------------------
    # Date anomalies
    # -------------------------------

    "future_date_flag",

    "days_beyond_allowed_end",

    "invoice_month"
]


# ==========================================
# CREATE ML DATASET
# ==========================================

ml_df = df[
    ml_features
].copy()


# ==========================================
# FINAL SAFETY CHECK
# ==========================================

ml_df = ml_df.replace(
    [np.inf, -np.inf],
    np.nan
)


ml_df = ml_df.fillna(0)


# Make sure all ML columns are numeric
# (invoice_no will be inserted later and is NOT converted)

for column in ml_df.columns:

    ml_df[column] = pd.to_numeric(
        ml_df[column],
        errors="coerce"
    )


ml_df = ml_df.fillna(0)


# ==========================================
# PRESERVE INVOICE IDENTIFIERS (TRACEABILITY)
# ==========================================

# Insert invoice_no as the first column for traceability, but do NOT
# include it as an ML feature (it was not part of ml_df columns above).
out_df = ml_df.copy()

if "invoice_no" in df.columns:
    out_df.insert(0, "invoice_no", df["invoice_no"].astype(str))
else:
    # Fallback to index-based IDs if original invoice_no is missing
    out_df.insert(0, "invoice_no", df.index.astype(str))


# ==========================================
# SAVE ML DATASET
# ==========================================

out_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(
    "\n=========================================="
)

print(
    "PREPROCESSING COMPLETED"
)

print(
    "=========================================="
)


print(
    "\nRaw dataset shape:",
    df.shape
)


print(
    "ML dataset shape:",
    ml_df.shape
)


print(
    "\nNumber of ML features:",
    len(ml_features)
)


print(
    "\nML features:"
)


for feature in ml_features:

    print(
        "-",
        feature
    )


# ==========================================
# FEATURE DIAGNOSTICS
# ==========================================

print(
    "\n=========================================="
)

print(
    "FEATURE DIAGNOSTICS"
)

print(
    "=========================================="
)


print(
    "\nFuture-date records:",
    df[
        "future_date_flag"
    ].sum()
)


print(
    "Duplicate transaction records:",
    df[
        "duplicate_transaction_flag"
    ].sum()
)


print(
    "Supplier daily maximum frequency:",
    df[
        "supplier_daily_frequency"
    ].max()
)


print(
    "Buyer daily maximum frequency:",
    df[
        "buyer_daily_frequency"
    ].max()
)


print(
    "\nRecords with CGST/SGST split difference:",
    (
        df[
            "cgst_sgst_split_difference"
        ] > 0.01
    ).sum()
)


print(
    "\nRecords with GST difference:",
    (
        df[
            "gst_difference"
        ] > 0.01
    ).sum()
)


print(
    "\nRecords with total tax difference:",
    (
        df[
            "total_tax_difference"
        ] > 0.01
    ).sum()
)


print(
    "\nSaved ML dataset as:"
)

print(
    OUTPUT_FILE
)