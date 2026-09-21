import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# ============================================================
# SETTINGS
# ============================================================

NUM_INVOICES = 20000
NUM_SUPPLIERS = 500
NUM_BUYERS = 1000

ANOMALY_PERCENTAGE = 0.05

random.seed(42)
np.random.seed(42)


# ============================================================
# DATE RANGE
# ============================================================

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)


# ============================================================
# BASIC DATA
# ============================================================

product_categories = [
    "Electronics",
    "Clothing",
    "Food",
    "Furniture",
    "Industrial Equipment",
    "Pharmaceuticals",
    "Stationery"
]


hsn_codes = {
    "Electronics": "8504",
    "Clothing": "6109",
    "Food": "2106",
    "Furniture": "9403",
    "Industrial Equipment": "8419",
    "Pharmaceuticals": "3004",
    "Stationery": "4820"
}


gst_rates = [
    5,
    12,
    18,
    28
]


payment_methods = [
    "Cash",
    "UPI",
    "Bank Transfer",
    "Credit Card",
    "Cheque"
]


# ============================================================
# ANOMALY TYPES
# ============================================================

anomaly_types = [

    "Extreme Invoice Amount",
    "Extreme Unit Price",
    "Unusual Quantity",
    "Unusual Discount",
    "Tax Calculation Mismatch",
    "Incorrect CGST SGST Split",
    "Incorrect IGST",
    "Duplicate Transaction",
    "Unusual Supplier Activity",
    "Unusual Buyer Activity",
    "Future Invoice Date"
]


# ============================================================
# GSTIN GENERATOR
# ============================================================

def generate_gstin():

    state_code = random.randint(
        1,
        37
    )

    pan_part = ''.join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            k=5
        )
    )

    digits = ''.join(
        random.choices(
            "0123456789",
            k=4
        )
    )

    letter = random.choice(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    entity_number = random.choice(
        "123456789"
    )

    checksum = random.choice(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    return (
        f"{state_code:02d}"
        f"{pan_part}"
        f"{digits}"
        f"{letter}"
        f"{entity_number}"
        f"Z"
        f"{checksum}"
    )


# ============================================================
# CREATE SUPPLIERS
# ============================================================

suppliers = {}

for i in range(
    1,
    NUM_SUPPLIERS + 1
):

    suppliers[i] = {

        "supplier_id":
            f"S{i:04d}",

        "supplier_gstin":
            generate_gstin()
    }


# ============================================================
# CREATE BUYERS
# ============================================================

buyers = {}

for i in range(
    1,
    NUM_BUYERS + 1
):

    buyers[i] = {

        "buyer_id":
            f"B{i:04d}",

        "buyer_gstin":
            generate_gstin()
    }


# ============================================================
# HELPER: RANDOM DATE
# ============================================================

def random_date():

    days = (
        END_DATE -
        START_DATE
    ).days

    return START_DATE + timedelta(
        days=random.randint(
            0,
            days
        )
    )


# ============================================================
# HELPER: UNIT PRICE
# ============================================================

def generate_unit_price(category):

    price_ranges = {

        "Electronics":
            (500, 50000),

        "Clothing":
            (200, 5000),

        "Food":
            (50, 3000),

        "Furniture":
            (1000, 50000),

        "Industrial Equipment":
            (5000, 100000),

        "Pharmaceuticals":
            (100, 10000),

        "Stationery":
            (20, 2000)
    }

    low, high = price_ranges[
        category
    ]

    return round(
        random.uniform(
            low,
            high
        ),
        2
    )


# ============================================================
# HELPER: CALCULATE GST VALUES
# ============================================================

def calculate_tax_values(
    taxable_value,
    gst_rate,
    use_igst
):

    if use_igst:

        cgst = 0.0

        sgst = 0.0

        igst = round(
            taxable_value *
            gst_rate /
            100,
            2
        )

    else:

        cgst = round(
            taxable_value *
            (gst_rate / 2) /
            100,
            2
        )

        sgst = round(
            taxable_value *
            (gst_rate / 2) /
            100,
            2
        )

        igst = 0.0

    return (
        cgst,
        sgst,
        igst
    )


# ============================================================
# HELPER: RECALCULATE FINANCIAL VALUES
# ============================================================

def recalculate_invoice(
    row,
    quantity=None,
    unit_price=None,
    discount_percentage=None
):

    if quantity is None:
        quantity = row["quantity"]

    if unit_price is None:
        unit_price = row["unit_price"]

    if discount_percentage is None:
        discount_percentage = (
            row["discount_percentage"]
        )

    gross_value = (
        quantity *
        unit_price
    )

    discount_amount = round(
        gross_value *
        discount_percentage /
        100,
        2
    )

    taxable_value = round(
        gross_value -
        discount_amount,
        2
    )

    gst_rate = row["gst_rate"]

    use_igst = (
        row["igst"] > 0
    )

    cgst, sgst, igst = (
        calculate_tax_values(
            taxable_value,
            gst_rate,
            use_igst
        )
    )

    cess_rate = row["cess_rate"]

    cess_amount = round(
        taxable_value *
        cess_rate /
        100,
        2
    )

    total_tax = round(
        cgst +
        sgst +
        igst +
        cess_amount,
        2
    )

    total_amount = round(
        taxable_value +
        total_tax,
        2
    )

    return {

        "quantity":
            quantity,

        "unit_price":
            round(
                unit_price,
                2
            ),

        "discount_percentage":
            discount_percentage,

        "discount_amount":
            discount_amount,

        "taxable_value":
            taxable_value,

        "cgst":
            cgst,

        "sgst":
            sgst,

        "igst":
            igst,

        "cess_amount":
            cess_amount,

        "total_tax":
            total_tax,

        "total_amount":
            total_amount
    }


# ============================================================
# CREATE NORMAL INVOICES
# ============================================================

records = []


for i in range(
    1,
    NUM_INVOICES + 1
):

    invoice_no = (
        f"INV{i:06d}"
    )

    invoice_date = random_date()


    # --------------------------------------------------------
    # SUPPLIER
    # --------------------------------------------------------

    supplier = suppliers[
        random.randint(
            1,
            NUM_SUPPLIERS
        )
    ]


    # --------------------------------------------------------
    # BUYER
    # --------------------------------------------------------

    buyer = buyers[
        random.randint(
            1,
            NUM_BUYERS
        )
    ]


    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    category = random.choice(
        product_categories
    )

    hsn_code = hsn_codes[
        category
    ]


    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    quantity = random.randint(
        1,
        100
    )


    # --------------------------------------------------------
    # UNIT PRICE
    # --------------------------------------------------------

    unit_price = generate_unit_price(
        category
    )


    # --------------------------------------------------------
    # DISCOUNT
    # --------------------------------------------------------

    discount_percentage = random.choice(
        [
            0,
            0,
            0,
            5,
            10,
            15
        ]
    )


    gross_value = (
        quantity *
        unit_price
    )

    discount_amount = round(
        gross_value *
        discount_percentage /
        100,
        2
    )

    taxable_value = round(
        gross_value -
        discount_amount,
        2
    )


    # --------------------------------------------------------
    # GST
    # --------------------------------------------------------

    gst_rate = random.choice(
        gst_rates
    )


    # 70% CGST + SGST
    # 30% IGST

    use_igst = (
        random.random() >= 0.70
    )


    cgst, sgst, igst = (
        calculate_tax_values(
            taxable_value,
            gst_rate,
            use_igst
        )
    )


    # --------------------------------------------------------
    # CESS
    # --------------------------------------------------------

    if random.random() < 0.90:

        cess_rate = 0.0

    else:

        cess_rate = random.choice(
            [
                1,
                2,
                5
            ]
        )


    cess_amount = round(
        taxable_value *
        cess_rate /
        100,
        2
    )


    # --------------------------------------------------------
    # TOTAL TAX
    # --------------------------------------------------------

    total_tax = round(
        cgst +
        sgst +
        igst +
        cess_amount,
        2
    )


    # --------------------------------------------------------
    # TOTAL AMOUNT
    # --------------------------------------------------------

    total_amount = round(
        taxable_value +
        total_tax,
        2
    )


    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    payment_method = random.choice(
        payment_methods
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    records.append({

        "invoice_no":
            invoice_no,

        "invoice_date":
            invoice_date.strftime(
                "%Y-%m-%d"
            ),

        "supplier_id":
            supplier[
                "supplier_id"
            ],

        "supplier_gstin":
            supplier[
                "supplier_gstin"
            ],

        "buyer_id":
            buyer[
                "buyer_id"
            ],

        "buyer_gstin":
            buyer[
                "buyer_gstin"
            ],

        "product_category":
            category,

        "hsn_code":
            hsn_code,

        "quantity":
            quantity,

        "unit_price":
            unit_price,

        "discount_percentage":
            discount_percentage,

        "discount_amount":
            discount_amount,

        "taxable_value":
            taxable_value,

        "gst_rate":
            gst_rate,

        "cgst":
            cgst,

        "sgst":
            sgst,

        "igst":
            igst,

        "cess_rate":
            cess_rate,

        "cess_amount":
            cess_amount,

        "total_tax":
            total_tax,

        "total_amount":
            total_amount,

        "payment_method":
            payment_method
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# SELECT ANOMALY INDICES
# ============================================================

num_anomalies = int(
    NUM_INVOICES *
    ANOMALY_PERCENTAGE
)


anomaly_indices = random.sample(
    range(NUM_INVOICES),
    num_anomalies
)


random.shuffle(
    anomaly_indices
)


# ============================================================
# BALANCED ANOMALY ASSIGNMENT
# ============================================================

base_count = (
    num_anomalies //
    len(anomaly_types)
)

remainder = (
    num_anomalies %
    len(anomaly_types)
)


anomaly_assignments = []

position = 0


for i, anomaly_type in enumerate(
    anomaly_types
):

    count = base_count

    if i < remainder:
        count += 1


    selected = anomaly_indices[
        position:
        position + count
    ]


    for index in selected:

        anomaly_assignments.append(
            (
                index,
                anomaly_type
            )
        )


    position += count


# ============================================================
# GROUND TRUTH
# ============================================================

ground_truth = []


def add_ground_truth(
    index,
    anomaly_type
):

    ground_truth.append({

        "invoice_no":
            df.loc[
                index,
                "invoice_no"
            ],

        "actual_anomaly":
            1,

        "anomaly_type":
            anomaly_type
    })


# ============================================================
# GROUP ANOMALY INDICES
# ============================================================

supplier_activity_indices = []

buyer_activity_indices = []

duplicate_indices = []


for index, anomaly_type in anomaly_assignments:

    if anomaly_type == (
        "Unusual Supplier Activity"
    ):

        supplier_activity_indices.append(
            index
        )

    elif anomaly_type == (
        "Unusual Buyer Activity"
    ):

        buyer_activity_indices.append(
            index
        )

    elif anomaly_type == (
        "Duplicate Transaction"
    ):

        duplicate_indices.append(
            index
        )


# ============================================================
# 1. EXTREME INVOICE AMOUNT
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Extreme Invoice Amount"
    ):
        continue


    original_amount = df.loc[
        index,
        "total_amount"
    ]


    # Deliberate amount manipulation.
    #
    # Other invoice components remain unchanged.

    df.loc[
        index,
        "total_amount"
    ] = round(
        original_amount *
        random.uniform(
            10,
            30
        ),
        2
    )


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 2. EXTREME UNIT PRICE
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Extreme Unit Price"
    ):
        continue


    original_price = df.loc[
        index,
        "unit_price"
    ]


    new_price = (
        original_price *
        random.uniform(
            10,
            20
        )
    )


    values = recalculate_invoice(

        df.loc[index],

        quantity=df.loc[
            index,
            "quantity"
        ],

        unit_price=new_price,

        discount_percentage=df.loc[
            index,
            "discount_percentage"
        ]
    )


    for key, value in values.items():

        df.loc[
            index,
            key
        ] = value


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 3. UNUSUAL QUANTITY
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Unusual Quantity"
    ):
        continue


    new_quantity = random.randint(
        5000,
        20000
    )


    values = recalculate_invoice(

        df.loc[index],

        quantity=new_quantity,

        unit_price=df.loc[
            index,
            "unit_price"
        ],

        discount_percentage=df.loc[
            index,
            "discount_percentage"
        ]
    )


    for key, value in values.items():

        df.loc[
            index,
            key
        ] = value


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 4. UNUSUAL DISCOUNT
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Unusual Discount"
    ):
        continue


    new_discount = random.choice(
        [
            70,
            80,
            90,
            95
        ]
    )


    values = recalculate_invoice(

        df.loc[index],

        quantity=df.loc[
            index,
            "quantity"
        ],

        unit_price=df.loc[
            index,
            "unit_price"
        ],

        discount_percentage=new_discount
    )


    for key, value in values.items():

        df.loc[
            index,
            key
        ] = value


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 5. TAX CALCULATION MISMATCH
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Tax Calculation Mismatch"
    ):
        continue


    original_tax = df.loc[
        index,
        "total_tax"
    ]


    # Modify total_tax only.
    #
    # This creates a direct tax consistency
    # anomaly.

    df.loc[
        index,
        "total_tax"
    ] = round(
        original_tax *
        random.uniform(
            1.5,
            3
        ),
        2
    )


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 6. INCORRECT CGST / SGST SPLIT
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Incorrect CGST SGST Split"
    ):
        continue


    taxable_value = df.loc[
        index,
        "taxable_value"
    ]

    gst_rate = df.loc[
        index,
        "gst_rate"
    ]


    expected_gst = (
        taxable_value *
        gst_rate /
        100
    )


    # Deliberately create an unequal
    # CGST / SGST distribution.

    cgst = round(
        expected_gst * 0.80,
        2
    )

    sgst = round(
        expected_gst - cgst,
        2
    )


    df.loc[
        index,
        "cgst"
    ] = cgst

    df.loc[
        index,
        "sgst"
    ] = sgst

    df.loc[
        index,
        "igst"
    ] = 0.0


    # Total GST remains approximately
    # correct. Only the split is abnormal.

    df.loc[
        index,
        "total_tax"
    ] = round(
        cgst +
        sgst +
        df.loc[
            index,
            "cess_amount"
        ],
        2
    )


    df.loc[
        index,
        "total_amount"
    ] = round(
        df.loc[
            index,
            "taxable_value"
        ]
        +
        df.loc[
            index,
            "total_tax"
        ],
        2
    )


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 7. INCORRECT IGST
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Incorrect IGST"
    ):
        continue


    taxable_value = df.loc[
        index,
        "taxable_value"
    ]

    gst_rate = df.loc[
        index,
        "gst_rate"
    ]


    expected_igst = (
        taxable_value *
        gst_rate /
        100
    )


    # Force this invoice to an IGST
    # transaction and then make IGST wrong.

    wrong_igst = round(

        expected_igst *
        random.choice(
            [
                0.40,
                1.50,
                2.00
            ]
        ),

        2
    )


    df.loc[
        index,
        "cgst"
    ] = 0.0

    df.loc[
        index,
        "sgst"
    ] = 0.0

    df.loc[
        index,
        "igst"
    ] = wrong_igst


    df.loc[
        index,
        "total_tax"
    ] = round(

        wrong_igst +
        df.loc[
            index,
            "cess_amount"
        ],

        2
    )


    df.loc[
        index,
        "total_amount"
    ] = round(

        df.loc[
            index,
            "taxable_value"
        ]
        +
        df.loc[
            index,
            "total_tax"
        ],

        2
    )


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# 8. DUPLICATE TRANSACTION
# ============================================================

# Only normal invoices are used as source records.

normal_indices = [
    i
    for i in range(NUM_INVOICES)
    if i not in anomaly_indices
]


random.shuffle(
    normal_indices
)


for position, index in enumerate(
    duplicate_indices
):

    # Use a different normal transaction
    # for each duplicate where possible.

    source_index = normal_indices[
        position
        %
        len(normal_indices)
    ]


    source = df.loc[
        source_index
    ].copy()


    for column in df.columns:

        if column != "invoice_no":

            df.loc[
                index,
                column
            ] = source[column]


    # IMPORTANT:
    # The invoice number stays unique.

    df.loc[
        index,
        "invoice_no"
    ] = f"INV{index + 1:06d}"


    add_ground_truth(
        index,
        "Duplicate Transaction"
    )


# ============================================================
# 9. UNUSUAL SUPPLIER ACTIVITY
# ============================================================

if supplier_activity_indices:

    selected_supplier = random.choice(
        list(
            suppliers.values()
        )
    )


    burst_date = random_date()


    for index in supplier_activity_indices:

        df.loc[
            index,
            "supplier_id"
        ] = selected_supplier[
            "supplier_id"
        ]

        df.loc[
            index,
            "supplier_gstin"
        ] = selected_supplier[
            "supplier_gstin"
        ]

        df.loc[
            index,
            "invoice_date"
        ] = burst_date.strftime(
            "%Y-%m-%d"
        )


        add_ground_truth(
            index,
            "Unusual Supplier Activity"
        )


# ============================================================
# 10. UNUSUAL BUYER ACTIVITY
# ============================================================

if buyer_activity_indices:

    selected_buyer = random.choice(
        list(
            buyers.values()
        )
    )


    burst_date = random_date()


    for index in buyer_activity_indices:

        df.loc[
            index,
            "buyer_id"
        ] = selected_buyer[
            "buyer_id"
        ]

        df.loc[
            index,
            "buyer_gstin"
        ] = selected_buyer[
            "buyer_gstin"
        ]

        df.loc[
            index,
            "invoice_date"
        ] = burst_date.strftime(
            "%Y-%m-%d"
        )


        add_ground_truth(
            index,
            "Unusual Buyer Activity"
        )


# ============================================================
# 11. FUTURE INVOICE DATE
# ============================================================

for index, anomaly_type in anomaly_assignments:

    if anomaly_type != (
        "Future Invoice Date"
    ):
        continue


    # Normal data ends on 31-Dec-2025.
    # This date is deliberately outside
    # the valid period.

    future_date = datetime(
        2026,
        12,
        31
    )


    df.loc[
        index,
        "invoice_date"
    ] = future_date.strftime(
        "%Y-%m-%d"
    )


    add_ground_truth(
        index,
        anomaly_type
    )


# ============================================================
# ADD NORMAL RECORDS TO GROUND TRUTH
# ============================================================

anomaly_invoice_numbers = set(

    item["invoice_no"]
    for item in ground_truth
)


for invoice_no in df["invoice_no"]:

    if invoice_no not in (
        anomaly_invoice_numbers
    ):

        ground_truth.append({

            "invoice_no":
                invoice_no,

            "actual_anomaly":
                0,

            "anomaly_type":
                "Normal"
        })


# ============================================================
# CREATE GROUND TRUTH DATAFRAME
# ============================================================

ground_truth_df = pd.DataFrame(
    ground_truth
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n=========================================="
)

print(
    "DATASET VALIDATION"
)

print(
    "=========================================="
)


print(
    "\nNumber of invoices:",
    len(df)
)


print(
    "Number of raw columns:",
    len(df.columns)
)


print(
    "Ground truth rows:",
    len(ground_truth_df)
)


print(
    "Anomaly records:",
    (
        ground_truth_df[
            "actual_anomaly"
        ] == 1
    ).sum()
)


print(
    "Normal records:",
    (
        ground_truth_df[
            "actual_anomaly"
        ] == 0
    ).sum()
)


print(
    "\nDuplicate invoice numbers:",
    df[
        "invoice_no"
    ].duplicated().sum()
)


print(
    "\nAnomaly distribution:"
)


anomaly_distribution = (

    ground_truth_df[
        ground_truth_df[
            "actual_anomaly"
        ] == 1
    ]["anomaly_type"]
    .value_counts()
    .sort_index()
)


print(
    anomaly_distribution
)


# ============================================================
# CHECK RAW COLUMN COUNT
# ============================================================

expected_columns = 22


if len(df.columns) != expected_columns:

    print(
        "\nWARNING:"
    )

    print(
        "Expected",
        expected_columns,
        "raw columns but found",
        len(df.columns)
    )

else:

    print(
        "\nRaw column count check: PASSED"
    )


# ============================================================
# SAVE RAW DATASET
# ============================================================

df.to_csv(
    "data/gst_invoices.csv",
    index=False
)


# ============================================================
# SAVE GROUND TRUTH
# ============================================================

ground_truth_df.to_csv(
    "data/ground_truth.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)

print(
    "GST DATASET GENERATED SUCCESSFULLY"
)

print(
    "=========================================="
)


print(
    "\nFiles created:"
)

print(
    "1. data/gst_invoices.csv"
)

print(
    "2. data/ground_truth.csv"
)