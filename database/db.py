"""SQLite database initialization and helpers.

Phase 11 will add functions to initialize the invoice table and persist results.
"""

import sqlite3
from sqlite3 import Connection
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "gst_invoices.db"


def get_connection() -> Connection:
    conn = sqlite3.connect(str(DB_PATH))
    return conn
