"""
Generate 10,000 dirty transaction records and load them into the source
(legacy) database.

Simulates a messy legacy transactional system on purpose: null values,
negative amounts, malformed amount/date strings, inconsistent currency and
status casing, and duplicate rows. This gives the migration pipeline real
cleaning work to do when moving data from source_db -> target_db.

Usage:
    docker compose up -d
    pip install -r requirements.txt
    python generate_data.py
"""

import random

import pandas as pd
from faker import Faker
from sqlalchemy import create_engine, text

# --- Config -----------------------------------------------------------------

DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "legacy_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

TOTAL_RECORDS = 10_000

# Dirty-data injection rates (as a fraction of TOTAL_RECORDS)
DUPLICATE_RATE = 0.03
NULL_NAME_RATE = 0.05
NULL_AMOUNT_RATE = 0.03
NEGATIVE_AMOUNT_RATE = 0.05
BAD_AMOUNT_FORMAT_RATE = 0.03
BAD_DATE_FORMAT_RATE = 0.05
NULL_DATE_RATE = 0.02

CURRENCY_VARIANTS = ["USD", "usd", "US Dollar", "EUR", "eur", "GBP", "JPY", ""]
STATUS_VALUES = ["completed", "COMPLETED", "pending", "failed", "Failed", "refunded", None]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d %H:%M:%S"]
BAD_AMOUNT_STRINGS = ["N/A", "unknown", "--", "abc", "$100", "1,234.56", ""]
GARBAGE_DATES = ["not-a-date", "00/00/0000", "2023-13-45", "yesterday"]

fake = Faker()
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# amount/transaction_date are kept as VARCHAR (not NUMERIC/DATE) so the table
# can actually hold the malformed legacy values instead of rejecting them.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50),
    customer_id VARCHAR(50),
    customer_name VARCHAR(200),
    amount VARCHAR(50),
    currency VARCHAR(20),
    transaction_date VARCHAR(50),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
"""


def random_date_string():
    """Transaction date: usually well-formed, sometimes wrong format, sometimes null/garbage."""
    if random.random() < NULL_DATE_RATE:
        return None
    dt = fake.date_time_between(start_date="-2y", end_date="now")
    if random.random() < BAD_DATE_FORMAT_RATE:
        if random.random() < 0.3:
            return random.choice(GARBAGE_DATES)
        return dt.strftime(random.choice(DATE_FORMATS))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def random_amount():
    """Transaction amount as a string: usually a positive number, sometimes negative,
    malformed, or null."""
    if random.random() < NULL_AMOUNT_RATE:
        return None
    if random.random() < BAD_AMOUNT_FORMAT_RATE:
        return random.choice(BAD_AMOUNT_STRINGS)
    amount = round(random.uniform(1, 5000), 2)
    if random.random() < NEGATIVE_AMOUNT_RATE:
        amount = -amount
    return str(amount)


def generate_row(row_id):
    customer_name = None if random.random() < NULL_NAME_RATE else fake.name()
    return {
        "transaction_id": f"TXN-{row_id:07d}",
        "customer_id": f"CUST-{random.randint(1, 2000):05d}",
        "customer_name": customer_name,
        "amount": random_amount(),
        "currency": random.choice(CURRENCY_VARIANTS),
        "transaction_date": random_date_string(),
        "status": random.choice(STATUS_VALUES),
    }


def main():
    n_duplicates = int(TOTAL_RECORDS * DUPLICATE_RATE)
    n_unique = TOTAL_RECORDS - n_duplicates

    print(f"Generating {n_unique} unique records + {n_duplicates} duplicates "
          f"({TOTAL_RECORDS} total)...")

    rows = [generate_row(i) for i in range(1, n_unique + 1)]

    # Inject duplicates by re-appending copies of existing rows verbatim
    # (same transaction_id and all), so total row count == TOTAL_RECORDS.
    duplicates = [dict(random.choice(rows)) for _ in range(n_duplicates)]
    rows.extend(duplicates)

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    print(f"Connecting to {DB_NAME} on port {DB_PORT}...")
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

    print("Loading data into transactions table...")
    df.to_sql("transactions", engine, if_exists="append", index=False, chunksize=1000)

    print(f"Done. Inserted {len(df)} rows (including {n_duplicates} duplicate rows) "
          f"into {DB_NAME}.transactions.")


if __name__ == "__main__":
    main()
