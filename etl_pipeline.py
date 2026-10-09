"""
ETL pipeline: legacy transactions (source_db) -> clean_transactions (target_db).

Extract : read every row of `transactions` from the legacy source database.
Transform:
    - Deduplicate on transaction_id.
    - Parse amount / transaction_date out of their raw VARCHAR legacy formats.
    - Drop rows whose amount or transaction_date can't be salvaged (null or
      unparseable) - a transaction record is meaningless without both.
    - Filter out negative amounts (data-entry corruption in this legacy feed
      is not tied to the "refunded" status, so a negative amount here is
      treated as invalid rather than a real refund and is dropped).
    - Fill/standardize the remaining soft fields (customer_name, currency,
      status) rather than dropping rows over them.
    - Standardize transaction_date to a single timestamp format.
    - Every dropped row is captured, in its original raw form, tagged with a
      reject_reason - nothing extracted from source is silently discarded.
Load    : write the cleaned dataset to target_db.clean_transactions and the
          dropped rows to target_db.quarantine_transactions (the dead-letter
          queue).
Verify  : print an extract -> transform -> load funnel and confirm both that
          the row counts landed in target_db match the transform output, and
          that clean + quarantine == extracted.

Usage:
    docker compose up -d
    pip install -r requirements.txt
    python etl_pipeline.py
"""

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Config -------------------------------------------------------------

SOURCE = dict(host="localhost", port=5433, dbname="legacy_db", user="postgres", password="postgres")
TARGET = dict(host="localhost", port=5434, dbname="target_warehouse", user="postgres", password="postgres")

SOURCE_TABLE = "transactions"
TARGET_TABLE = "clean_transactions"
QUARANTINE_TABLE = "quarantine_transactions"

RAW_COLUMNS = ["transaction_id", "customer_id", "customer_name", "amount",
               "currency", "transaction_date", "status"]

# Explicit legacy date formats (mirrors generate_data.py's DATE_FORMATS), tried
# in order before falling back to generic parsing for anything left over.
DATE_FORMATS_TRY = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d %H:%M:%S",
]

CURRENCY_MAP = {
    "USD": "USD", "US DOLLAR": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
}

CREATE_TARGET_TABLE_SQL = f"""
DROP TABLE IF EXISTS {TARGET_TABLE};
CREATE TABLE {TARGET_TABLE} (
    transaction_id   VARCHAR(50) PRIMARY KEY,
    customer_id      VARCHAR(50) NOT NULL,
    customer_name    VARCHAR(200) NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL,
    currency         VARCHAR(20) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    status           VARCHAR(20) NOT NULL,
    loaded_at        TIMESTAMP DEFAULT NOW()
);
"""

# No primary key here: a duplicate-id rejection deliberately shares its
# transaction_id with the surviving row in clean_transactions, and the raw
# amount/transaction_date are kept as the original (possibly malformed) text
# rather than the parsed types, since that's exactly what needs auditing.
CREATE_QUARANTINE_TABLE_SQL = f"""
DROP TABLE IF EXISTS {QUARANTINE_TABLE};
CREATE TABLE {QUARANTINE_TABLE} (
    id                SERIAL PRIMARY KEY,
    transaction_id    VARCHAR(50),
    customer_id       VARCHAR(50),
    customer_name     VARCHAR(200),
    amount            VARCHAR(50),
    currency          VARCHAR(20),
    transaction_date  VARCHAR(50),
    status            VARCHAR(20),
    reject_reason     VARCHAR(30) NOT NULL,
    quarantined_at    TIMESTAMP DEFAULT NOW()
);
"""


def make_engine(cfg):
    url = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    return create_engine(url)


# --- Extract --------------------------------------------------------------

def extract(engine):
    print(f"[extract] reading {SOURCE_TABLE} from {SOURCE['dbname']}:{SOURCE['port']} ...")
    df = pd.read_sql_table(SOURCE_TABLE, engine)
    print(f"[extract] got {len(df)} rows")
    return df


# --- Transform --------------------------------------------------------------

def clean_amount(series):
    """Strip currency symbols/thousands separators and coerce to a float; anything
    unparseable ("N/A", "abc", "--", "") becomes NaN."""
    stripped = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[$,]", "", regex=True)
    )
    stripped = stripped.where(series.notna(), np.nan)  # preserve real nulls
    return pd.to_numeric(stripped, errors="coerce")


def clean_dates(series):
    """Try each known legacy date format in turn, then fall back to generic
    parsing for anything still unmatched. Unparseable/garbage values -> NaT."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.copy()
    for fmt in DATE_FORMATS_TRY:
        mask = parsed.isna() & remaining.notna()
        attempt = pd.to_datetime(remaining[mask], format=fmt, errors="coerce")
        parsed.loc[attempt.index] = attempt
    mask = parsed.isna() & remaining.notna()
    if mask.any():
        # Whatever's left is genuinely irregular (or garbage like "yesterday",
        # "00/00/0000") - fall back to dateutil's best-effort guess per value.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            attempt = pd.to_datetime(remaining[mask], errors="coerce")
        parsed.loc[attempt.index] = attempt
    return parsed


def clean_currency(series):
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
    cleaned = cleaned.map(lambda c: CURRENCY_MAP.get(c, c))
    return cleaned


def transform(df):
    stats = {"extracted": len(df)}

    # Pristine snapshot (original raw text, original index) to pull from when
    # quarantining - regardless of which stage rejects a row, its dead-letter
    # copy is always the untouched value as it came out of source_db.
    raw = df[RAW_COLUMNS].copy()
    out = df.copy()
    quarantine_frames = []

    def quarantine(mask, reason):
        """mask is a boolean Series aligned with out's current index."""
        idx = out.index[mask]
        stats[reason] = len(idx)
        if len(idx):
            q = raw.loc[idx].copy()
            q["reject_reason"] = reason
            quarantine_frames.append(q)
        return idx

    # 1. Deduplicate on the business key.
    dup_mask = out.duplicated(subset=["transaction_id"], keep="first")
    out = out.drop(index=quarantine(dup_mask, "DUPLICATE_ID"))

    # 2. Drop rows with no transaction_id / customer_id at all (unusable keys).
    missing_mask = out["transaction_id"].isna() | out["customer_id"].isna()
    out = out.drop(index=quarantine(missing_mask, "MISSING_KEY"))

    # 3. Parse amount; drop rows where it's null or unparseable.
    parsed_amount = clean_amount(out["amount"])
    invalid_amount_mask = parsed_amount.isna()
    out = out.drop(index=quarantine(invalid_amount_mask, "INVALID_AMOUNT"))
    out["amount"] = parsed_amount[~invalid_amount_mask]

    # 4. Filter out negative amounts.
    negative_mask = out["amount"] < 0
    out = out.drop(index=quarantine(negative_mask, "NEGATIVE_AMOUNT"))

    # 5. Parse transaction_date; drop rows where it's null or unparseable.
    parsed_date = clean_dates(out["transaction_date"])
    invalid_date_mask = parsed_date.isna()
    out = out.drop(index=quarantine(invalid_date_mask, "INVALID_DATE"))
    out["transaction_date"] = parsed_date[~invalid_date_mask]

    # 6. Standardize the remaining soft fields instead of dropping rows.
    out["customer_name"] = out["customer_name"].fillna("Unknown").str.strip()
    out.loc[out["customer_name"] == "", "customer_name"] = "Unknown"

    out["currency"] = clean_currency(out["currency"])

    out["status"] = out["status"].astype(str).str.strip().str.lower()
    out["status"] = out["status"].replace({"": "unknown", "nan": "unknown", "none": "unknown"})

    out["amount"] = out["amount"].round(2)

    out = out[["transaction_id", "customer_id", "customer_name", "amount",
               "currency", "transaction_date", "status"]].reset_index(drop=True)

    if quarantine_frames:
        quarantine_df = pd.concat(quarantine_frames, ignore_index=True)
    else:
        quarantine_df = pd.DataFrame(columns=RAW_COLUMNS + ["reject_reason"])

    stats["cleaned"] = len(out)
    stats["quarantined"] = len(quarantine_df)
    return out, quarantine_df, stats


# --- Load --------------------------------------------------------------

def _run_ddl(engine, ddl):
    with engine.begin() as conn:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))


def load(engine, clean_df, quarantine_df):
    print(f"[load] (re)creating {TARGET_TABLE} and {QUARANTINE_TABLE} in "
          f"{TARGET['dbname']}:{TARGET['port']} ...")
    _run_ddl(engine, CREATE_TARGET_TABLE_SQL)
    _run_ddl(engine, CREATE_QUARANTINE_TABLE_SQL)

    print(f"[load] inserting {len(clean_df)} cleaned rows into {TARGET_TABLE} ...")
    clean_df.to_sql(TARGET_TABLE, engine, if_exists="append", index=False, chunksize=1000)

    print(f"[load] inserting {len(quarantine_df)} rejected rows into {QUARANTINE_TABLE} ...")
    quarantine_df.to_sql(QUARANTINE_TABLE, engine, if_exists="append", index=False, chunksize=1000)
    print("[load] done")


# --- Verify --------------------------------------------------------------

REJECT_REASONS = ["DUPLICATE_ID", "MISSING_KEY", "INVALID_AMOUNT", "NEGATIVE_AMOUNT", "INVALID_DATE"]


def verify(target_engine, extracted_count, cleaned_df, quarantine_df, stats):
    with target_engine.begin() as conn:
        clean_loaded = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
        quarantine_loaded = conn.execute(text(f"SELECT COUNT(*) FROM {QUARANTINE_TABLE}")).scalar()

    print("\n" + "=" * 60)
    print("ETL SUMMARY")
    print("=" * 60)
    print(f"{'Extracted from source_db':35s}: {stats['extracted']:>8,}")
    for reason in REJECT_REASONS:
        print(f"{'  - ' + reason.replace('_', ' ').lower():35s}: {stats.get(reason, 0):>8,}")
    print("-" * 60)
    print(f"{'Cleaned rows (transform output)':35s}: {stats['cleaned']:>8,}")
    print(f"{'Quarantined rows (transform output)':35s}: {stats['quarantined']:>8,}")
    funnel_match = stats["cleaned"] + stats["quarantined"] == stats["extracted"]
    print(f"{'  clean + quarantine == extracted':35s}: {'YES' if funnel_match else 'NO'} "
          f"({stats['cleaned']:,} + {stats['quarantined']:,} = "
          f"{stats['cleaned'] + stats['quarantined']:,} vs {stats['extracted']:,})")
    print("-" * 60)
    print(f"{'Rows loaded into clean_transactions':35s}: {clean_loaded:>8,}")
    print(f"{'Rows loaded into quarantine_transactions':35s}: {quarantine_loaded:>8,}")
    clean_match = clean_loaded == stats["cleaned"] == len(cleaned_df)
    quarantine_match = quarantine_loaded == stats["quarantined"] == len(quarantine_df)
    print(f"{'Load verification':35s}: "
          f"{'MATCH' if clean_match and quarantine_match else 'MISMATCH'}")
    print("=" * 60)

    if not cleaned_df.empty:
        print("\nCleaned amount stats:")
        print(cleaned_df["amount"].describe().to_string())
        print(f"\nDate range: {cleaned_df['transaction_date'].min()} -> {cleaned_df['transaction_date'].max()}")
        print("\nCurrency distribution:")
        print(cleaned_df["currency"].value_counts().to_string())
        print("\nStatus distribution:")
        print(cleaned_df["status"].value_counts().to_string())

    if not quarantine_df.empty:
        print("\nQuarantine reject_reason breakdown:")
        print(quarantine_df["reject_reason"].value_counts().to_string())

    if not (clean_match and quarantine_match and funnel_match):
        print("\n[WARN] row-count verification failed!", file=sys.stderr)
        sys.exit(1)


def main():
    start = datetime.now()
    source_engine = make_engine(SOURCE)
    target_engine = make_engine(TARGET)

    raw_df = extract(source_engine)
    cleaned_df, quarantine_df, stats = transform(raw_df)
    load(target_engine, cleaned_df, quarantine_df)
    verify(target_engine, len(raw_df), cleaned_df, quarantine_df, stats)

    print(f"\nCompleted in {(datetime.now() - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
