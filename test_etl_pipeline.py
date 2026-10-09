"""
Unit tests for the pure transformation logic in etl_pipeline.py.

Everything here operates on in-memory pandas DataFrames built by hand - no
Postgres connection, no docker containers, no network I/O. Only extract(),
load() and main() touch a database; transform() and its helpers
(clean_amount, clean_dates, clean_currency) are pure functions of a
DataFrame, which is exactly what's under test.

Run with:
    pytest test_etl_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from etl_pipeline import (
    RAW_COLUMNS,
    clean_amount,
    clean_currency,
    clean_dates,
    transform,
)


# --- helpers --------------------------------------------------------------

def make_row(transaction_id="TXN-0000001", **overrides):
    """A single valid raw transaction row, with any field overridden."""
    row = {
        "transaction_id": transaction_id,
        "customer_id": "CUST-00001",
        "customer_name": "Jane Doe",
        "amount": "100.00",
        "currency": "USD",
        "transaction_date": "2024-01-15 10:30:00",
        "status": "completed",
    }
    row.update(overrides)
    return row


def make_df(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def reasons_for(quarantine_df, transaction_id):
    return quarantine_df.loc[
        quarantine_df["transaction_id"] == transaction_id, "reject_reason"
    ].tolist()


# ===========================================================================
# 1. Deduplication
# ===========================================================================

class TestDeduplication:
    def test_exact_duplicate_ids_are_deduplicated(self):
        df = make_df([
            make_row("TXN-0000001", customer_name="First"),
            make_row("TXN-0000001", customer_name="First"),  # byte-for-byte identical
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 1
        assert stats["DUPLICATE_ID"] == 1
        assert reasons_for(quarantine, "TXN-0000001") == ["DUPLICATE_ID"]

    def test_duplicate_id_with_different_payload_still_flagged(self):
        # Same business key, different customer_name - still a duplicate on
        # transaction_id, which is what the pipeline dedupes on.
        df = make_df([
            make_row("TXN-0000002", customer_name="Alice"),
            make_row("TXN-0000002", customer_name="Alice Impersonator"),
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 1
        assert stats["DUPLICATE_ID"] == 1

    def test_first_occurrence_is_kept(self):
        df = make_df([
            make_row("TXN-0000003", customer_name="Original"),
            make_row("TXN-0000003", customer_name="Duplicate Copy"),
            make_row("TXN-0000003", customer_name="Another Duplicate"),
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 1
        assert clean.iloc[0]["customer_name"] == "Original"
        assert stats["DUPLICATE_ID"] == 2
        assert len(quarantine) == 2

    def test_near_identical_ids_are_treated_as_distinct(self):
        # Dedup matches transaction_id exactly; ids that merely look alike
        # (different case, trailing whitespace) are NOT collapsed together -
        # they're kept as three separate, independently-valid records.
        df = make_df([
            make_row("TXN-0000004"),
            make_row("txn-0000004"),
            make_row("TXN-0000004 "),
        ])
        clean, quarantine, stats = transform(df)

        assert stats["DUPLICATE_ID"] == 0
        assert len(clean) == 3
        assert set(clean["transaction_id"]) == {"TXN-0000004", "txn-0000004", "TXN-0000004 "}

    def test_three_way_duplicate_keeps_only_one(self):
        df = make_df([make_row("TXN-0000005") for _ in range(3)])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 1
        assert stats["DUPLICATE_ID"] == 2


# ===========================================================================
# 2. Amount parsing
# ===========================================================================

class TestAmountParsing:
    def test_strips_dollar_sign(self):
        result = clean_amount(pd.Series(["$100", "$1500.50"]))
        assert result.tolist() == [100.0, 1500.5]

    def test_strips_thousands_commas(self):
        result = clean_amount(pd.Series(["1,234.56", "12,000"]))
        assert result.tolist() == [1234.56, 12000.0]

    def test_strips_dollar_and_commas_combined(self):
        result = clean_amount(pd.Series(["$1,234.56"]))
        assert result.tolist() == [1234.56]

    def test_null_becomes_nan(self):
        result = clean_amount(pd.Series([None, np.nan]))
        assert result.isna().all()

    def test_invalid_text_becomes_nan(self):
        result = clean_amount(pd.Series(["N/A", "unknown", "abc", "--", ""]))
        assert result.isna().all()

    def test_valid_numeric_string_passes_through(self):
        result = clean_amount(pd.Series(["42.5", "0"]))
        assert result.tolist() == [42.5, 0.0]

    def test_negative_amount_parses_but_stays_negative(self):
        # clean_amount() only parses; filtering negatives is transform()'s job.
        result = clean_amount(pd.Series(["-50.25"]))
        assert result.tolist() == [-50.25]

    def test_transform_quarantines_null_amount(self):
        df = make_df([make_row("TXN-1000001", amount=None)])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert stats["INVALID_AMOUNT"] == 1
        assert reasons_for(quarantine, "TXN-1000001") == ["INVALID_AMOUNT"]

    def test_transform_quarantines_unparseable_amount(self):
        df = make_df([
            make_row("TXN-1000002", amount="N/A"),
            make_row("TXN-1000003", amount="abc"),
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert stats["INVALID_AMOUNT"] == 2

    def test_transform_quarantines_negative_amount(self):
        df = make_df([make_row("TXN-1000004", amount="-171.61")])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert stats["NEGATIVE_AMOUNT"] == 1
        assert reasons_for(quarantine, "TXN-1000004") == ["NEGATIVE_AMOUNT"]
        # the raw (unparsed) value is preserved for audit, not the float
        assert quarantine.loc[quarantine["transaction_id"] == "TXN-1000004", "amount"].iloc[0] == "-171.61"

    def test_transform_keeps_positive_and_zero_amount(self):
        df = make_df([
            make_row("TXN-1000005", amount="99.99"),
            make_row("TXN-1000006", amount="0"),
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 2
        assert stats.get("NEGATIVE_AMOUNT", 0) == 0
        assert stats.get("INVALID_AMOUNT", 0) == 0

    def test_transform_rounds_amount_to_two_decimals(self):
        df = make_df([make_row("TXN-1000007", amount="19.999")])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["amount"] == 20.0


# ===========================================================================
# 3. Date parsing
# ===========================================================================

class TestDateParsing:
    def test_primary_format_with_time(self):
        result = clean_dates(pd.Series(["2024-01-15 10:30:00"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15 10:30:00")

    def test_date_only_format(self):
        result = clean_dates(pd.Series(["2024-01-15"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15")

    def test_day_month_year_slash_format(self):
        result = clean_dates(pd.Series(["15/01/2024"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15")

    def test_month_day_year_dash_format(self):
        result = clean_dates(pd.Series(["01-15-2024"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15")

    def test_year_month_day_slash_with_time_format(self):
        result = clean_dates(pd.Series(["2024/01/15 10:30:00"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15 10:30:00")

    def test_relative_word_yesterday_is_unparseable(self):
        result = clean_dates(pd.Series(["yesterday"]))
        assert pd.isna(result.iloc[0])

    def test_all_zero_date_is_unparseable(self):
        result = clean_dates(pd.Series(["00/00/0000"]))
        assert pd.isna(result.iloc[0])

    def test_out_of_range_month_and_day_is_unparseable(self):
        result = clean_dates(pd.Series(["2023-13-45"]))
        assert pd.isna(result.iloc[0])

    def test_garbage_text_is_unparseable(self):
        result = clean_dates(pd.Series(["not-a-date"]))
        assert pd.isna(result.iloc[0])

    def test_null_is_unparseable(self):
        result = clean_dates(pd.Series([None]))
        assert pd.isna(result.iloc[0])

    def test_mixed_batch_parses_each_independently(self):
        result = clean_dates(pd.Series([
            "2024-01-15 10:30:00", "yesterday", "15/01/2024", "00/00/0000",
        ]))
        assert result.iloc[0] == pd.Timestamp("2024-01-15 10:30:00")
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pd.Timestamp("2024-01-15")
        assert pd.isna(result.iloc[3])

    def test_transform_quarantines_garbage_date(self):
        df = make_df([make_row("TXN-2000001", transaction_date="yesterday")])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert stats["INVALID_DATE"] == 1
        assert reasons_for(quarantine, "TXN-2000001") == ["INVALID_DATE"]
        # raw text preserved, not silently dropped
        assert quarantine.loc[quarantine["transaction_id"] == "TXN-2000001", "transaction_date"].iloc[0] == "yesterday"

    def test_transform_quarantines_null_date(self):
        df = make_df([make_row("TXN-2000002", transaction_date=None)])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert stats["INVALID_DATE"] == 1

    def test_transform_keeps_row_with_alternate_valid_format(self):
        df = make_df([make_row("TXN-2000003", transaction_date="15/01/2024")])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 1
        assert clean.iloc[0]["transaction_date"] == pd.Timestamp("2024-01-15")


# ===========================================================================
# 4. Soft-field standardization
# ===========================================================================

class TestSoftFieldStandardization:
    def test_currency_lowercase_is_uppercased(self):
        assert clean_currency(pd.Series(["usd", "eur"])).tolist() == ["USD", "EUR"]

    def test_currency_verbose_name_maps_to_code(self):
        assert clean_currency(pd.Series(["US Dollar"])).tolist() == ["USD"]

    def test_currency_blank_defaults_to_unknown(self):
        assert clean_currency(pd.Series([""])).tolist() == ["UNKNOWN"]

    def test_currency_null_defaults_to_unknown(self):
        assert clean_currency(pd.Series([None])).tolist() == ["UNKNOWN"]

    def test_currency_unrecognized_code_passes_through_uppercased(self):
        # No mapping for it, but it's not blank/null either - kept as-is.
        assert clean_currency(pd.Series(["cad"])).tolist() == ["CAD"]

    def test_currency_whitespace_is_trimmed(self):
        assert clean_currency(pd.Series([" gbp "])).tolist() == ["GBP"]

    def test_transform_customer_name_null_imputed_to_unknown(self):
        df = make_df([make_row("TXN-3000001", customer_name=None)])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["customer_name"] == "Unknown"

    def test_transform_customer_name_blank_imputed_to_unknown(self):
        df = make_df([make_row("TXN-3000002", customer_name="   ")])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["customer_name"] == "Unknown"

    def test_transform_customer_name_is_trimmed_not_replaced(self):
        df = make_df([make_row("TXN-3000003", customer_name="  Jane Doe  ")])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["customer_name"] == "Jane Doe"

    def test_transform_status_null_imputed_to_unknown(self):
        df = make_df([make_row("TXN-3000004", status=None)])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["status"] == "unknown"

    def test_transform_status_is_lowercased_and_trimmed(self):
        df = make_df([make_row("TXN-3000005", status="COMPLETED")])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["status"] == "completed"

    def test_transform_currency_variant_standardized_end_to_end(self):
        df = make_df([make_row("TXN-3000006", currency="")])
        clean, quarantine, stats = transform(df)
        assert clean.iloc[0]["currency"] == "UNKNOWN"


# ===========================================================================
# 5. Integration: mixed batch through transform(), no DB involved
# ===========================================================================

class TestTransformIntegration:
    def test_clean_and_quarantine_partition_the_input(self):
        """Every row must land in exactly one of clean/quarantine, and the
        two counts must always sum back to the number extracted - the same
        invariant the pipeline verifies against target_db after loading."""
        df = make_df([
            make_row("TXN-4000001"),                              # valid
            make_row("TXN-4000001"),                              # duplicate
            make_row("TXN-4000002", amount="bad"),                # invalid amount
            make_row("TXN-4000003", amount="-10"),                # negative
            make_row("TXN-4000004", transaction_date="yesterday"),  # invalid date
            make_row("TXN-4000005", customer_name=None, status=None, currency=""),  # soft-cleanable
        ])
        clean, quarantine, stats = transform(df)

        assert len(clean) + len(quarantine) == len(df)
        assert stats["cleaned"] + stats["quarantined"] == stats["extracted"]
        assert len(clean) == 2  # TXN-4000001 (first copy) and TXN-4000005
        assert len(quarantine) == 4

    def test_quarantine_dataframe_has_reject_reason_column(self):
        df = make_df([make_row("TXN-4000006", amount="-1")])
        _, quarantine, _ = transform(df)
        assert "reject_reason" in quarantine.columns
        assert quarantine.iloc[0]["reject_reason"] == "NEGATIVE_AMOUNT"

    def test_empty_input_produces_empty_outputs(self):
        df = make_df([])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 0
        assert len(quarantine) == 0
        assert stats["extracted"] == 0
        assert list(quarantine.columns) == RAW_COLUMNS + ["reject_reason"]

    def test_all_valid_rows_produce_no_quarantine(self):
        df = make_df([make_row(f"TXN-500{i:04d}") for i in range(5)])
        clean, quarantine, stats = transform(df)

        assert len(clean) == 5
        assert len(quarantine) == 0

    def test_clean_output_column_order_and_types(self):
        df = make_df([make_row("TXN-6000001")])
        clean, _, _ = transform(df)

        assert list(clean.columns) == [
            "transaction_id", "customer_id", "customer_name", "amount",
            "currency", "transaction_date", "status",
        ]
        assert pd.api.types.is_float_dtype(clean["amount"])
        assert pd.api.types.is_datetime64_any_dtype(clean["transaction_date"])

    def test_no_database_connection_required(self, monkeypatch):
        """Guardrail: transform() must not touch sqlalchemy/psycopg2 at all."""
        import etl_pipeline

        def fail(*a, **kw):
            raise AssertionError("transform() attempted to open a DB connection")

        monkeypatch.setattr(etl_pipeline, "create_engine", fail)
        df = make_df([make_row("TXN-7000001")])
        clean, quarantine, stats = transform(df)
        assert len(clean) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
