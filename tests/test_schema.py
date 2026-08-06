import pytest
from tinyllm.schema import ExpenseRecord


def test_valid_record_parses():
    ExpenseRecord.model_validate(
        {
            "is_transaction": True,
            "txn_type": "debit",
            "amount": "450.00",
            "currency": "INR",
            "counterparty": "SWIGGY",
            "account_tail": "1234",
            "channel": "UPI",
            "category": "food",
            "is_suspected_scam": False,
        }
    )


def test_bad_amount_rejected():
    with pytest.raises(Exception):
        ExpenseRecord.model_validate(
            {"is_transaction": True, "amount": "45O.OO"}
        )  # letter O!


def test_unknown_category_rejected():
    with pytest.raises(Exception):
        ExpenseRecord.model_validate(
            {"is_transaction": True, "category": "crypto_gambling"}
        )
