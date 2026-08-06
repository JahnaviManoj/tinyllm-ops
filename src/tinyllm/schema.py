from enum import Enum
from pydantic import BaseModel, field_validator
from decimal import Decimal, InvalidOperation


class TxnType(str, Enum):
    debit = "debit"
    credit = "credit"


class Channel(str, Enum):
    upi = "UPI"
    card = "card"
    netbanking = "netbanking"
    atm = "ATM"
    wallet = "wallet"


class Category(str, Enum):
    food = "food"
    groceries = "groceries"
    transport = "transport"
    shopping = "shopping"
    bills_utilities = "bills_utilities"
    entertainment = "entertainment"
    health = "health"
    education = "education"
    travel = "travel"
    rent = "rent"
    salary = "salary"
    transfer = "transfer"
    investment = "investment"
    fees = "fees"
    other = "other"


class ExpenseRecord(BaseModel):
    is_transaction: bool
    txn_type: TxnType | None = None  # None when is_transaction is False
    amount: str | None = None  # decimal STRING ("450.00") — floats corrupt money values
    currency: str | None = None
    counterparty: str | None = None  # who you paid / who paid you
    account_tail: str | None = None  # last digits of the account, e.g. "1234"
    channel: Channel | None = None
    category: Category | None = None
    is_suspected_scam: bool = False

    @field_validator("amount")
    @classmethod
    def amount_is_decimal(cls, v):
        if v is not None:
            try:
                Decimal(v)
            except InvalidOperation:
                raise ValueError(f"amount is not a valid decimal string: {v!r}")
        return v
