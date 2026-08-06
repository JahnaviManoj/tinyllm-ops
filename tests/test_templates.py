import re

from tinyllm.templates import TEMPLATES

ALLOWED = {
    "amount",
    "bal",
    "tail",
    "merchant",
    "vpa",
    "name",
    "bank",
    "date",
    "ref",
    "otp",
    "phone",
    "url",
}


def test_enough_templates():
    assert 30 <= len(TEMPLATES) <= 60


def test_placeholders_from_fixed_vocabulary():
    for t in TEMPLATES:
        assert set(re.findall(r"\{(\w+)\}", t.shape)) <= ALLOWED, t.shape


def test_no_real_looking_numbers():
    # any run of 3+ digits outside a {placeholder} smells like unsanitized data
    for t in TEMPLATES:
        assert not re.search(r"\d{3,}", t.shape), t.shape


def test_axis_coverage():
    channels = {t.channel for t in TEMPLATES if t.channel}
    assert channels == {"UPI", "card", "netbanking", "ATM", "wallet"}
    assert {t.hint_txn_type for t in TEMPLATES} >= {"debit", "credit"}
    assert sum(t.is_negative for t in TEMPLATES) >= 5
    assert sum(t.is_scam for t in TEMPLATES) >= 3
