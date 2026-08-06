import json
from tinyllm.eval import evaluate

GOLD = [
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
    },
    {
        "is_transaction": True,
        "txn_type": "credit",
        "amount": "100.00",
        "currency": "INR",
        "counterparty": "FRAUDSTER",
        "account_tail": "9999",
        "channel": "UPI",
        "category": "transfer",
        "is_suspected_scam": True,
    },
]


def test_metrics_match_hand_computation():
    preds = [json.dumps(GOLD[0]), '{"broken json']  # 1 perfect, 1 unparseable
    report = evaluate(preds, GOLD)
    assert report["json_parse_rate"] == 0.5  # 1 of 2 parsed
    assert report["exact_match"] == 0.5  # the parsed one is perfect
    assert report["per_field_accuracy"]["amount"] == 0.5
    assert report["scam_recall"] == 0.0  # the scam example didn't parse → missed


def test_all_unparseable_scores_zero():
    report = evaluate(["garbage", "garbage"], GOLD)
    assert report["json_parse_rate"] == 0.0
    assert report["exact_match"] == 0.0  # unparseable = wrong, never skipped


NON_TXN = {
    "is_transaction": False,
    "txn_type": None,
    "amount": None,
    "currency": None,
    "counterparty": None,
    "account_tail": None,
    "channel": None,
    "category": None,
    "is_suspected_scam": False,
}


def test_none_fields_on_non_transaction_sms():
    # an OTP/promo SMS: every field None is the *correct* answer, so None==None must
    # score as a hit, not be skipped
    report = evaluate([json.dumps(NON_TXN)], [NON_TXN])
    assert report["exact_match"] == 1.0
    assert report["per_field_accuracy"]["amount"] == 1.0
    assert report["per_field_accuracy"]["category"] == 1.0

    # and predicting a transaction where there is none is wrong everywhere
    report = evaluate([json.dumps(GOLD[0])], [NON_TXN])
    assert report["json_parse_rate"] == 1.0  # parsed fine, just wrong
    assert report["exact_match"] == 0.0
    assert report["per_field_accuracy"]["is_transaction"] == 0.0


def test_empty_input_does_not_divide_by_zero():
    report = evaluate([], [])
    assert report["json_parse_rate"] == 0.0
    assert report["exact_match"] == 0.0
    assert all(v == 0.0 for v in report["per_field_accuracy"].values())
