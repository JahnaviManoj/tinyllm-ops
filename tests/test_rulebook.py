from tinyllm.rulebook import apply_rulebook


def _ex(counterparty, category, is_transaction=True, txn_type="debit"):
    return {
        "sms": "x",
        "label": {
            "is_transaction": is_transaction,
            "txn_type": txn_type,
            "counterparty": counterparty,
            "category": category,
        },
    }


def test_merchant_corrects_category():
    ex, hit = apply_rulebook(_ex("DMart", "shopping"))
    assert hit and ex["label"]["category"] == "groceries"


def test_cab_is_transport_not_travel():
    ex, hit = apply_rulebook(_ex("Uber India", "travel"))
    assert hit and ex["label"]["category"] == "transport"


def test_matches_counterparty_only():
    # "paid to landlord via Amazon Pay": counterparty is the landlord, not Amazon
    ex, hit = apply_rulebook(_ex("landlord", "rent"))
    assert not hit and ex["label"]["category"] == "rent"


def test_word_boundaries():
    assert not apply_rulebook(_ex("VISA", "shopping"))[1]  # \bvi\b must not fire
    assert not apply_rulebook(_ex("license fee", "fees"))[1]


def test_salary_and_investment_are_protected():
    assert not apply_rulebook(_ex("SALARY-INDIGO", "salary", txn_type="credit"))[1]
    assert not apply_rulebook(_ex("Swiggy", "salary", txn_type="credit"))[1]
    assert not apply_rulebook(_ex("DMart Supermarket folio", "investment"))[1]


def test_amazon_prime_and_amazon_pay():
    ex, hit = apply_rulebook(_ex("Amazon Prime", "shopping"))
    assert hit and ex["label"]["category"] == "entertainment"
    assert not apply_rulebook(_ex("Amazon Pay", "food", txn_type="credit"))[1]
    ex, hit = apply_rulebook(_ex("AMAZON", "other"))
    assert hit and ex["label"]["category"] == "shopping"


def test_non_transaction_untouched():
    _, hit = apply_rulebook({"sms": "x", "label": {"is_transaction": False}})
    assert not hit
