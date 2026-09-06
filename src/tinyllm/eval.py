from sklearn.metrics import f1_score, precision_score, recall_score
from tinyllm.schema import ExpenseRecord

FIELDS = [
    "is_transaction",
    "txn_type",
    "amount",
    "currency",
    "counterparty",
    "account_tail",
    "channel",
    "category",
]


def row_correct(prediction: str, gold: dict) -> bool:
    """Exact match for ONE row — the unit the tie rule / McNemar test counts.
    Same definition evaluate() uses for exact_match: parse the raw string as an
    ExpenseRecord, every field equal (scam flag included); unparseable = wrong."""
    try:
        pred = ExpenseRecord.model_validate_json(prediction)
    except Exception:
        return False
    g = ExpenseRecord.model_validate(gold)
    return all(getattr(pred, f) == getattr(g, f) for f in FIELDS) and (
        pred.is_suspected_scam == g.is_suspected_scam
    )


def evaluate(predictions: list[str], gold: list[dict]) -> dict:
    # an empty eval set is a zeroed report, never a pass: every key downstream gates
    # read stays present, and nothing divides by zero
    if not gold:
        return {
            "json_parse_rate": 0.0,
            "per_field_accuracy": {f: 0.0 for f in FIELDS},
            "category_macro_f1": 0.0,
            "scam_precision": 0.0,
            "scam_recall": 0.0,
            "exact_match": 0.0,
        }

    parsed, parse_fail = [], 0
    for raw in predictions:
        try:
            parsed.append(ExpenseRecord.model_validate_json(raw))
        except Exception:
            parsed.append(None)
            parse_fail += 1

    fields = FIELDS
    field_acc = {f: 0 for f in fields}
    for pred, g in zip(parsed, gold):
        g_rec = ExpenseRecord.model_validate(g)
        if pred is None:
            continue
        for f in fields:
            if getattr(pred, f) == getattr(g_rec, f):
                field_acc[f] += 1
    # one definition of "this row is right" — shared with model_eval --rows-out
    exact = sum(row_correct(p, g) for p, g in zip(predictions, gold))

    n = len(gold)
    y_true = [
        bool(g.get("is_suspected_scam", False)) for g in gold
    ]  # key may be absent
    y_pred = [(p.is_suspected_scam if p else False) for p in parsed]
    return {
        "json_parse_rate": 1 - parse_fail / n,
        "per_field_accuracy": {f: c / n for f, c in field_acc.items()},
        # sklearn can't handle None labels: None category (non-transaction SMS) → "none";
        # unparseable outputs get their own "PARSE_FAIL" label so they always count as wrong
        "category_macro_f1": f1_score(
            [(g.get("category") or "none") for g in gold],
            [
                "PARSE_FAIL"
                if p is None
                else (p.category.value if p.category else "none")
                for p in parsed
            ],
            average="macro",
        ),
        "scam_precision": precision_score(y_true, y_pred, zero_division=0),
        "scam_recall": recall_score(y_true, y_pred, zero_division=0),
        "exact_match": exact / n,
    }
