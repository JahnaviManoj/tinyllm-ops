"""Deterministic merchant→category rulebook (era-2 lever #3).

Sources: Era-1 TRAIN-side review conventions + general knowledge. NEVER gate/final
rows — enforced by git history: this file's commit predates the gate_v2 review.
Used (a) to post-correct teacher labels at assembly and (b) as a Stage-4 serve-time
guard (reported as a separate column there — never silently folded in)."""

import re

# Never overridden by a merchant match: a credit labelled salary is a payout whatever the
# employer is called ("SALARY-INDIGO" is not travel); a SIP/folio debit is an investment
# whatever the fund is named. Found on the train_v3 dry run at C7 (2026-09-10), 12 rows.
PROTECTED_CATEGORIES = {"salary", "investment"}
# Counterparties that are a payment rail, not a merchant — the rulebook leaves them alone
# ("Amazon Pay: cashback for your Swiggy order" is not shopping).
SKIP_COUNTERPARTIES = ("amazon pay",)

RULES = {  # merchant-pattern: category — STARTER SET, grow to ~200
    # more specific patterns FIRST: dict order is match order
    "amazon prime": "entertainment",
    # transport = intra-city rides (cab/auto/metro/fuel); travel = intercity/flights/
    # trains/hotels. Era-1 hand-reviewed gold: uber/ola → transport 15/15 (2026-09-07 fix,
    # justified from gate_v1/final_v1 + train_v3 conventions, NOT from gate_v2 rows).
    "swiggy": "food",
    "zomato": "food",
    "dominos": "food",
    "kfc": "food",
    "zepto": "groceries",
    "blinkit": "groceries",
    "bigbasket": "groceries",
    "dmart": "groceries",
    "jiomart": "groceries",
    "irctc": "travel",
    "makemytrip": "travel",
    "redbus": "travel",
    "ola": "transport",
    "uber": "transport",
    "rapido": "transport",
    "indigo": "travel",
    "bescom": "bills_utilities",
    "msedcl": "bills_utilities",
    "tneb": "bills_utilities",
    "airtel": "bills_utilities",
    "jio": "bills_utilities",
    "netflix": "entertainment",
    "hotstar": "entertainment",
    "spotify": "entertainment",
    "bookmyshow": "entertainment",
    "amazon": "shopping",
    "flipkart": "shopping",
    "myntra": "shopping",
    "ajio": "shopping",
    "nykaa": "shopping",
    "apollo": "health",
    "pharmeasy": "health",
    "1mg": "health",
    "practo": "health",
    "policybazaar": "insurance",
    "zerodha": "investment",
    "groww": "investment",
    "upstox": "investment",
}
_PATTERNS = {k: re.compile(rf"\b{re.escape(k)}\b", re.I) for k in RULES}
_SKIP = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in SKIP_COUNTERPARTIES]


def apply_rulebook(ex: dict) -> tuple[dict, bool]:
    """Correct category ONLY when the rule merchant IS the labeled counterparty."""
    lbl = ex["label"]
    cp = (lbl.get("counterparty") or "") if lbl.get("is_transaction") else ""
    if not cp or lbl.get("category") in PROTECTED_CATEGORIES:
        return ex, False
    if any(p.search(cp) for p in _SKIP):
        return ex, False
    for merchant, pat in _PATTERNS.items():
        if pat.search(cp):  # match the counterparty, not the whole SMS
            cat = RULES[merchant]
            if lbl.get("category") != cat:
                return {**ex, "label": {**lbl, "category": cat}}, True
            return ex, False
    return ex, False
