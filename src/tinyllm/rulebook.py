"""Deterministic merchant→category rulebook (era-2 lever #3).

Sources: Era-1 TRAIN-side review conventions + general knowledge. NEVER gate/final
rows — enforced by git history: this file's commit predates the gate_v2 review.
Used (a) to post-correct teacher labels at assembly and (b) as a Stage-4 serve-time
guard (reported as a separate column there — never silently folded in)."""

import re

RULES = {  # merchant-pattern: category — STARTER SET, grow to ~200
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
    "ola": "travel",
    "uber": "travel",
    "rapido": "travel",
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


def apply_rulebook(ex: dict) -> tuple[dict, bool]:
    """Correct category ONLY when the rule merchant IS the labeled counterparty."""
    lbl = ex["label"]
    cp = (lbl.get("counterparty") or "") if lbl.get("is_transaction") else ""
    if not cp:
        return ex, False
    for merchant, pat in _PATTERNS.items():
        if pat.search(cp):  # match the counterparty, not the whole SMS
            cat = RULES[merchant]
            if lbl.get("category") != cat:
                return {**ex, "label": {**lbl, "category": cat}}, True
            return ex, False
    return ex, False
