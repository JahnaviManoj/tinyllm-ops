"""Regex baseline (tutorial 2.4) — the honest, non-strawman competitor.

Real expense apps parse SMS with per-bank regex templates. This baseline IS
that app: every shape in templates.py compiles into one regex (literal text
escaped, placeholders becoming capture groups), so it covers exactly the
formats a diligent regex maintainer would have encoded. Two built-in
weaknesses are the point of the comparison:

  - category is never inferred (a regex can't know SWIGGY is food), and
  - any SMS whose format isn't in the template list returns None — total
    failure on unseen shapes, which is where the fine-tuned model earns
    its place.

CLI: uv run python -m tinyllm.regex_baseline data/generated/val.jsonl \
         data/generated/test_gate.jsonl
prints match-rate + the 2.1 eval report per file (val ≈ seen shapes,
gate ≈ unseen real-world shapes from the different-teacher test set).
"""

import argparse
import json
import re

from tinyllm.data_gen import normalize_amount
from tinyllm.eval import evaluate
from tinyllm.templates import TEMPLATES

# First occurrence captures; repeats of the same placeholder fall back to the
# generic (uncaptured) pattern since a named group may appear only once.
CAPTURE = {
    "amount": r"(?P<amount>[\d,]+(?:\.\d+)?)",
    "tail": r"(?P<tail>\d{2,6})",
    "merchant": r"(?P<counterparty>[^\n]{2,40}?)",
    "name": r"(?P<counterparty>[^\n]{2,40}?)",
    "vpa": r"(?P<counterparty>[\w.-]+@\w+)",
}
GENERIC = {
    "amount": r"[\d,]+(?:\.\d+)?",
    "bal": r"[\d,]+(?:\.\d+)?",
    "tail": r"\d{2,6}",
    "merchant": r"[^\n]{2,40}?",
    "name": r"[^\n]{2,40}?",
    "vpa": r"[\w.-]+@\w+",
    "bank": r"[A-Za-z][\w ]{1,25}?",
    "date": r"[\w/:,. -]{5,25}?",
    "ref": r"[\w/-]{4,25}",
    "otp": r"\d{4,8}",
    "phone": r"[\d+-]{5,15}",
    "url": r"\S+",
}
_TOKEN = re.compile(r"\{(\w+)\}")


def _flex_ws(escaped: str) -> str:
    """Collapse whitespace runs in re.escape output to \\s+ . Python 3.7+
    escapes spaces as '\\ ', so the run pattern must eat the backslashes too —
    a bare \\s+ substitution would leave orphan backslashes behind."""
    return re.sub(r"(?:\\\s|\s)+", r"\\s+", escaped)


def _shape_to_regex(shape: str) -> re.Pattern:
    parts, used, last = [], set(), 0
    for m in _TOKEN.finditer(shape):
        parts.append(
            _flex_ws(re.escape(shape[last : m.start()]))
        )  # tolerate spacing variation
        ph = m.group(1)
        if (
            ph in CAPTURE
            and ph not in used
            and "counterparty"
            not in (u for u in used if ph in ("merchant", "name", "vpa"))
        ):
            grp = CAPTURE[ph]
            # merchant/name/vpa share the counterparty group — only one may capture
            if "(?P<counterparty>" in grp:
                if any(p in used for p in ("merchant", "name", "vpa")):
                    grp = GENERIC[ph]
            parts.append(grp)
            used.add(ph)
        else:
            parts.append(GENERIC[ph])
        last = m.end()
    parts.append(_flex_ws(re.escape(shape[last:])))
    return re.compile("".join(parts), re.IGNORECASE)


PATTERNS = [(_shape_to_regex(t.shape), t) for t in TEMPLATES]


def regex_parse(sms: str) -> dict | None:
    """Parse via the template list; None = format not in the list (real apps
    silently drop these — the generalization failure this baseline exists to show)."""
    for pattern, t in PATTERNS:
        m = pattern.search(sms)
        if not m:
            continue
        if t.is_negative:
            return {
                "is_transaction": False,
                "txn_type": None,
                "amount": None,
                "currency": None,
                "counterparty": None,
                "account_tail": None,
                "channel": None,
                "category": None,
                "is_suspected_scam": t.is_scam,
            }
        g = m.groupdict()
        label = {
            "is_transaction": True,
            "txn_type": t.hint_txn_type,
            "amount": (g.get("amount") or "").replace(",", "") or None,
            "currency": "USD" if "USD" in t.shape else "INR",
            "counterparty": (g.get("counterparty") or "").strip() or None,
            "account_tail": g.get("tail"),
            "channel": t.channel,
            "category": None,  # a regex cannot know SWIGGY is food
            "is_suspected_scam": False,
        }
        return normalize_amount(label)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the regex baseline on labeled JSONL files."
    )
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        rows = [json.loads(line) for line in open(path)]
        preds = [regex_parse(r["sms"]) for r in rows]
        matched = sum(p is not None for p in preds)
        report = evaluate(
            [json.dumps(p) if p else "" for p in preds],
            [r["label"] for r in rows],
        )
        print(
            f"== {path}: matched {matched}/{len(rows)} ({100 * matched / len(rows):.0f}%)"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
