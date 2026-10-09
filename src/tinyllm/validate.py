import argparse
import json
import re
from collections import Counter

from tinyllm.schema import ExpenseRecord


def validate_dataset(examples: list[dict]) -> dict:
    failures, cat_counts, tmpl_counts = [], Counter(), Counter()
    for i, ex in enumerate(examples):
        try:
            rec = ExpenseRecord.model_validate(ex["label"])
            cat_counts[rec.category] += 1
            tmpl_counts[ex.get("template_id")] += 1
        except Exception as e:
            failures.append({"index": i, "error": str(e)})
    report = {
        "total": len(examples),
        "valid": len(examples) - len(failures),
        "failures": failures,
        "category_balance": dict(cat_counts),
        "template_balance": dict(tmpl_counts),
    }
    if failures:
        for f in failures[:10]:  # show what to regenerate, not just the count
            print(f"  bad example {f['index']}: {f['error']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        raise SystemExit(
            f"VALIDATION FAILED: {len(failures)} bad examples — regenerate them."
        )
    return report


# Schema-valid but suspect: the teacher sometimes glues the style's currency
# format onto the template's own prefix ("Rs.INR 5400.00"). Not a gate failure
# (real SMS are messy), but if it's common the style prompt needs tightening.
DOUBLED_CURRENCY = re.compile(r"(?<![A-Za-z])(?:₹|Rs\.?|INR)\s*(?:₹|Rs\.?|INR)\s*\d")


def main():
    parser = argparse.ArgumentParser(
        description="Fail-loudly validation gate for generated JSONL datasets."
    )
    parser.add_argument(
        "paths", nargs="+", help="JSONL file(s), e.g. data/generated/train.jsonl"
    )
    args = parser.parse_args()
    for path in args.paths:
        with open(path) as f:
            examples = [json.loads(line) for line in f]
        print(f"== {path} ({len(examples)} examples)")
        report = validate_dataset(examples)  # SystemExit here if anything is invalid
        print(json.dumps(report, indent=2, default=str))
        doubled = sum(bool(DOUBLED_CURRENCY.search(ex["sms"])) for ex in examples)
        if doubled:
            pct = 100 * doubled / len(examples)
            print(
                f"note: {doubled} SMS ({pct:.1f}%) have a doubled currency prefix "
                f"(e.g. 'Rs.INR 5400') — teacher artifact, see docs/decisions.md"
            )
        non_iso = sum(
            1
            for ex in examples
            if (c := ex["label"].get("currency")) and not re.fullmatch(r"[A-Z]{3}", c)
        )
        if non_iso:
            print(
                f"note: {non_iso} labels have a non-ISO currency (e.g. 'Rs.' instead "
                f"of 'INR') — teacher artifact, see docs/decisions.md"
            )
        # class-balance guards (era 2): starvation caught BEFORE training
        n = len(examples)
        if n:
            scam = sum(1 for ex in examples if ex["label"].get("is_suspected_scam"))
            neg = sum(1 for ex in examples if not ex["label"].get("is_transaction"))
            if scam / n < 0.02:
                print(
                    f"WARN: scam share {100 * scam / n:.1f}% < 2% — class starvation risk"
                )
            if not 0.05 <= neg / n <= 0.40:
                print(f"WARN: negative share {100 * neg / n:.1f}% outside 5–40% band")
        odd_amount = sum(
            1
            for ex in examples
            if (a := ex["label"].get("amount")) and not re.fullmatch(r"\d+\.\d{2}", a)
        )
        if odd_amount:
            print(
                f"note: {odd_amount} amounts break the two-decimal convention "
                f"('2000.00', decided 2026-08-04) — see docs/decisions.md"
            )


if __name__ == "__main__":
    main()
