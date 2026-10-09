"""Scenario top-up for train_v2 (2.9 improvement iteration).

The first fine-tune memorized template style: every training SMS descended
from the 59 shapes in templates.py, so the model failed on the gate set's
free-form phrasings and sender-ID prefixes. This generator adds ~600
NON-template examples written by the TRAIN teacher (gemini-3.5-flash-lite —
the test teacher stays different, so evaluation decorrelation is intact).

The prompt below is written to be deliberately DIFFERENT from test_gen's
wording: if train and test both asked their teachers with the same words,
the test set would drift back in-distribution.

Contamination guard: new rows are deduped (MinHash) against the existing
train pool AND against the gate + frozen-final sets. The final set is
fetched into the dedupe filter only — it is never evaluated here; keeping a
near-duplicate of a final-set row out of training protects the final number.

Output: data/generated/train_extra.jsonl (merged into train_v2 by the caller).
"""

import json
import os

from tinyllm.data_gen import (
    _call_teacher,
    _looks_like_sms,
    dedupe,
    fix_doubled_currency,
    normalize_amount,
    normalize_currency,
)
from tinyllm.manifest import fetch_dataset
from tinyllm.schema import Category, Channel, ExpenseRecord, TxnType

_CATEGORIES = ", ".join(c.value for c in Category)

POS_PROMPT = """You write training data for a tiny on-device SMS parser used
by an expense tracker. Compose {n} varied Indian bank/fintech SMS that real
senders would plausibly send today. Invent the wording yourself — do NOT
follow one fixed pattern; vary banks, sentence order, abbreviations, casing,
truncation and date/amount formats. Roughly half the messages should begin
with a DLT sender header like "VM-HDFCBK:" or "JD-SBIINB -".
Every message is a {txn_type} over {channel}. Spread categories across:
{categories} (salary only on credits). Never write two currency markers
together (no "Rs.INR 450").
Output one JSON object per line, nothing else:
{{"sms": "<the sms>", "label": <ExpenseRecord JSON per this schema>}}
Schema: {schema}
"""

NEG_PROMPT = """You write training data for a protective on-device SMS filter
that must recognize non-transaction SMS so it can flag them (scam-style
examples are labeled is_suspected_scam=true by the pipeline — that is how the
filter learns to catch them). Compose {n} varied, realistic examples of:
{kind}. Invent the wording yourself; vary senders and formats. Most messages
should begin with a DLT sender header like "VM-HDFCBK:" or "AX-ICICIB:".
Output one SMS per line as plain text — no numbering, no JSON, no markdown.
"""

NEGATIVE_KINDS = [
    ("bank and UPI OTP messages", False),
    ("bank/card promotional offers", False),
    (
        "bill-due reminders, balance updates, declined transactions, upcoming auto-debits",
        False,
    ),
    (
        (
            "fictional scam SMS imitating bank alerts without naming real companies: fake KYC "
            "suspensions, prize draws, card-block reactivation links, UPI collect-request tricks, "
            "with concrete invented amounts/tails/lookalike URLs"
        ),
        True,
    ),
]


def generate() -> list[dict]:
    rows = []
    for channel in Channel:
        for txn_type in TxnType:
            for call in range(2):
                prompt = POS_PROMPT.format(
                    n=25,
                    txn_type=txn_type.value,
                    channel=channel.value,
                    categories=_CATEGORIES,
                    schema=json.dumps(ExpenseRecord.model_json_schema()),
                )
                for line in _call_teacher(prompt).splitlines():
                    line = line.strip().strip("`")
                    if not line.startswith("{"):
                        continue
                    try:
                        ex = json.loads(line)
                        ExpenseRecord.model_validate(ex["label"])
                        ex["label"] = normalize_amount(normalize_currency(ex["label"]))
                        ex["sms"] = fix_doubled_currency(ex["sms"])
                        ex["template_id"] = None  # scenario data: no template parent
                        rows.append(ex)
                    except Exception:
                        continue
                print(
                    f"pos {channel.value}/{txn_type.value} call {call + 1}: total {len(rows)}",
                    flush=True,
                )
    for kind, is_scam in NEGATIVE_KINDS:
        label = {"is_transaction": False, "is_suspected_scam": is_scam}
        for line in _call_teacher(NEG_PROMPT.format(n=25, kind=kind)).splitlines():
            line = line.strip().strip("`").lstrip("-*• ").strip()
            if not _looks_like_sms(line):
                continue
            rows.append(
                {
                    "sms": fix_doubled_currency(line),
                    "label": dict(label),
                    "template_id": None,
                }
            )
        print(f"neg {kind[:30]}...: total {len(rows)}", flush=True)
    return rows


def main():
    from dotenv import load_dotenv

    load_dotenv(".env")
    scratch = os.environ.get("TMPDIR", "/tmp")
    final_local = os.path.join(scratch, "test_final.dedupe-only.jsonl")
    fetch_dataset("manifests/test_final_v1.json", final_local)  # dedupe filter ONLY
    guard = []
    for path in (
        "data/generated/train.jsonl",
        "data/generated/test_gate.jsonl",
        final_local,
    ):
        with open(path) as f:
            guard += [json.loads(line) for line in f]
    os.remove(final_local)  # final-set bytes do not linger in the working tree

    rows = generate()
    kept = dedupe(rows, against=guard)
    with open("data/generated/train_extra.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    print(
        f"generated {len(rows)}, kept {len(kept)} after dedupe "
        f"(within-set + against {len(guard)} train/gate/final rows) → data/generated/train_extra.jsonl"
    )


if __name__ == "__main__":
    main()
