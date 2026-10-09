"""Test-set generator (tutorial 1.4) — deliberately DIFFERENT from data_gen.py.

The training data comes from template shapes fed to gemini-3.5-flash-lite. If
the test set came from the same model+prompt, it would only measure "can the
student mimic that one teacher", and the teacher's quirks would sit on both
sides of the evaluation. So this generator differs on every axis it can:

  - different model  (gemini-3.5-flash, not -lite)
  - different prompt (scenario-driven: no template shapes are shown; the
    teacher invents formats the way real banks do)
  - cross-train dedupe: anything near-matching the training pool is dropped

The output is UNREVIEWED. Per the tutorial, every example must be hand-reviewed
before it may be called ground truth — write the reviewed file to
data/generated/test.jsonl yourself; this script refuses to use that name.
"""

import argparse
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
from tinyllm.schema import Category, Channel, ExpenseRecord, TxnType

TEST_TEACHER_MODEL = "gemini-3.5-flash"  # pinned; train teacher is 3.5-flash-lite

_CATEGORIES = ", ".join(c.value for c in Category)

POS_PROMPT = """You are building a TEST set of Indian bank transaction SMS.
Write {n} DIVERSE, realistic SMS exactly as real Indian banks and fintech apps
send them (HDFC, ICICI, SBI, Axis, Kotak, IDFC, Paytm, PhonePe, slice, Jupiter,
smaller banks too). Do not follow any fixed template — vary the wording,
ordering, punctuation, casing, truncation and formats as real senders do.

Every SMS must be a {txn_type} via {channel}. Spread the examples across many
spending categories from: {categories}. (salary only ever appears as a credit.)
Vary amount formats (Rs.450.00 / INR 1,234.56 / ₹99), account/card tails,
date formats, and include a few Hinglish messages. Never write two currency
markers next to each other (no "Rs.INR 450").

For each SMS output ONE JSON object on its own line, nothing else:
{{"sms": "<the sms text>", "label": <ExpenseRecord JSON per this schema>}}
Schema: {schema}
"""

NEG_PROMPT = """You are building the TEST set for an on-device SMS filter that
PROTECTS users: it must recognize non-transaction SMS — including scam
attempts — so it can flag them. Each example will be labeled accordingly
(scam examples get is_suspected_scam=true), so the filter learns to catch
them. Write {n} DIVERSE, realistic examples of: {kind}.
Every SMS must read as sent to a user in INDIA — Indian banks, card networks,
UPI apps, fintechs and businesses (HDFC, SBI, ICICI, Axis, Kotak, Paytm,
PhonePe, Amazon.in, Jio …), Rs./INR amounts, Indian phone and date formats.
No US/UK/Gulf senders, no $ or AED. Vary senders, wording and formats as real
SMS do.
Output one SMS per line as plain text — no numbering, no JSON, no markdown.
"""

NEGATIVE_KINDS = [
    ("OTP messages from banks, card networks and UPI apps", 22, False),
    ("promotional offers from banks and credit cards", 22, False),
    (
        (
            "bank/fintech informational SMS: bill-due reminders, balance updates, "
            "declined transactions, upcoming auto-debits, delivery updates"
        ),
        22,
        False,
    ),
    (
        (
            "fictional scam SMS in the style that targets Indian bank users, WITHOUT "
            "naming any real company — most real scams don't: 'Dear customer, your "
            "bank account will be suspended today, complete KYC at "
            "http://kyc-update-secure.in'; prize-draw wins asking for account details "
            "or a claim fee; card-block reactivation via an unfamiliar link; UPI "
            "collect-request tricks ('approve to RECEIVE Rs.5000'). Use concrete "
            "invented details: Rs amounts, digit tails, fictional lookalike URLs. "
            "No placeholders like [Bank] or [Link]"
        ),
        60,
        True,
    ),
]

# gemini-3.5-flash refuses scam-SMS generation under every framing tried
# (see docs/decisions.md); the scam spec alone uses a different teacher.
# Safe for decorrelation: scam labels are by-construction, not teacher-trusted.
SCAM_TEACHER_MODEL = "gemini-3.1-flash-lite"


def _specs():
    """Deterministic generation plan; spec_id keys the checkpoint."""
    specs = []
    for channel in Channel:
        for txn_type in TxnType:
            specs.append(
                {
                    "kind": "pos",
                    "channel": channel.value,
                    "txn_type": txn_type.value,
                    "n": 36,
                }
            )
    for desc, n, is_scam in NEGATIVE_KINDS:
        specs.append({"kind": "neg", "desc": desc, "n": n, "is_scam": is_scam})
    return specs


def generate_spec(spec) -> list[dict]:
    rows = []
    if spec["kind"] == "pos":
        prompt = POS_PROMPT.format(
            n=spec["n"],
            txn_type=spec["txn_type"],
            channel=spec["channel"],
            categories=_CATEGORIES,
            schema=json.dumps(ExpenseRecord.model_json_schema()),
        )
        for line in _call_teacher(prompt, model=TEST_TEACHER_MODEL).splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                ex = json.loads(line)
                ExpenseRecord.model_validate(ex["label"])
                ex["label"] = normalize_amount(normalize_currency(ex["label"]))
                ex["sms"] = fix_doubled_currency(ex["sms"])
                rows.append(ex)
            except Exception:
                continue
    else:
        label = {"is_transaction": False, "is_suspected_scam": spec["is_scam"]}
        model = SCAM_TEACHER_MODEL if spec["is_scam"] else TEST_TEACHER_MODEL
        prompt = NEG_PROMPT.format(n=spec["n"], kind=spec["desc"])
        for line in _call_teacher(prompt, model=model).splitlines():
            line = line.strip().strip("`").lstrip("-*• ").strip()
            if not _looks_like_sms(line):
                continue
            rows.append({"sms": fix_doubled_currency(line), "label": dict(label)})
    return rows


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate the UNREVIEWED 1.4 test set from a different teacher."
    )
    parser.add_argument("--out", default="data/generated/test_unreviewed.jsonl")
    parser.add_argument(
        "--train-pool",
        default="data/generated/train.jsonl",
        help="existing train data to dedupe the test set against",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip specs already checkpointed in <out>.raw",
    )
    args = parser.parse_args()
    if os.path.basename(args.out) == "test.jsonl":
        raise SystemExit(
            "test.jsonl is reserved for the HAND-REVIEWED set — "
            "generate to test_unreviewed.jsonl and review it first."
        )

    raw_path = args.out + ".raw"
    prior = []
    if args.resume and os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    prior.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    done = {ex["spec_id"] for ex in prior}

    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    with open(raw_path, "w") as raw_f:
        raw_f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in prior)
        raw_f.flush()
        for spec_id, spec in enumerate(_specs()):
            if spec_id in done:
                print(f"spec {spec_id}: skipped (resume)", flush=True)
                continue
            rows = generate_spec(spec)
            for ex in rows:
                ex["spec_id"] = spec_id
                raw_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            raw_f.flush()
            prior += rows
            print(f"spec {spec_id}: +{len(rows)} (total {len(prior)})", flush=True)

    train_pool = []
    if os.path.exists(args.train_pool):
        with open(args.train_pool) as f:
            train_pool = [json.loads(line) for line in f]
    kept = dedupe(prior, against=train_pool)
    with open(args.out, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"generated {len(prior)}, kept {len(kept)} after dedupe "
        f"(within-set + against {len(train_pool)} train rows) → {args.out}"
    )
    print(
        "NEXT: hand-review every example, then save the reviewed set as data/generated/test.jsonl"
    )


if __name__ == "__main__":
    main()
