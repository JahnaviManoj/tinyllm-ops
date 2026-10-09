import argparse
import itertools
import json
import os
import random
import re
import threading
import time

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from tinyllm.schema import Category, ExpenseRecord, TxnType
from tinyllm.templates import TEMPLATES, Template

# Pinned, not "-latest": the dataset must be reproducible from the manifest.
# (gemini-2.5-flash from the tutorial is closed to new accounts as of 2026-08,
# and gemini-3.6-flash's free tier is capped at 20 requests/day — a full run
# needs ~620. The lite tier has the bulk-friendly quota; the schema-validation
# gate below compensates for the weaker teacher by dropping bad rows.)
TEACHER_MODEL = "gemini-3.5-flash-lite"

# Lazy client: module import must never require secrets (pytest collection,
# `from tinyllm.data_gen import dedupe` on CI). Only generating needs the key.
_client = None


_client_lock = threading.Lock()


def get_client():
    global _client
    with (
        _client_lock
    ):  # baselines --workers: concurrent callers must not race to build one
        if _client is None:
            # One attempt at the SDK layer: retries live in _call_teacher only, so the two
            # layers can never multiply (free-tier DAILY quota counts failed attempts too).
            _client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=genai_types.HttpOptions(
                    retry_options=genai_types.HttpRetryOptions(attempts=1)
                ),
            )
        return _client


GEN_PROMPT = """You generate realistic Indian bank/UPI transaction SMS for training data.
Template shape: {template}
Category: {category} | Type: {txn_type} | Style: {style}
Generate {n} DIFFERENT examples. For each, output one JSON object per line:
{{"sms": "<the sms text>", "label": <ExpenseRecord JSON matching the schema below>}}
The style describes only how the amount NUMBER is written. If the template
already has a currency marker (Rs., INR, ₹) before the amount, keep that one
and do not add another — never write doubled markers like "Rs.INR 450".
Schema: {schema}
"""

# The teacher sometimes obeys an "amount as INR 1,234.56" style by writing the
# marker INTO the amount anyway, doubling the template's own prefix
# ("Rs.INR 450"). Repair deterministically at ingest: collapse adjacent
# currency markers before a digit, keeping the first (the template's own).
_DOUBLED_CURRENCY = re.compile(
    r"(?<![A-Za-z])(₹|Rs\.|Rs|INR)\s*(?:₹|Rs\.|Rs|INR)\s*(?=\d)"
)

# Teachers sometimes copy the SMS's literal marker into the currency field
# ("Rs.", "₹") instead of the ISO code the schema intends.
_INR_ALIASES = {"rs", "rs.", "₹", "inr", "rupees", "rupee"}


def normalize_currency(label: dict) -> dict:
    c = label.get("currency")
    if c and c.strip().rstrip(".").lower() in {a.rstrip(".") for a in _INR_ALIASES}:
        label["currency"] = "INR"
    return label


def normalize_amount(label: dict) -> dict:
    """Project convention (decided 2026-08-04): amounts are decimal strings
    with exactly two places — "2000.00", not "2000" — regardless of how the
    SMS wrote it. The eval comparator string-matches amounts, so labels must
    be consistent about it."""
    from decimal import Decimal

    a = label.get("amount")
    if a:
        label["amount"] = str(Decimal(a.replace(",", "")).quantize(Decimal("0.01")))
    return label


def fix_doubled_currency(sms: str) -> str:
    def keep_first(m):
        first = m.group(1)
        return first if first == "₹" or first.endswith(".") else first + " "

    prev = None
    while prev != sms:  # triples collapse over two passes
        prev = sms
        sms = _DOUBLED_CURRENCY.sub(keep_first, sms)
    return sms


# Negatives: the label is known by construction, so only ask for SMS text.
# The protects-users framing matters: without it, teachers refuse the
# scam-shaped templates and return a smishing explainer instead of data.
NEG_PROMPT = """You generate training data for an on-device SMS filter that
PROTECTS users: it must recognize OTPs, promos, reminders and scam attempts so
it can flag them. Each example below will be labeled accordingly (scam-shaped
examples are labeled is_suspected_scam=true), so the filter learns to catch
them. These are NOT transaction alerts (no money actually moved).
Template shape: {template}
Style: {style}
Generate {n} DIFFERENT examples matching this shape, with varied realistic
values filled in for the placeholders. Output one SMS per line as plain text —
no numbering, no JSON, no markdown.
"""

# Teacher output that is not an SMS: refusals, markdown structure, meta-prose.
_NOT_SMS_PREFIXES = (
    "sorry",
    "i cannot",
    "i can't",
    "i am unable",
    "as an ai",
    "here are",
    "sure,",
    "note:",
    "these messages",
    "mechanism",
    "structural",
    "linguistic",
)
_SMSISH = re.compile(
    r"(bank|a/c|account|card|upi|kyc|otp|rs\.?\s?\d|inr|₹|http|"
    r"call|sms|prize|won|block|wallet|loan|bill|order|deliver)",
    re.IGNORECASE,
)


def _looks_like_sms(line: str) -> bool:
    if not line or line.startswith("#") or "**" in line:
        return False
    if re.search(r"\[(?:Bank|Link|Amount|Number|Name|URL|X+)\]", line, re.IGNORECASE):
        return False  # teacher hedged with placeholders instead of values
    if re.search(
        r"classifier|synthetic|test suite|these examples|evaluat|"
        r"linguistic|structural feature",
        line,
        re.IGNORECASE,
    ):
        return False  # meta-prose about the task instead of an SMS
    if line.lower().startswith(_NOT_SMS_PREFIXES):
        return False
    return bool(_SMSISH.search(line))


STYLES = [
    "clean and complete",
    "truncated mid-sentence",
    "Hinglish mix (e.g. 'aapke account se debit hua')",
    "amount as ₹1,234.56",
    "amount as INR 1234.56",
    "amount as Rs.1234",
]


def _call_teacher(prompt: str, retries: int = 4, model: str = TEACHER_MODEL) -> str:
    """One generate_content call with backoff on transient failures: per-minute
    429s, 5xx, and network drops (httpx.TransportError covers RemoteProtocolError/
    ConnectError/timeouts — a long generation run must survive a dropped connection).

    Two rules learned on gate_v2 (2026-09-07), where the free tier allows only
    20 gemini-3.5-flash requests per DAY and every failed attempt counts:
      - a daily-quota 429 (quotaId ...PerDay...) raises at once — retrying cannot
        help and the old 6-step backoff burned minutes for nothing;
      - transient errors get at most `retries` attempts with 30 s → 60 s → 120 s
        waits, so one 503 storm costs a few requests, not the whole day."""
    for attempt in range(retries):
        try:
            resp = get_client().models.generate_content(model=model, contents=prompt)
            return resp.text or ""
        except (genai_errors.APIError, httpx.TransportError) as e:
            if (
                isinstance(e, genai_errors.APIError)
                and e.code == 429
                and "PerDay" in str(e)
            ):
                raise SystemExit(
                    f"DAILY free-tier quota exhausted for {model} — resets 00:00 "
                    "America/Los_Angeles. Re-run with --resume tomorrow."
                ) from e
            transient = isinstance(e, httpx.TransportError) or e.code in (
                429,
                500,
                502,
                503,
                504,
            )
            if transient and attempt < retries - 1:
                time.sleep(min(30 * 2**attempt, 120))
                continue
            raise
    return ""


def generate_batch(
    template: str, category: str, txn_type: str, style: str, n: int = 15
) -> list[dict]:
    """One API call → 10-20 labeled examples (batching saves your free-tier quota)."""
    prompt = GEN_PROMPT.format(
        template=template,
        category=category,
        txn_type=txn_type,
        style=style,
        n=n,
        schema=json.dumps(ExpenseRecord.model_json_schema()),
    )
    examples = []
    for line in _call_teacher(prompt).splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            ex = json.loads(line)
            ExpenseRecord.model_validate(
                ex["label"]
            )  # validate NOW — reject bad teacher output
            ex["label"] = normalize_amount(normalize_currency(ex["label"]))
            ex["sms"] = fix_doubled_currency(ex["sms"])
            examples.append(ex)
        except Exception:
            continue  # bad line → drop it, regenerate later
    return examples


def generate_negative_batch(template: Template, style: str, n: int = 15) -> list[dict]:
    """Negatives (OTP/promo/scam shapes): the teacher only writes the SMS text;
    the label is built here — by construction is_transaction=False and
    is_suspected_scam comes from the template tag."""
    prompt = NEG_PROMPT.format(template=template.shape, style=style, n=n)
    label = {"is_transaction": False, "is_suspected_scam": template.is_scam}
    ExpenseRecord.model_validate(label)
    examples = []
    for line in _call_teacher(prompt).splitlines():
        line = line.strip().strip("`").lstrip("-*• ").strip()
        if not _looks_like_sms(line):
            continue
        examples.append({"sms": fix_doubled_currency(line), "label": dict(label)})
    return examples


def generate_dataset(
    templates=TEMPLATES,
    per_cell: int = 3,
    neg_per_template: int = 15,
    seed: int = 42,
    skip=frozenset(),
    on_template=None,
):
    """Stratified loop over template × Category × TxnType with impossible cells
    pruned via template tags, so no cell of the grid is starved or nonsensical.

    `skip` holds template indices already generated (resume after a quota crash);
    their random.choice calls still run so the RNG stream — and therefore every
    later template's style picks — matches a fresh run. `on_template` is called
    with (idx, rows) after each template completes, for incremental checkpointing.
    """
    random.seed(seed)
    dataset = []
    for idx, t in enumerate(templates):
        rows = []
        if t.is_negative:
            style = random.choice(STYLES)
            if idx not in skip:
                rows = generate_negative_batch(t, style, n=neg_per_template)
        else:
            for category, txn_type in itertools.product(Category, TxnType):
                if t.hint_txn_type and t.hint_txn_type != txn_type.value:
                    continue  # impossible cell: shape's wording contradicts the direction
                if category is Category.salary and txn_type is not TxnType.credit:
                    continue  # salary only ever arrives as a credit
                style = random.choice(STYLES)
                if idx in skip:
                    continue
                rows += generate_batch(
                    t.shape, category.value, txn_type.value, style, n=per_cell
                )
        if idx in skip:
            print(f"template {idx}: skipped (resume)", flush=True)
            continue
        for ex in rows:
            ex["template_id"] = idx
        dataset += rows
        if on_template:
            on_template(idx, rows)
        print(f"template {idx}: +{len(rows)} (total {len(dataset)})", flush=True)
    return dataset


def _minhash(sms: str):
    from datasketch import MinHash  # gen extra; only dedupe's callers need it

    m = MinHash(num_perm=128)
    for word in sms.lower().split():
        m.update(word.encode())
    return m


def dedupe(examples, threshold=0.85, against=None):
    """Keep only examples whose word-shingles aren't ~identical to one already
    kept — nor to anything in `against` (e.g. the train pool, so test examples
    that near-match a training example don't become freebies)."""
    from datasketch import MinHashLSH  # gen extra; only dedupe needs it

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for j, ex in enumerate(against or []):
        lsh.insert(f"against-{j}", _minhash(ex["sms"]))
    kept = []
    for i, ex in enumerate(examples):
        m = _minhash(ex["sms"])
        if not lsh.query(m):  # nothing similar seen before
            lsh.insert(str(i), m)
            kept.append(ex)
    return kept


def _write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in rows)


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate synthetic labeled SMS via the Gemini teacher."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output JSONL path, e.g. data/generated/train.jsonl",
    )
    parser.add_argument(
        "--val-out",
        default=None,
        help="if set, split off a val set to this path after dedupe",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--per-cell",
        type=int,
        default=3,
        help="examples per template×category×type cell",
    )
    parser.add_argument(
        "--neg-per-template",
        type=int,
        default=15,
        help="examples per negative template",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-templates",
        type=int,
        default=None,
        help="only use the first N templates (smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip templates already checkpointed in <out>.raw",
    )
    args = parser.parse_args()

    # Checkpoint: each completed template is appended to <out>.raw immediately,
    # so a quota/network crash mid-run loses at most one template's calls.
    raw_path = args.out + ".raw"
    prior = []
    if args.resume and os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    prior.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # truncated final write from a crash — drop it
    skip = {ex["template_id"] for ex in prior}

    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    raw_f = open(raw_path, "w")  # noqa: SIM115 — checkpoint file, kept open across the generation loop, closed below
    raw_f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in prior)
    raw_f.flush()

    def checkpoint(idx, rows):
        raw_f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in rows)
        raw_f.flush()

    templates = TEMPLATES[: args.max_templates] if args.max_templates else TEMPLATES
    fresh = generate_dataset(
        templates,
        per_cell=args.per_cell,
        neg_per_template=args.neg_per_template,
        seed=args.seed,
        skip=skip,
        on_template=checkpoint,
    )
    raw_f.close()

    # Stable template order regardless of resume history, so dedupe keeps the
    # same winners a single-shot run would have kept.
    dataset = sorted(prior + fresh, key=lambda ex: ex["template_id"])
    deduped = dedupe(dataset)

    if args.val_out:
        rng = random.Random(args.seed)
        rng.shuffle(deduped)
        n_val = int(len(deduped) * args.val_frac)
        _write_jsonl(args.val_out, deduped[:n_val])
        _write_jsonl(args.out, deduped[n_val:])
        print(
            f"generated {len(dataset)}, kept {len(deduped)} after dedupe → "
            f"{len(deduped) - n_val} train ({args.out}) + {n_val} val ({args.val_out})"
        )
    else:
        _write_jsonl(args.out, deduped)
        print(
            f"generated {len(dataset)}, kept {len(deduped)} after dedupe → {args.out}"
        )


if __name__ == "__main__":
    main()
