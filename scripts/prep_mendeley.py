"""Mendeley corpus → real-scam EVAL HOLDOUT + scam/ham train candidates (TUTORIAL_V2 C2).

Dataset_5971.csv (Mishra & Soni, https://data.mendeley.com/datasets/f45bkkt8pr):
LABEL ∈ {ham, spam, Smishing}, TEXT, URL, EMAIL, PHONE. Candidates still pass dedupe +
validation before touching train. The finance filter is word-bounded (an unanchored
won|pan|a/?c matched "wonder", "pants", "beach"); ~60 smishing rows are held out as a
real-scam eval slice BEFORE any training candidate is written; rows carry
source: mendeley-uk so the sender-ID augmentation skips them.

Encoding: the file is valid UTF-8 (genuine "£" and curly quotes throughout), but 145
rows already contain U+FFFD from the authors' own conversion. Reading it as latin-1 (the
tutorial's first guess) would mojibake every "£" into "Â£"; reading as UTF-8 and dropping
the rows that carry U+FFFD keeps the clean ones intact. Labels come in case variants
(Smishing/smishing, spam/Spam) — hence .lower().

    uv run python scripts/prep_mendeley.py data/scam/raw/Dataset_5971.csv
"""

import csv
import hashlib
import json
import random
import re
import sys

FIN = re.compile(
    r"\b(bank|a/c|acc(?:oun)?t|card|debit|credit|upi|kyc|paytm|netbanking|loan|"
    r"reward|refund|prize|won|winner|claim|blocked|suspend(?:ed)?|verify|pan|otp)\b",
    re.IGNORECASE,
)


# Reviewed 2026-09-06 (C2 eyeball): ex-'spam' rows that bait with money/prizes contradict
# is_suspected_scam: false, so they are dropped from the ham candidates. Keyed by sha1 of
# the text so the result is reproducible from the raw CSV; the script fails if one stops
# matching. Applied AFTER the 300-row slice so no unreviewed row slips in to replace them.
EXCLUDE_HAM = {
    "bd60ab261257",  # 'Ur cash-balance is currently 500 pounds - to maximize ur c'
    "9abfffdd3f99",  # 'Ur cash-balance is currently 500 pounds - to maximize ur c'
    "91e1be88e6ce",  # 'Free entry to the gr8prizes wkly comp 4 a chance to win th'
    "963c1d8508c0",  # 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st '
    "cd40786ddffd",  # 'PRIVATE! Your 2003 Account Statement for 078'
    "7d0da56dd99d",  # 'Your free ringtone is waiting to be collected. Simply text'
}


def ok(s: str) -> bool:  # v3 junk guard + corrupted-byte guard
    return 15 <= len(s) <= 320 and not re.search(r"(.)\1{9,}", s) and "�" not in s


def row(text: str, scam: bool) -> dict:
    return {
        "sms": text,
        "source": "mendeley-uk",
        "label": {"is_transaction": False, "is_suspected_scam": scam},
    }


with open(sys.argv[1], encoding="utf-8", errors="replace", newline="") as f:
    rows = list(csv.DictReader(f))
rng = random.Random(1006)

scam = [
    row(r["TEXT"].strip(), True)
    for r in rows
    if r["LABEL"].lower() == "smishing"
    and FIN.search(r["TEXT"])
    and ok(r["TEXT"].strip())
]
ham = [
    row(r["TEXT"].strip(), False)
    for r in rows
    if r["LABEL"].lower() in ("ham", "spam") and ok(r["TEXT"].strip())
]

rng.shuffle(scam)
rng.shuffle(ham)
holdout, scam_train = scam[:60], scam[60:]  # eval slice FIRST — never trained
ham_300 = ham[:300]
hit = {hashlib.sha1(r["sms"].encode()).hexdigest()[:12] for r in ham_300} & EXCLUDE_HAM
if hit != EXCLUDE_HAM:
    raise SystemExit(
        f"EXCLUDE_HAM entries not found in the sample: {EXCLUDE_HAM - hit}"
    )
ham_kept = [
    r
    for r in ham_300
    if hashlib.sha1(r["sms"].encode()).hexdigest()[:12] not in EXCLUDE_HAM
]
for path, data in [
    ("data/scam/scam_holdout.jsonl", holdout),
    ("data/scam/mendeley_scam.jsonl", scam_train),
    ("data/scam/mendeley_ham.jsonl", ham_kept),
]:
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    print(path, len(data))
