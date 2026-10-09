"""OOD eval-set builder (tutorial 1.5) — fetch-at-eval-time, never committed.

The originally reserved fixture repos turned out not to commit their real test
data (see docs/decisions.md 2026-08-03), so the OOD sheet is built from the
licensed public scraps that do exist:

  - avinal's "Indian Bank SMS" gist (~30 real formats, values sanitized)
  - MabudAlam/transaction_sms_parser README + example strings (MIT)
  - saurabhgupta050890/transaction-sms-parser README string (MIT)

These are REAL sender formats with placeholder values ("Rs.xxx.xx"). Lowercase
`x` runs are the sanitized bits — they're filled with deterministic seeded
digits so amounts/refs are concrete enough to label. Uppercase `XX` runs are
genuine mask formatting (Card XX1234) and are kept verbatim. So: real formats,
synthetic values — an honest step down from a real inbox export (unavailable),
recorded as a limitation in docs/decisions.md.

Output rows carry "label": null and fail the 1.6 gate by construction —
hand-labeling is what turns this sheet into the OOD eval set. Never train on it.
"""

import json
import os
import random
import re
import subprocess
import urllib.request

from tinyllm.data_gen import dedupe

GIST_RAW = (
    "https://gist.githubusercontent.com/avinal/4079e1752e5b987530315b4802e51287/raw"
)
REPOS_DIR = "data/ood/repos"
REPOS = {
    "transaction_sms_parser": "https://github.com/MabudAlam/transaction_sms_parser",
    "transaction-sms-parser": "https://github.com/saurabhgupta050890/transaction-sms-parser",
}
SMS_HINT = re.compile(
    r"(debited|credited|withdrawn|spent|received|due by|sent to|Paid Rs)", re.IGNORECASE
)
QUOTED = re.compile(r"'([^']{40,300})'|\"([^\"]{40,300})\"")
LOWER_X_RUN = re.compile(r"x{2,}")  # sanitized digits; uppercase X = mask, kept


def fill_placeholders(sms: str, rng: random.Random) -> str:
    def digits(m):
        n = len(m.group(0))
        return str(rng.randint(1, 9)) + "".join(
            str(rng.randint(0, 9)) for _ in range(n - 1)
        )

    return LOWER_X_RUN.sub(digits, sms)


def from_gist() -> list[dict]:
    text = urllib.request.urlopen(GIST_RAW, timeout=30).read().decode("utf8")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") and SMS_HINT.search(line) and len(line) > 60:
            rows.append(
                {
                    "sms": line[2:].strip(),
                    "source": "gist:avinal/indian-bank-sms",
                    "license": "unlicensed-gist (local eval only, never commit)",
                }
            )
    return rows


def from_repos() -> list[dict]:
    rows = []
    for name, url in REPOS.items():
        path = os.path.join(REPOS_DIR, name)
        if not os.path.exists(path):
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", url, path], check=True
            )
        for rel in ("README.md", "example/example.dart"):
            fp = os.path.join(path, rel)
            if not os.path.exists(fp):
                continue
            with open(fp, encoding="utf8", errors="ignore") as fh:
                src = fh.read()
            for m in QUOTED.finditer(src):
                s = (m.group(1) or m.group(2)).strip()
                if SMS_HINT.search(s) and not s.endswith(("=", ";")):
                    rows.append(
                        {"sms": s, "source": f"github:{name}/{rel}", "license": "MIT"}
                    )
    return rows


def main():
    rng = random.Random(1005)  # fixed seed: sheet is reproducible
    rows = from_gist() + from_repos()
    for r in rows:
        r["sms"] = fill_placeholders(r["sms"], rng)
        r["label"] = None  # fails the 1.6 gate until hand-labeled
    kept = dedupe(rows)
    out = "data/ood/ood_unlabeled.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} scraped, {len(kept)} after dedupe → {out}")
    print(
        "NEXT: hand-label every row (fill 'label'), save as data/ood/ood.jsonl, "
        "then run the 1.6 gate on it. Never train on this file."
    )


if __name__ == "__main__":
    main()
