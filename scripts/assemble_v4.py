"""train_v4 = train_v3 + scam + ham candidates, rulebook-corrected; val_v3 carved from v4
(TUTORIAL_V2 C7).

Leak guards, in order: new candidates are near-dedup'd (MinHash 0.85) against train_v3 AND
against every eval set — gate_v2, final_v2 (fetched into the filter only, then deleted:
the frozen-final rule), OOD, and the real-scam holdout. The rulebook then post-corrects
teacher categories on the whole pool (counterparty-only matching; see rulebook.py), and
val_v3 is a seeded 10% slice of the corrected pool so it sees the v4 distribution
including scams.

    uv run python scripts/assemble_v4.py
"""

import json
import os
import random

from dotenv import load_dotenv

from tinyllm.data_gen import dedupe
from tinyllm.manifest import fetch_dataset
from tinyllm.rulebook import apply_rulebook

TRAIN_OUT = "data/generated/train_v4.jsonl"
VAL_OUT = "data/generated/val_v3.jsonl"
FINAL_TMP = "data/generated/.final_v2.dedupe-only.jsonl"


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    load_dotenv()  # AZURE_STORAGE_CONNECTION_STRING for the final_v2 fetch
    train = load("data/generated/train_v3.jsonl")
    new = (
        load("data/scam/mendeley_scam.jsonl")
        + load("data/scam/scenario_scam.jsonl")
        + load("data/scam/mendeley_ham.jsonl")
    )
    fetch_dataset("manifests/test_final_v2.json", FINAL_TMP)  # fetch-into-filter-only
    guard = (
        load("data/generated/test_gate_v2.jsonl")
        + load(FINAL_TMP)
        + load("data/ood/ood.jsonl")
        + load("data/scam/scam_holdout.jsonl")
    )
    os.remove(FINAL_TMP)

    before = len(new)
    new = dedupe(new, against=train + guard)
    pool, corrected = [], 0
    for ex in train + new:
        ex, hit = apply_rulebook(ex)
        corrected += hit
        pool.append(ex)

    rng = random.Random(1007)
    rng.shuffle(pool)
    n_val = int(0.10 * len(pool))
    val, tr = pool[:n_val], pool[n_val:]
    for path, data in [(TRAIN_OUT, tr), (VAL_OUT, val)]:
        with open(path, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    scam = sum(1 for r in tr if r["label"].get("is_suspected_scam"))
    print(
        f"train_v4 {len(tr)} (scam {100 * scam / len(tr):.1f}%) · val_v3 {len(val)} · "
        f"rulebook corrections {corrected} · new rows kept {len(new)} of {before}"
    )


if __name__ == "__main__":
    main()
