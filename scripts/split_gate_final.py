"""test_v2 → test_gate_v2 (2/3) + test_final_v2 (1/3), stratified, seed 24 (TUTORIAL_V2 C6).

The gate takes dozens of decisions across Stage 2; the final takes exactly one (the
once-only ritual). Strata = (is_transaction, is_suspected_scam, channel, txn_type) so both
halves carry the same mix of channels, directions and negatives.

    uv run python scripts/split_gate_final.py
"""

import json
import random
from collections import defaultdict

SRC = "data/generated/test_v2.jsonl"
GATE = "data/generated/test_gate_v2.jsonl"
FINAL = "data/generated/test_final_v2.jsonl"


def main() -> None:
    with open(SRC) as f:
        rows = [json.loads(line) for line in f]
    strata = defaultdict(list)
    for r in rows:
        lbl = r["label"]
        key = (
            lbl.get("is_transaction"),
            lbl.get("is_suspected_scam"),
            lbl.get("channel"),
            lbl.get("txn_type"),
        )
        strata[key].append(r)
    rng, gate, final = random.Random(24), [], []
    for key in sorted(strata, key=str):
        grp = strata[key]
        rng.shuffle(grp)
        k = round(len(grp) * 2 / 3)
        gate += grp[:k]
        final += grp[k:]
    for path, data in [(GATE, gate), (FINAL, final)]:
        with open(path, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    print(f"gate_v2 {len(gate)} rows · final_v2 {len(final)} rows")


if __name__ == "__main__":
    main()
