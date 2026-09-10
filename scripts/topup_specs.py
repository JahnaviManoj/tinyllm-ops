"""Top up under-yielding test_gen specs (TUTORIAL_V2 C4).

test_gen asks each spec for n rows and silently drops lines whose JSON label fails schema
validation, so some specs land well short (gate_v2 first pass: ATM × credit kept 10/36).
This re-asks ONLY the named specs, appends the new rows to <out>.raw with the same spec_id,
and reports why lines were dropped. Then `python -m tinyllm.test_gen --out <out> --resume`
skips every spec (all are checkpointed) and re-runs the within-set + pool dedupe.
Done BEFORE review, so it adds rows, never selects them.

    uv run python scripts/topup_specs.py --out data/generated/gate2_unreviewed.jsonl \
        --specs 0,1,2,7,8,9 --ask 18,32,28,36,20,18
"""

import argparse
import json

from dotenv import load_dotenv

from tinyllm.test_gen import _specs, generate_spec


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", required=True, help="comma-separated spec ids")
    ap.add_argument("--ask", required=True, help="rows to request per spec, same order")
    a = ap.parse_args()
    ids = [int(x) for x in a.specs.split(",")]
    asks = [int(x) for x in a.ask.split(",")]
    specs = _specs()
    with open(a.out + ".raw", "a") as raw_f:
        for spec_id, n in zip(ids, asks, strict=True):
            spec = {**specs[spec_id], "n": n}
            rows = generate_spec(spec)
            for ex in rows:
                ex["spec_id"] = spec_id
                raw_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            raw_f.flush()
            what = (
                f"{spec['channel']} × {spec['txn_type']}"
                if spec["kind"] == "pos"
                else spec["desc"][:40]
            )
            print(f"spec {spec_id} ({what}): asked {n}, kept {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
