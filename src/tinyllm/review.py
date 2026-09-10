"""Hand-review a generated eval set, row by row. AI audits propose; YOU verdict.

CLI: uv run python -m tinyllm.review --in data/generated/gate2_unreviewed.jsonl \\
         --out data/generated/test_v2.jsonl [--flags data/generated/gate2_flags.json]
Keys: [enter]=accept  e=edit a field  d=drop row  q=quit (checkpointed, rerun resumes)

Labels validate on every edit; field names are checked against the schema (Pydantic
ignores unknown keys, so a typo like txn_typ would otherwise pass silently); drops are
checkpointed to <out>.dropped so a rerun never shows a decided row twice.
"""

import argparse
import json
import os

from pydantic import ValidationError

from tinyllm.schema import ExpenseRecord

FIELDS = set(ExpenseRecord.model_fields)
PROMPT = "[enter]=ok  e=edit  d=drop  q=quit > "


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _parse(raw: str):
    """JSON if it parses; otherwise a bare word is a string, null/none → None."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        low = raw.lower()
        if low in ("null", "none", ""):
            return None
        if low in ("true", "false"):
            return low == "true"
        return raw


def _edit(label: dict) -> None:
    field = input("field: ").strip()
    if field not in FIELDS:
        print(f"unknown field (schema has: {sorted(FIELDS)})")
        return
    value = _parse(input("value (food / 450.00 / false / null): "))
    trial = {**label, field: value}
    if trial.get("is_transaction"):
        try:
            ExpenseRecord.model_validate(trial)
        except ValidationError as e:
            err = e.errors()[0]
            print(f"rejected — {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
            return
    label[field] = value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--flags", default=None, help="{row_index: note} JSON from AI audit passes"
    )
    a = p.parse_args()
    rows = _load(a.inp)
    flags = {}
    if a.flags:
        with open(a.flags) as f:
            flags = json.load(f)
    drop_path = a.out + ".dropped"
    seen = {
        r["sms"]
        for path in (a.out, drop_path)
        if os.path.exists(path)
        for r in _load(path)
    }
    with open(a.out, "a") as out, open(drop_path, "a") as dropped:
        try:
            _review(rows, seen, flags, out, dropped)
        except (KeyboardInterrupt, EOFError):
            print("\ninterrupted — progress is saved, rerun to resume")
    print("reviewed →", a.out)


def _review(rows, seen, flags, out, dropped) -> None:
    for i, r in enumerate(rows):
        if r["sms"] in seen:
            continue
        flag = flags.get(str(i))
        print(f"\n--- {i + 1}/{len(rows)} " + (f"⚑ {flag}" if flag else ""))
        print("SMS:", r["sms"])
        print("LBL:", json.dumps(r["label"], ensure_ascii=False))
        cmd = input(PROMPT).strip().lower()
        while cmd == "e":
            _edit(r["label"])
            print("LBL:", json.dumps(r["label"], ensure_ascii=False))
            cmd = input(PROMPT).strip().lower()
        if cmd == "q":
            break
        (dropped if cmd == "d" else out).write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        dropped.flush()


if __name__ == "__main__":
    main()
