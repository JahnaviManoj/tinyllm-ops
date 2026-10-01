"""4.4 step 6: the rulebook guard's effect on the gate, from a rows file — no model.

The rows file (model_eval --rows-out) holds the raw prediction per row in gate
order; the gate JSONL holds the labels. Score as-is, then with tinyllm.rulebook
applied to each parsed prediction (what the gateway does with RULEBOOK=1).

  uv run python scripts/rulebook_delta.py outputs/exp_111/gguf/gate2_rows_Q8_0.json \
      data/generated/test_gate_v2.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinyllm.eval import evaluate, row_correct
from tinyllm.rulebook import apply_rulebook
from tinyllm.schema import ExpenseRecord


def with_rulebook(sms: str, pred: str) -> tuple[str, bool]:
    try:
        rec = ExpenseRecord.model_validate_json(pred)
    except ValueError:  # pydantic ValidationError / bad JSON → unparseable, unchanged
        return pred, False
    ex, hit = apply_rulebook({"sms": sms, "label": rec.model_dump(mode="json")})
    return json.dumps(ex["label"]), hit


def main(rows_path: str, gate_path: str) -> None:
    with open(rows_path) as f:
        rows = json.load(f)
    with open(gate_path) as f:
        gate = [json.loads(line) for line in f]
    assert len(rows) == len(gate) and all(
        r["sms"] == g["sms"] for r, g in zip(rows, gate)
    )
    gold = [g["label"] for g in gate]
    base = evaluate([r["pred"] for r in rows], gold)
    guarded, hits = zip(*(with_rulebook(r["sms"], r["pred"]) for r in rows))
    after = evaluate(list(guarded), gold)
    n = len(gold)
    before_ok = [row_correct(r["pred"], g) for r, g in zip(rows, gold)]
    after_ok = [row_correct(p, g) for p, g in zip(guarded, gold)]
    fixed = sum(a and not b for a, b in zip(after_ok, before_ok))
    broken = sum(b and not a for a, b in zip(after_ok, before_ok))
    print(
        f"{os.path.basename(rows_path)}  n={n}  rulebook fired on {sum(hits)} rows: "
        f"fixed {fixed}, broke {broken}, net {fixed - broken:+d}"
    )
    for name, rep in (("as-is", base), ("rulebook", after)):
        print(
            f"  {name:9s} exact {rep['exact_match']:.4f} = {round(rep['exact_match'] * n)}/{n}"
            f"  category {rep['per_field_accuracy']['category']:.4f}"
            f"  macro-F1 {rep['category_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
