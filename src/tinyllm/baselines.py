"""Zero-shot / few-shot baselines (tutorial §2.6) — the "before" numbers.

The baseline ladder this file provides rungs for:
    regex → zero-shot local → few-shot local → few-shot BIG → fine-tuned

- zero-local / few-local: any HF instruct model, pinned by --local-model /
  --local-revision (Qwen3.5-2B and -0.8B).
  predict_local applies the tokenizer's chat template itself — there is no
  --chat flag here because the base instruct model has never seen our
  training prompt; the baseline IS "what the model does with a chat request".
- few-big: Gemma-4-26B via the Gemini API — a different model family from
  the test-set teacher (gemini-3.5-flash), so the big baseline is not grading
  its own homework.

Few-shot exemplars come from --train-pool (train data only — never from the
gate being measured), chosen deterministically: first row matching each
FEW_SHOT_SPEC kind. Pre-register the pool + this policy before running.
Every run logs params + metrics to MLflow (MLFLOW_TRACKING_URI).

CLI: uv run python -m tinyllm.baselines --mode few-local \
        --local-model Qwen/Qwen3.5-0.8B --local-revision <sha> \
        --data data/generated/test_gate_v2.jsonl
     modes: zero-local | few-local | few-big
"""

import argparse
import json

from tinyllm.eval import evaluate
from tinyllm.extract import (
    extract_json,
)

BIG_MODEL_ID = "gemma-4-26b-a4b-it"  # via Gemini API; NOT the test-set teacher family

PROMPT = """Parse this bank SMS into JSON with fields: is_transaction, txn_type, amount,
currency, counterparty, account_tail, channel, category, is_suspected_scam.
Respond with ONLY the JSON.
{examples}SMS: {sms}
JSON:"""

# Sensitivity variant (NOT the pre-registered bar; reported alongside it, 2026-09-13): the
# pre-registered prompt names the fields but never lists the allowed values, so a capable
# model answers "channel": "Debit Card" / "category": "dividend" and fails schema validation
# (26B few-shot: 34% schema-valid on gate_v2, near-perfect when valid). This variant tells
# the model the enums and the amount format — a STRONGER competitor, which can only make our
# claim harder. The champion title rests on the pre-registered prompt; this is for honesty.
SCHEMA_HINT = """Rules: amount is a decimal string with 2 places ("450.00"); currency "INR";
txn_type is "debit" or "credit"; channel is one of UPI, card, netbanking, ATM, wallet (or null);
category is one of food, groceries, transport, shopping, bills_utilities, entertainment, health,
education, travel, rent, salary, transfer, investment, fees, other (or null);
if no money moved (OTP, reminder, promo, scam) set is_transaction false and every other field null
except is_suspected_scam.
"""

# Few-shot examples are drawn deterministically from the train pool (never the
# eval set): a UPI debit, a salary credit, an OTP negative, and a scam — the
# four behaviors the schema most needs demonstrated.
FEW_SHOT_SPEC = [
    ("debit", "UPI"),
    ("credit", None),
    (None, None),  # non-transaction, non-scam
    ("scam", None),
]


def few_shot_block(train_path: str) -> str:
    with open(train_path) as f:
        rows = [json.loads(line) for line in f]
    picked = []
    for kind, channel in FEW_SHOT_SPEC:
        for r in rows:
            lb = r["label"]
            ok = (
                (kind == "scam" and lb.get("is_suspected_scam"))
                or (
                    kind is None
                    and not lb["is_transaction"]
                    and not lb.get("is_suspected_scam")
                )
                or (
                    kind in ("debit", "credit")
                    and lb.get("txn_type") == kind
                    and (channel is None or lb.get("channel") == channel)
                    and (kind != "credit" or lb.get("category") == "salary")
                )
            )
            if ok and r not in picked and len(r["sms"]) < 200:
                picked.append(r)
                break
    return "".join(
        f"SMS: {r['sms']}\nJSON: {json.dumps(r['label'], ensure_ascii=False)}\n\n"
        for r in picked
    )


def predict_local(
    sms_list: list[str], examples: str, model_id: str, revision: str | None = None
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float32, NOT float16: small models' activations can overflow fp16 and pre-Ampere
    # GPUs lack bf16. 0.8B ≈ 3.2 GB (fits a 4 GB card, barely),
    # 2B ≈ 8 GB → Colab T4 only.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32
    ).to(device)
    outs = []
    for i, sms in enumerate(sms_list):
        msgs = [{"role": "user", "content": PROMPT.format(examples=examples, sms=sms)}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        outs.append(
            extract_json(
                tok.decode(
                    out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sms_list)}", flush=True)
    return outs


def predict_api(
    sms_list: list[str], examples: str, workers: int = 1, schema_hint: bool = False
) -> list[str]:
    """One API call per SMS, in order. `workers` > 1 overlaps calls (the 26B takes
    17–30 s each; 356 rows sequentially is ~2.3 h, with 6 in flight ~20 min). Output
    order and content are unchanged — only throughput."""
    from concurrent.futures import ThreadPoolExecutor

    from tinyllm.data_gen import _call_teacher, get_client

    get_client()  # build the shared client in the main thread before any worker starts

    prompt = (
        PROMPT.replace(
            "Respond with ONLY the JSON.", SCHEMA_HINT + "Respond with ONLY the JSON."
        )
        if schema_hint
        else PROMPT
    )

    def one(sms: str) -> str:
        return extract_json(
            _call_teacher(prompt.format(examples=examples, sms=sms), model=BIG_MODEL_ID)
        )

    outs: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, out in enumerate(pool.map(one, sms_list)):
            outs.append(out)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(sms_list)}", flush=True)
    return outs


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run a ladder baseline on a labeled JSONL file."
    )
    parser.add_argument(
        "--mode", required=True, choices=["zero-local", "few-local", "few-big"]
    )
    parser.add_argument("--local-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--local-revision", default=None)
    parser.add_argument(
        "--train-pool",
        default="data/generated/train_v4.jsonl",
        help="few-shot exemplar source (era-2 pool, not the stale v1 default)",
    )
    parser.add_argument("--data", default="data/generated/test_gate_v2.jsonl")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N rows (smoke tests)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="concurrent API calls (few-big only)"
    )
    parser.add_argument(
        "--schema-hint",
        action="store_true",
        help="few-big sensitivity variant: tell the model the enums (NOT the pre-registered bar)",
    )
    args = parser.parse_args()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f][: args.limit]
    sms_list = [r["sms"] for r in rows]
    examples = "" if args.mode == "zero-local" else few_shot_block(args.train_pool)
    model_id = BIG_MODEL_ID if args.mode == "few-big" else args.local_model
    if args.mode == "few-big":
        preds = predict_api(
            sms_list, examples, workers=args.workers, schema_hint=args.schema_hint
        )
    else:
        preds = predict_local(sms_list, examples, args.local_model, args.local_revision)
    report = evaluate(preds, [r["label"] for r in rows])

    import mlflow

    mlflow.set_experiment("baselines")
    suffix = "-schema-hint" if args.schema_hint else ""
    with mlflow.start_run(run_name=f"{args.mode}-{model_id.split('/')[-1]}{suffix}"):
        mlflow.log_params(
            {
                "mode": args.mode,
                "model": model_id,
                "revision": args.local_revision,
                "train_pool": args.train_pool if args.mode != "zero-local" else None,
                "data": args.data,
                "n": len(rows),
                "few_shot": args.mode != "zero-local",
                "schema_hint": args.schema_hint,
            }
        )
        for k, v in report.items():
            if isinstance(v, dict):
                mlflow.log_metrics({f"field_acc_{f}": acc for f, acc in v.items()})
            else:
                mlflow.log_metric(k, v)
    print(f"== {args.mode} ({model_id}) on {args.data} ({len(rows)} rows)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
