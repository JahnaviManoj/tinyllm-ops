"""Zero-shot / few-shot baselines (tutorial 2.5) — the "before" numbers.

The ladder this file provides rungs for:
    regex → zero-shot 270M → few-shot 270M → few-shot BIG model → fine-tuned 270M

- zero-270m / few-270m: google/gemma-3-270m-it runs locally (fits a 4GB GPU).
- few-big: Gemma-4-26B via the Gemini API — ~96x the student's size, and a
  different model family from the test-set teacher (gemini-3.5-flash), so the
  big baseline is not grading its own homework.

Few-shot examples come from TRAIN data only — never from the gate set being
measured. Every run logs params + metrics to MLflow (local ./mlruns until 2.8
points tracking at Azure ML).

CLI: uv run python -m tinyllm.baselines --mode zero-270m --data data/generated/test_gate.jsonl
     modes: zero-270m | few-270m | few-big
"""

import argparse
import json

from tinyllm.eval import evaluate

LOCAL_MODEL_ID = "google/gemma-3-270m-it"
BIG_MODEL_ID = "gemma-4-26b-a4b-it"  # via Gemini API; NOT the test-set teacher family

PROMPT = """Parse this bank SMS into JSON with fields: is_transaction, txn_type, amount,
currency, counterparty, account_tail, channel, category, is_suspected_scam.
Respond with ONLY the JSON.
{examples}SMS: {sms}
JSON:"""

# Few-shot examples are drawn deterministically from train.jsonl (never the
# eval set): a UPI debit, a salary credit, an OTP negative, and a scam — the
# four behaviors the schema most needs demonstrated.
FEW_SHOT_SPEC = [
    ("debit", "UPI"),
    ("credit", None),
    (None, None),  # non-transaction, non-scam
    ("scam", None),
]


def few_shot_block(train_path: str = "data/generated/train.jsonl") -> str:
    rows = [json.loads(line) for line in open(train_path)]
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


def extract_json(text: str) -> str:
    """First balanced {...} block — models love fences and preambles; being
    unable to find ANY object still counts as a parse failure downstream."""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return text[start : i + 1]
    return text[start:].strip()


def predict_local(sms_list: list[str], examples: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float32, NOT float16: Gemma's activations overflow fp16 (all-pad output,
    # NaN logits) and pre-Ampere GPUs lack bf16. 270M in fp32 is ~1.1GB — fine.
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID, dtype=torch.float32
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


def predict_api(sms_list: list[str], examples: str) -> list[str]:
    from tinyllm.data_gen import _call_teacher

    outs = []
    for i, sms in enumerate(sms_list):
        raw = _call_teacher(
            PROMPT.format(examples=examples, sms=sms), model=BIG_MODEL_ID
        )
        outs.append(extract_json(raw))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sms_list)}", flush=True)
    return outs


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run a 2.5 baseline on a labeled JSONL file."
    )
    parser.add_argument(
        "--mode", required=True, choices=["zero-270m", "few-270m", "few-big"]
    )
    parser.add_argument("--data", default="data/generated/test_gate.jsonl")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N rows (smoke tests)"
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    sms_list = [r["sms"] for r in rows]
    examples = "" if args.mode == "zero-270m" else few_shot_block()
    model_id = BIG_MODEL_ID if args.mode == "few-big" else LOCAL_MODEL_ID
    predict = predict_api if args.mode == "few-big" else predict_local

    preds = predict(sms_list, examples)
    report = evaluate(preds, [r["label"] for r in rows])

    import mlflow

    mlflow.set_experiment("baselines")
    with mlflow.start_run(run_name=args.mode):
        mlflow.log_params(
            {
                "mode": args.mode,
                "model": model_id,
                "data": args.data,
                "n": len(rows),
                "few_shot": args.mode != "zero-270m",
            }
        )
        for k, v in report.items():
            if isinstance(v, dict):
                mlflow.log_metrics({f"field_acc_{f}": acc for f, acc in v.items()})
            else:
                mlflow.log_metric(k, v)
    print(f"== {args.mode} on {args.data} ({len(rows)} rows)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
