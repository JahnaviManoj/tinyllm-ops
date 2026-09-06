"""Evaluate a merged fine-tuned model on a labeled JSONL file (tutorial §2.4).

Used for winner selection on the gate set, the once-only frozen-final/OOD/
scam-holdout eval, and later the Stage 3 promotion gate. Prompts come from
tinyllm.prompt — the same bytes the model saw in training; using any other
prompt here would measure train/serve skew, not the model.

--chat      chat-template the prompt (Qwen, Era 2). MUST match how the model
            was trained: --chat for exp_1xx, no flag for the Era-1 270M runs.
--rows-out  per-row correctness JSON for scripts/mcnemar.py (the tie rule).

CLI: uv run python -m tinyllm.model_eval --model outputs/exp_101/merged --chat \
        --data data/generated/test_gate_v2.jsonl --run-name gate2-exp_101 \
        --rows-out outputs/exp_101/gate2_rows.json
"""

import argparse
import json
import os

from tinyllm.baselines import extract_json
from tinyllm.eval import evaluate, row_correct
from tinyllm.prompt import build_prompt, chat_prompt


def predict_merged(
    model_dir: str, sms_list: list[str], chat: bool = False
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(
        device
    )
    model.eval()
    outs = []
    for i, sms in enumerate(sms_list):
        # Identical bytes to training. chat: the template string already
        # carries every special token, so add none (TRL tokenized the training
        # prompt the same way). raw: era-1 behaviour, unchanged.
        text_in = chat_prompt(tok, sms) if chat else build_prompt(sms)
        inputs = tok(text_in, return_tensors="pt", add_special_tokens=not chat).to(
            device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        outs.append(extract_json(text))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sms_list)}", flush=True)
    return outs


def eval_model(
    model_dir: str,
    data_path: str,
    run_name: str | None = None,
    experiment: str = "tinyllm-finetune",
    chat: bool = False,
    rows_out: str | None = None,
) -> dict:
    rows = [json.loads(line) for line in open(data_path)]
    preds = predict_merged(model_dir, [r["sms"] for r in rows], chat=chat)
    report = evaluate(preds, [r["label"] for r in rows])
    if rows_out:  # per-row verdicts — what McNemar pairs up
        per = [
            {"sms": r["sms"], "correct": row_correct(p, r["label"])}
            for p, r in zip(preds, rows)
        ]
        os.makedirs(os.path.dirname(rows_out) or ".", exist_ok=True)
        json.dump(per, open(rows_out, "w"))
    if run_name:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model_dir": model_dir,
                    "data": data_path,
                    "n": len(rows),
                    "chat": chat,
                }
            )
            for k, v in report.items():
                if isinstance(v, dict):
                    mlflow.log_metrics({f"field_acc_{f}": acc for f, acc in v.items()})
                else:
                    mlflow.log_metric(k, v)
            if rows_out:
                mlflow.log_artifact(rows_out)  # survives the Colab VM
    print(f"== {model_dir} on {data_path} ({len(rows)} rows, chat={chat})")
    print(json.dumps(report, indent=2))
    return report


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate a merged model on labeled JSONL."
    )
    parser.add_argument(
        "--model", required=True, help="merged model dir, e.g. outputs/exp_101/merged"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--run-name", default=None, help="log to MLflow under this run name"
    )
    parser.add_argument("--experiment", default="tinyllm-finetune")
    parser.add_argument(
        "--chat", action="store_true", help="chat-template prompts (Qwen models)"
    )
    parser.add_argument(
        "--rows-out", default=None, help="write per-row exact-match JSON (for McNemar)"
    )
    args = parser.parse_args()
    eval_model(
        args.model,
        args.data,
        run_name=args.run_name,
        experiment=args.experiment,
        chat=args.chat,
        rows_out=args.rows_out,
    )


if __name__ == "__main__":
    main()
