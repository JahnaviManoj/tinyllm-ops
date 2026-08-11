"""Evaluate a merged fine-tuned model on a labeled JSONL file (tutorial 2.9).

Used for winner selection on the gate set, the once-only frozen-final/OOD
eval, and later the Stage 3 promotion gate. Prompts come from tinyllm.prompt —
the same bytes the model saw in training; using any other prompt here would
measure train/serve skew, not the model.

CLI: uv run python -m tinyllm.model_eval --model outputs/exp_001/merged \
        --data data/generated/test_gate.jsonl --run-name gate-exp_001
"""

import argparse
import json

from tinyllm.baselines import extract_json
from tinyllm.eval import evaluate
from tinyllm.prompt import build_prompt


def predict_merged(model_dir: str, sms_list: list[str]) -> list[str]:
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
        # Raw completion prompt — identical bytes to training; NO chat template,
        # because training didn't use one either (train/serve consistency).
        inputs = tok(build_prompt(sms), return_tensors="pt").to(device)
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
) -> dict:
    rows = [json.loads(line) for line in open(data_path)]
    preds = predict_merged(model_dir, [r["sms"] for r in rows])
    report = evaluate(preds, [r["label"] for r in rows])
    if run_name:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {"model_dir": model_dir, "data": data_path, "n": len(rows)}
            )
            for k, v in report.items():
                if isinstance(v, dict):
                    mlflow.log_metrics({f"field_acc_{f}": acc for f, acc in v.items()})
                else:
                    mlflow.log_metric(k, v)
    print(f"== {model_dir} on {data_path} ({len(rows)} rows)")
    print(json.dumps(report, indent=2))
    return report


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate a merged model on labeled JSONL."
    )
    parser.add_argument(
        "--model", required=True, help="merged model dir, e.g. outputs/exp_001/merged"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--run-name", default=None, help="log to MLflow under this run name"
    )
    parser.add_argument("--experiment", default="tinyllm-finetune")
    args = parser.parse_args()
    eval_model(
        args.model, args.data, run_name=args.run_name, experiment=args.experiment
    )


if __name__ == "__main__":
    main()
