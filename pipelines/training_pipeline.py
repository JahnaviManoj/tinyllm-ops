"""Stage 3: data → validate → train → gate → register, as one ZenML DAG.

Steps are thin. Every piece of logic is imported from src/tinyllm so it stays
testable (and usable on Colab) without zenml.
"""

import os
from typing import Annotated

from zenml import pipeline, step

from tinyllm.config import load_config
from tinyllm.manifest import fetch_local, fetch_rows
from tinyllm.model_eval import eval_model, run_behaviors
from tinyllm.registry import register_as_challenger
from tinyllm.train import train
from tinyllm.validate import validate_dataset

# final_v2, OOD and the scam holdout are spent — never in a pipeline.
GATE_MANIFEST = "manifests/test_gate_v2.json"


@step
def load_data(manifest_path: str) -> list[dict]:
    return fetch_rows(manifest_path)


@step
def validate(examples: list[dict]) -> list[dict]:
    try:
        validate_dataset(examples)
    except (
        SystemExit
    ) as e:  # validate.py is CLI-first; turn its exit into a step failure
        raise RuntimeError(f"Data gate FAILED: {e}") from e
    return examples


@step
def train_step(
    examples: list[dict],  # unused on purpose: this input is the DAG edge from validate
    config_path: str,
    max_steps: int | None,
    limit: int | None,
) -> tuple[Annotated[str, "run_id"], Annotated[str, "merged_dir"]]:
    run_id = train(config_path, max_steps=max_steps, limit=limit, log_merged=True)
    exp = os.path.splitext(os.path.basename(config_path))[0]
    return run_id, os.path.join("outputs", exp, "merged")


@step
def evaluate_step(
    merged_dir: str, config_path: str, threshold: float, strict_behaviors: bool = True
) -> dict:
    chat = load_config(config_path).chat_format
    report = eval_model(merged_dir, fetch_local(GATE_MANIFEST), chat=chat)
    if report["exact_match"] < threshold:  # gate 1: how much is it right
        raise RuntimeError(
            f"Eval gate FAILED: {report['exact_match']:.3f} < {threshold}"
        )
    report["behaviors"] = run_behaviors(merged_dir, chat=chat)
    failing = [
        b for b, ok in report["behaviors"].items() if not ok
    ]  # gate 2: in what ways
    if (
        failing and strict_behaviors
    ):  # smoke runs record behaviors but don't enforce them
        raise RuntimeError(f"Behavioral gate FAILED: {failing}")
    return report


@step
def register_step(run_id: str, report: dict) -> None:
    register_as_challenger(run_id, report)  # see 3.3


@pipeline
def training_pipeline(
    config_path: str,
    threshold: float,
    max_steps: int | None = None,
    limit: int | None = None,
    strict_behaviors: bool = True,
):
    manifest_path = load_config(config_path).dataset_manifest  # one source of truth
    data = load_data(manifest_path)
    data = validate(data)
    run_id, merged_dir = train_step(data, config_path, max_steps, limit)
    report = evaluate_step(merged_dir, config_path, threshold, strict_behaviors)
    register_step(run_id, report)
