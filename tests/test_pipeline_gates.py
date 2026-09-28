"""The pipeline's gates (tutorial 3.6), with fake scores — no model is loaded.

A bug in a gate ships garbage silently, so each one is called directly via the
step's ``.entrypoint`` and shown to block what it must: bad exact-match, one
broken behaviour, and invalid data. Smoke mode must record but not enforce.
"""

import json

import pytest

from pipelines import training_pipeline as tp
from tinyllm.registry import GATE_MANIFEST

SMOKE_CONFIG = "configs/exp_100_smoke.yaml"
GOOD_BEHAVIORS = {
    "amount_tracks": True,
    "merchant_swap_keeps_category": True,
    "promo_noise_ignored": True,
    "amount_format_invariant": True,
}


@pytest.fixture
def fake_scores(monkeypatch):
    """Set what eval_model / run_behaviors return, without a model."""
    state = {"exact_match": 0.30, "behaviors": dict(GOOD_BEHAVIORS)}
    monkeypatch.setattr(
        tp,
        "eval_model",
        lambda *a, **k: {"exact_match": state["exact_match"], "per_field_accuracy": {}},
    )
    monkeypatch.setattr(tp, "run_behaviors", lambda *a, **k: dict(state["behaviors"]))
    monkeypatch.setattr(tp, "fetch_local", lambda *a, **k: "unused.jsonl")
    return state


def evaluate(threshold=0.595, strict_behaviors=True):
    return tp.evaluate_step.entrypoint(
        merged_dir="x",
        config_path=SMOKE_CONFIG,
        threshold=threshold,
        strict_behaviors=strict_behaviors,
    )


def test_eval_gate_blocks_low_exact_match(fake_scores):
    fake_scores["exact_match"] = 0.30
    with pytest.raises(RuntimeError, match="Eval gate FAILED: 0.300 < 0.595"):
        evaluate()


def test_behavioral_gate_blocks_one_broken_behavior(fake_scores):
    fake_scores["exact_match"] = 0.70  # aggregate fine ...
    fake_scores["behaviors"]["amount_tracks"] = False  # ... one behaviour broken
    with pytest.raises(
        RuntimeError, match=r"Behavioral gate FAILED: \['amount_tracks'\]"
    ):
        evaluate()


def test_smoke_mode_records_behaviors_but_does_not_enforce(fake_scores):
    fake_scores["exact_match"] = 0.70
    fake_scores["behaviors"]["amount_tracks"] = False

    report = evaluate(strict_behaviors=False)

    assert report["behaviors"]["amount_tracks"] is False  # recorded
    with open(GATE_MANIFEST) as f:
        assert report["gate_manifest_sha"] == json.load(f)["sha256"]  # stamped


def test_data_gate_turns_validation_exit_into_step_failure():
    bad_row = {"sms": "x", "label": {"is_transaction": True, "amount": "45O.OO"}}
    with pytest.raises(RuntimeError, match="Data gate FAILED"):
        tp.validate.entrypoint([bad_row])


def test_valid_rows_pass_the_data_gate_unchanged():
    row = {"sms": "x", "label": {"is_transaction": False}}
    assert tp.validate.entrypoint([row]) == [row]
