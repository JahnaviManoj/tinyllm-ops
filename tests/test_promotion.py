"""The referee's verdicts (tutorial 3.4/3.6) on a throwaway sqlite registry.

Reports are hand-written; no model is ever loaded. Rows are on the real gate
manifest (356 rows): floor 0.595 → 212, gain ≥ 17.
"""

import json

import mlflow
import pytest
from conftest import run_with
from mlflow.tracking import MlflowClient

from tinyllm import promote
from tinyllm.registry import (
    CHAMPION,
    GATE_MANIFEST,
    MODEL_NAME,
    get_version_by_alias,
    register_as_challenger,
    set_alias,
)

N = 356
with open(GATE_MANIFEST) as _f:
    GATE_SHA = json.load(_f)["sha256"]


def report(rows: int, sha: str = GATE_SHA) -> dict:
    return {"exact_match": rows / N, "behaviors": {}, "gate_manifest_sha": sha}


def registry_with(store, champion_rows, challenger_rows, champ_sha=GATE_SHA):
    """Two versions: the first wears @champion, the second @challenger."""
    champ = register_as_challenger(
        run_with(store, "model"), report(champion_rows, champ_sha)
    )
    set_alias(MlflowClient(), MODEL_NAME, CHAMPION, champ.version)
    chall = register_as_challenger(run_with(store, "model"), report(challenger_rows))
    return champ, chall


def champion_version() -> str:
    return get_version_by_alias(MlflowClient(), MODEL_NAME, CHAMPION).version


def test_challenger_below_floor_is_rejected_even_without_a_champion(store):
    chall = register_as_challenger(run_with(store, "model"), report(100))

    d = promote.promote_if_better(threshold=0.595)

    assert (d["promoted"], d["reason"]) == (False, "below_floor")
    assert d["line"] == f"REJECTED v{chall.version}: 100 < floor 212"
    with pytest.raises(mlflow.exceptions.MlflowException):
        champion_version()  # the auto-promote trap: nothing was crowned


def test_first_model_above_floor_is_promoted(store):
    chall = register_as_challenger(run_with(store, "model"), report(229))

    d = promote.promote_if_better(threshold=0.595)

    assert (d["promoted"], d["reason"]) == (True, "no_incumbent")
    assert champion_version() == chall.version


def test_worse_than_champion_is_rejected_and_alias_unchanged(store):
    champ, _ = registry_with(store, champion_rows=229, challenger_rows=212)

    d = promote.promote_if_better(threshold=0.595)

    assert (d["promoted"], d["reason"]) == (False, "inside_noise_floor")
    assert champion_version() == champ.version


def test_gain_inside_noise_floor_keeps_incumbent(store):
    champ, _ = registry_with(store, champion_rows=229, challenger_rows=240)  # +11 < 17

    d = promote.promote_if_better(threshold=0.595)

    assert d["promoted"] is False
    assert "inside the noise floor" in d["line"]
    assert champion_version() == champ.version


def test_gain_of_noise_floor_or_more_promotes(store):
    _, chall = registry_with(store, champion_rows=229, challenger_rows=250)  # +21

    d = promote.promote_if_better(threshold=0.595)

    assert (d["promoted"], d["reason"]) == (True, "beats_champion")
    assert champion_version() == chall.version


def test_dry_run_decides_but_moves_nothing(store):
    champ, _ = registry_with(store, champion_rows=229, challenger_rows=250)

    d = promote.promote_if_better(threshold=0.595, dry_run=True)

    assert d["promoted"] is True and d["dry_run"] is True
    assert champion_version() == champ.version


def test_stale_champion_report_triggers_one_rescore(store, monkeypatch):
    champ, _ = registry_with(store, 229, 250, champ_sha="old-gate")
    calls = []
    monkeypatch.setattr(promote, "download_model", lambda *a, **k: "model-dir")
    monkeypatch.setattr(promote, "fetch_local", lambda *a, **k: "gate.jsonl")
    monkeypatch.setattr(
        promote,
        "eval_model",
        lambda *a, **k: calls.append(a) or {"exact_match": 240 / N},
    )

    d = promote.promote_if_better(threshold=0.595)

    assert len(calls) == 1 and calls[0][0] == "model-dir"
    assert d["champion_rows"] == 240  # the fresh score, not the stale 229
    assert d["promoted"] is False  # 250 - 240 = +10 < 17
    # the re-score was attached, so the next run takes the fast path
    from tinyllm.registry import get_report

    assert get_report(MlflowClient(), champ)["gate_manifest_sha"] == GATE_SHA


def test_challenger_that_is_already_champion_is_rejected(store):
    chall = register_as_challenger(run_with(store, "model"), report(229))
    set_alias(MlflowClient(), MODEL_NAME, CHAMPION, chall.version)

    d = promote.promote_if_better(threshold=0.595)

    assert (d["promoted"], d["reason"]) == (False, "already_champion")


# ---- 4.6: promotion = alias move + GGUF publish, as one transaction


@pytest.fixture(autouse=True)
def publish(monkeypatch):
    """Every promotion test runs with the Blob publish faked (autouse); a test
    that wants the failure path sets ``fake.fail = True``."""
    calls = []

    def fake(mv, quant, client=None):
        calls.append((mv.version, quant))
        if getattr(fake, "fail", False):
            raise OSError("blob down")
        return {"sha256": "abc123def456", "quant": quant}

    monkeypatch.setattr(promote, "publish_champion_gguf", fake)
    return fake, calls


def test_promotion_publishes_the_served_quant(store, publish):
    _, calls = publish
    _, chall = registry_with(store, champion_rows=229, challenger_rows=250)

    d = promote.promote_if_better(threshold=0.595)

    assert d["promoted"] and calls == [(chall.version, "Q8_0")]
    assert d["published"]["sha256"] == "abc123def456" and "published" in d["line"]


def test_failed_publish_rolls_the_alias_back_to_the_old_champion(store, publish):
    fake, calls = publish
    fake.fail = True
    champ, _ = registry_with(store, champion_rows=229, challenger_rows=250)

    with pytest.raises(OSError, match="blob down"):
        promote.promote_if_better(threshold=0.595)

    assert champion_version() == champ.version  # moved, then moved back
    assert len(calls) == 1


def test_failed_publish_with_no_old_champion_unsets_the_alias(store, publish):
    fake, _ = publish
    fake.fail = True
    register_as_challenger(run_with(store, "model"), report(229))

    with pytest.raises(OSError):
        promote.promote_if_better(threshold=0.595)

    with pytest.raises(mlflow.exceptions.MlflowException):
        champion_version()  # nothing wears @champion


def test_dry_run_never_publishes(store, publish):
    _, calls = publish
    registry_with(store, champion_rows=229, challenger_rows=250)

    promote.promote_if_better(threshold=0.595, dry_run=True)

    assert calls == []
