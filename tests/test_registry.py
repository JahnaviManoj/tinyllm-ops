import json

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from tinyllm.registry import CHALLENGER, MODEL_NAME, register_as_challenger

REPORT = {"exact_match": 0.83, "behaviors": {"amount_tracks": 0.9}}


@pytest.fixture
def store(tmp_path):
    """A throwaway MLflow tracking + registry store, nothing under ./mlruns."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = mlflow.create_experiment("t", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_id=exp)
    return tmp_path


def run_with(tmp_path, *artifact_paths):
    with mlflow.start_run() as run:
        for p in artifact_paths:
            d = tmp_path / "src" / p
            d.mkdir(parents=True, exist_ok=True)
            (d / "weights.bin").write_bytes(b"x")
            mlflow.log_artifacts(str(d), artifact_path=p)
    return run.info.run_id


def test_registers_version_and_points_challenger_at_it(store):
    run_id = run_with(store, "adapter", "model")

    mv = register_as_challenger(run_id, REPORT)

    client = MlflowClient()
    assert (
        client.get_model_version_by_alias(MODEL_NAME, CHALLENGER).version == mv.version
    )
    assert mv.run_id == run_id
    assert (
        json.loads(
            (store / "artifacts").rglob("eval_report.json").__next__().read_text()
        )
        == REPORT
    )


def test_challenger_moves_to_newest_registration(store):
    first = register_as_challenger(run_with(store, "model"), REPORT)
    second = register_as_challenger(run_with(store, "model"), REPORT)

    assert second.version != first.version
    assert (
        MlflowClient().get_model_version_by_alias(MODEL_NAME, CHALLENGER).version
        == second.version
    )


def test_sweep_run_without_merged_model_is_refused(store):
    run_id = run_with(store, "adapter")  # log_merged=False → adapter/ only

    with pytest.raises(FileNotFoundError, match="no 'model/' artifact"):
        register_as_challenger(run_id, REPORT)
    assert MlflowClient().search_registered_models() == []  # nothing half-registered


# ---- Azure ML has no alias endpoint (404 → ENDPOINT_NOT_FOUND); tags stand in.


@pytest.fixture
def no_alias_api(monkeypatch):
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND

    def _404(*_a, **_k):
        raise MlflowException(
            "API request ... 404 != 200", error_code=ENDPOINT_NOT_FOUND
        )

    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", _404)
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", _404)


def test_alias_falls_back_to_tag_when_endpoint_missing(store, no_alias_api):
    from tinyllm.registry import get_version_by_alias

    mv = register_as_challenger(run_with(store, "model"), REPORT)

    client = MlflowClient()
    assert client.get_registered_model(MODEL_NAME).tags == {
        "alias.challenger": str(mv.version)
    }
    assert get_version_by_alias(client, MODEL_NAME, CHALLENGER).version == mv.version


def test_tag_alias_moves_and_unset_alias_raises(store, no_alias_api):
    from mlflow.exceptions import MlflowException

    from tinyllm.registry import get_version_by_alias

    register_as_challenger(run_with(store, "model"), REPORT)
    second = register_as_challenger(run_with(store, "model"), REPORT)
    client = MlflowClient()

    assert (
        get_version_by_alias(client, MODEL_NAME, CHALLENGER).version == second.version
    )
    with pytest.raises(MlflowException, match="no alias 'champion'"):
        get_version_by_alias(client, MODEL_NAME, "champion")
