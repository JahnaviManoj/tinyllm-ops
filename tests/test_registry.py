import json
from typing import ClassVar

import mlflow
import pytest
from conftest import run_with
from mlflow.tracking import MlflowClient

from tinyllm.registry import CHALLENGER, MODEL_NAME, register_as_challenger

REPORT = {"exact_match": 0.83, "behaviors": {"amount_tracks": 0.9}}


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


# ---- 4.6: the champion's GGUF goes to one fixed Blob address


class FakeBlob:
    store: ClassVar[dict[str, bytes]] = {}
    fail_on: ClassVar[str | None] = None

    def __init__(self, path):
        self.path = path

    def upload_blob(self, data, overwrite=False, **_):
        assert overwrite
        if self.path == FakeBlob.fail_on:
            raise OSError("blob upload failed")
        FakeBlob.store[self.path] = data if isinstance(data, bytes) else data.read()


@pytest.fixture
def fake_blob(monkeypatch):
    from tinyllm import registry

    FakeBlob.store, FakeBlob.fail_on = {}, None
    monkeypatch.setattr(registry, "blob_client", FakeBlob)
    return FakeBlob


def run_with_gguf(tmp_path, content=b"GGUF-bytes"):
    """A run like the pipeline leaves it: model/ (registrable) + gguf/<quant>.gguf."""
    m = tmp_path / "src" / "model"
    g = tmp_path / "src" / "gguf"
    m.mkdir(parents=True, exist_ok=True)
    g.mkdir(parents=True, exist_ok=True)
    (m / "weights.bin").write_bytes(b"x")
    (g / "Q8_0.gguf").write_bytes(content)
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(m), artifact_path="model")
        mlflow.log_artifacts(str(g), artifact_path="gguf")
    return run.info.run_id


def test_publish_champion_writes_model_and_info_at_the_fixed_address(store, fake_blob):
    import hashlib

    from tinyllm.registry import publish_champion_gguf

    mv = register_as_challenger(run_with_gguf(store), REPORT)

    info = publish_champion_gguf(mv, "Q8_0")

    assert fake_blob.store["champion/model.gguf"] == b"GGUF-bytes"
    written = json.loads(fake_blob.store["champion/model.json"])
    assert (
        written["sha256"] == hashlib.sha256(b"GGUF-bytes").hexdigest() == info["sha256"]
    )
    assert written["model_version"] == str(mv.version) and written["quant"] == "Q8_0"
    assert written["run_id"] == mv.run_id and written["bytes"] == len(b"GGUF-bytes")


def test_publish_without_a_gguf_on_the_run_fails_before_touching_blob(store, fake_blob):
    from tinyllm.registry import publish_champion_gguf

    mv = register_as_challenger(run_with(store, "model"), REPORT)  # no gguf/

    with pytest.raises(Exception):  # noqa: B017 — MLflow's error type varies by store
        publish_champion_gguf(mv, "Q8_0")
    assert fake_blob.store == {}
