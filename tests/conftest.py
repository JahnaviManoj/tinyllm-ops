"""Shared fixtures: a throwaway MLflow tracking + registry store (sqlite in
tmp_path, nothing under ./mlruns) and a helper that logs fake artifacts.

Creating an MLflow sqlite store runs its alembic migrations, 2–10 s depending
on the disk. That is paid once per session; each test gets a file copy of the
migrated, still-empty database and only creates its own experiment in it.
"""

import shutil

import mlflow
import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="session")
def _migrated_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("mlflow-schema") / "mlflow.db"
    uri = f"sqlite:///{path}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.search_experiments()  # tracking tables
    client.search_registered_models()  # registry tables
    return path


@pytest.fixture
def store(tmp_path, _migrated_db):
    shutil.copy(_migrated_db, tmp_path / "mlflow.db")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = mlflow.create_experiment("t", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_id=exp)
    return tmp_path


def run_with(tmp_path, *artifact_paths):
    """A finished run carrying one dummy file under each artifact path."""
    with mlflow.start_run() as run:
        for p in artifact_paths:
            d = tmp_path / "src" / p
            d.mkdir(parents=True, exist_ok=True)
            (d / "weights.bin").write_bytes(b"x")
            mlflow.log_artifacts(str(d), artifact_path=p)
    return run.info.run_id
