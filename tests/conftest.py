"""Shared fixtures: a throwaway MLflow tracking + registry store (sqlite in
tmp_path, nothing under ./mlruns) and a helper that logs fake artifacts."""

import mlflow
import pytest


@pytest.fixture
def store(tmp_path):
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
