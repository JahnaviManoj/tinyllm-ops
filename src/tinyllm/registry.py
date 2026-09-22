"""MLflow model registry (Stage 3.3): register a run's merged model as @challenger.

The registry is a shelf of immutable, numbered versions; an alias is a movable
label. This module only ever sets ``@challenger``. Moving ``@champion`` is a
separate decision (``tinyllm.promote``, run by the CD workflow) so a model can
never promote itself.

Azure ML's MLflow endpoint returns 404 for the alias API (verified 2026-09-22),
so aliases are emulated as registered-model tags ``alias.<name> = <version>``
there — same semantics (one movable pointer per alias), plain MLflow elsewhere.
``set_alias`` / ``get_version_by_alias`` are the only two doors; promote.py and
serving go through them too, so the backend never leaks into callers.
"""

from __future__ import annotations

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

MODEL_NAME = "tinyllm-sms-parser"
CHALLENGER = "challenger"
CHAMPION = "champion"
# train.py logs the merged model here only with --log-merged; sweep runs carry adapter/ alone.
MODEL_ARTIFACT = "model"


def register_as_challenger(
    run_id: str,
    report: dict,
    *,
    model_name: str = MODEL_NAME,
    artifact_path: str = MODEL_ARTIFACT,
) -> ModelVersion:
    """Register ``<run>/<artifact_path>`` as a new version and point
    ``@challenger`` at it. The eval report is attached to the run as evidence.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    logged = {a.path for a in client.list_artifacts(run_id)}
    if artifact_path not in logged:
        raise FileNotFoundError(
            f"run {run_id} has no '{artifact_path}/' artifact "
            f"(logged: {sorted(logged)}); re-run train.py with --log-merged"
        )
    # The merged model is a plain log_artifacts() directory, not an MLflow-3
    # LoggedModel, so mlflow.register_model("runs:/...") can't resolve it. The
    # client API registers any artifact URI directly (and waits until READY).
    _ensure_registered_model(client, model_name)
    source = f"{run.info.artifact_uri.rstrip('/')}/{artifact_path}"
    mv = client.create_model_version(model_name, source=source, run_id=run_id)
    set_alias(client, model_name, CHALLENGER, mv.version)
    client.log_dict(run_id, report, "eval_report.json")
    return mv


def set_alias(client: MlflowClient, name: str, alias: str, version: str) -> None:
    """Point ``@alias`` at ``version``; native alias API, or a tag on Azure ML."""
    try:
        client.set_registered_model_alias(name, alias, version)
    except MlflowException as e:
        if not _alias_api_missing(e):
            raise
        client.set_registered_model_tag(name, f"alias.{alias}", str(version))


def get_version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion:
    """Resolve ``@alias``; raises RESOURCE_DOES_NOT_EXIST when it is unset."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as e:
        if not _alias_api_missing(e):
            raise
    version = client.get_registered_model(name).tags.get(f"alias.{alias}")
    if version is None:
        raise MlflowException(
            f"Registered model '{name}' has no alias '{alias}'",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    return client.get_model_version(name, version)


def _alias_api_missing(e: MlflowException) -> bool:
    return e.error_code == "ENDPOINT_NOT_FOUND"


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.create_registered_model(name)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
