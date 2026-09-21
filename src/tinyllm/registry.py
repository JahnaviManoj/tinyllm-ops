"""MLflow model registry (Stage 3.3): register a run's merged model as @challenger.

The registry is a shelf of immutable, numbered versions; an alias is a movable
label. This module only ever sets ``@challenger``. Moving ``@champion`` is a
separate decision (``tinyllm.promote``, run by the CD workflow) so a model can
never promote itself.
"""

from __future__ import annotations

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "tinyllm-sms-parser"
CHALLENGER = "challenger"
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
    client.set_registered_model_alias(model_name, CHALLENGER, mv.version)
    client.log_dict(run_id, report, "eval_report.json")
    return mv


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.create_registered_model(name)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
