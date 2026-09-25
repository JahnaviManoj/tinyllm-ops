"""3.4 step 2 — the one manual @champion move in the project's life.

exp_111 (run 85f39fed) is adapter-only in MLflow. The merged fp32 model rebuilt
locally from that adapter re-scores 229/356 on gate_v2 (identical to Colab), so
this script (a) uploads it into the same run as model/ — file by file, skipping
files whose blob already has the right size, so an interrupted upload resumes —
(b) registers it with the gate_v2 report reconstructed from the gate2-exp_111
eval run, stamped with the gate manifest sha, (c) sets @champion on it by hand,
and (d) puts @challenger back where it was (registering always moves it).

  uv run python scripts/register_champion.py            # does it
  uv run python scripts/register_champion.py --dry-run  # report + per-file status, touches nothing

The azureml-mlflow plugin gives an upload batch 300 s to finish (then
"Failed to flush task queue"); the 3 GB safetensors needs far more, hence the
timeout below. An interrupted upload leaves a metadata entry with no blob
behind it; re-creating the entry is allowed, so the resume just re-uploads.
"""

import argparse
import json
import os
import sys

os.environ.setdefault(
    "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT", "14400"
)  # 4 h, read per batch

import mlflow
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient
from dotenv import load_dotenv
from mlflow.store.artifact.artifact_repository_registry import (
    get_artifact_repository,
)
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinyllm.registry import (
    CHALLENGER,
    CHAMPION,
    GATE_MANIFEST,
    MODEL_ARTIFACT,
    MODEL_NAME,
    get_version_by_alias,
    register_as_challenger,
    set_alias,
)

TRAIN_RUN = "85f39fed-b454-4b2f-b194-a4e4c3bf209d"  # exp_111, adapter/ only
GATE_RUN = "562d33ec-f135-44f8-9b91-e92c611f58d9"  # gate2-exp_111: 229/356
MERGED_DIR = "outputs/exp_111/merged"


def gate_report(client: MlflowClient) -> dict:
    """Rebuild eval_model()'s report dict from the gate-eval run's metrics."""
    m = client.get_run(GATE_RUN).data.metrics
    report = {k: v for k, v in m.items() if not k.startswith("field_acc_")}
    report["per_field_accuracy"] = {
        k.removeprefix("field_acc_"): v
        for k, v in m.items()
        if k.startswith("field_acc_")
    }
    with open(GATE_MANIFEST) as f:
        report["gate_manifest_sha"] = json.load(f)["sha256"]
    report["source"] = f"metrics of run {GATE_RUN} (gate2-exp_111), 2026-09-12"
    return report


def remote_size(art, path: str) -> int | None:
    """Bytes actually stored for a run artifact, or None if the blob is missing
    (Azure ML registers the path before the bytes land, so listing alone lies)."""
    sc = art._service_context
    try:
        info = art._client.run_artifacts.get_content_information(
            subscription_id=sc.subscription_id,
            resource_group_name=sc.resource_group_name,
            workspace_name=sc.workspace_name,
            experiment_name=art._experiment_name,
            run_id=TRAIN_RUN,
            path=path,
        )
        return BlobClient.from_blob_url(info.content_uri).get_blob_properties().size
    except ResourceNotFoundError:
        return None
    except Exception as e:  # no metadata entry at all → the REST client raises
        if "404" in str(e) or "not found" in str(e).lower():
            return None
        raise


def file_status(client: MlflowClient) -> list[tuple[str, str, int, int | None]]:
    """(local path, remote path, local bytes, remote bytes | None) per file."""
    art = get_artifact_repository(client.get_run(TRAIN_RUN).info.artifact_uri).artifacts
    out = []
    for name in sorted(os.listdir(MERGED_DIR)):
        local = os.path.join(MERGED_DIR, name)
        remote = f"{MODEL_ARTIFACT}/{name}"
        out.append((local, remote, os.path.getsize(local), remote_size(art, remote)))
    return out


def upload_missing(client: MlflowClient) -> None:
    for local, remote, want, have in file_status(client):
        if have == want:
            print(f"  ok       {remote} ({want} B)")
            continue
        print(
            f"  upload   {remote} ({want / 1e9:.2f} GB; remote had {have}) ...",
            flush=True,
        )
        client.log_artifact(TRAIN_RUN, local, MODEL_ARTIFACT)
    bad = [(r, w, h) for _, r, w, h in file_status(client) if h != w]
    if bad:
        sys.exit(f"upload incomplete, not registering: {bad}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    client = MlflowClient()

    report = gate_report(client)
    rows = round(report["exact_match"] * 356)
    print(f"report: exact_match={report['exact_match']:.4f} = {rows}/356")
    assert rows == 229, "that is not exp_111's gate number — stop"
    if args.dry_run:
        print(json.dumps(report, indent=2))
        print(
            f"files in run {TRAIN_RUN[:8]}/{MODEL_ARTIFACT}/ (local bytes -> remote bytes):"
        )
        for _, remote, want, have in file_status(client):
            print(
                f"  {'ok     ' if have == want else 'MISSING'} {remote}: {want} -> {have}"
            )
        return

    print(f"syncing {MERGED_DIR} -> run {TRAIN_RUN[:8]}/{MODEL_ARTIFACT}/")
    upload_missing(client)

    try:
        prev_challenger = get_version_by_alias(client, MODEL_NAME, CHALLENGER).version
    except mlflow.exceptions.MlflowException:
        prev_challenger = None

    mv = register_as_challenger(TRAIN_RUN, report)
    set_alias(client, MODEL_NAME, CHAMPION, mv.version)
    if prev_challenger is not None:
        set_alias(client, MODEL_NAME, CHALLENGER, prev_challenger)

    champ = get_version_by_alias(client, MODEL_NAME, CHAMPION)
    chall = get_version_by_alias(client, MODEL_NAME, CHALLENGER)
    print(f"@champion  → v{champ.version} (run {champ.run_id[:8]})")
    print(f"@challenger → v{chall.version} (run {chall.run_id[:8]})")


if __name__ == "__main__":
    main()
