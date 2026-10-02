"""The promotion referee (tutorial 3.4): move ``@champion`` onto ``@challenger``
only on a win under the rule pre-registered in docs/decisions.md (2026-09-24).

Two labels on the registry shelf: ``@challenger`` (the newest model that passed
the pipeline's gates, set by 3.3) and ``@champion`` (the one we ship). A program
decides whether the champion label moves — never a person — and it decides in
**rows on the gate set**, not percentages, because the gate's seed noise floor
was measured in rows (17 on gate_v2, 2026-09-12):

  floor   challenger rows >= ceil(threshold * n)     else REJECTED, even with no champion
  gain    challenger rows - champion rows >= 17      else REJECTED (ties keep the incumbent)

Scores are comparable only on the same exam. Each ``eval_report.json`` carries
``gate_manifest_sha``; the challenger's report (minutes old, written by the
pipeline) is used as-is, the champion is re-scored on the current gate only if
its sha differs — the one slow path, and rare.

  uv run python -m tinyllm.promote --threshold 0.595 --dry-run   # decide, move nothing
  uv run python -m tinyllm.promote --threshold 0.595             # CD: alias flip on PROMOTED
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from tinyllm.manifest import fetch_local
from tinyllm.registry import (
    CHALLENGER,
    CHAMPION,
    GATE_MANIFEST,
    MODEL_ARTIFACT,
    MODEL_NAME,
    REPORT_ARTIFACT,
    SERVED_QUANT,
    get_report,
    get_version_by_alias,
    publish_champion_gguf,
    set_alias,
    unset_alias,
)

# Measured seed noise floor on gate_v2, in rows (docs/decisions.md, 2026-09-12).
NOISE_FLOOR_ROWS = 17


def promote_if_better(
    threshold: float,
    min_gain_rows: int = NOISE_FLOOR_ROWS,
    *,
    dry_run: bool = False,
    model_name: str = MODEL_NAME,
    gate_manifest: str = GATE_MANIFEST,
    client: MlflowClient | None = None,
    quant: str = SERVED_QUANT,
) -> dict:
    """Decide, print one human line, move ``@champion`` only on PROMOTED
    (and never with ``dry_run``). Returns the decision as a dict.

    Promotion is a two-step transaction (4.6): move the alias, then copy the
    champion's GGUF to the fixed Blob address serving reads. If the copy fails
    the alias is moved back, so registry and production never disagree."""
    client = client or MlflowClient()
    with open(gate_manifest) as f:
        manifest = json.load(f)
    n, sha = manifest["num_examples"], manifest["sha256"]
    floor = math.ceil(threshold * n - 1e-9)

    challenger = get_version_by_alias(client, model_name, CHALLENGER)
    chall_report = get_report(client, challenger)
    if chall_report.get("gate_manifest_sha") != sha:
        # The pipeline wrote this minutes ago on the current gate; a mismatch means a
        # pre-3.4 report or a gate refresh since — say so, but the rule still applies.
        print(
            f"warning: challenger v{challenger.version}'s report is not stamped with "
            f"the current gate ({sha[:8]}); using it as-is"
        )
    chall_rows = _rows(chall_report, n)

    champion = _champion_or_none(client, model_name)
    champ_rows = None
    if champion is not None and champion.version != challenger.version:
        champ_report = _report_on_current_gate(
            client, champion, manifest, gate_manifest
        )
        champ_rows = _rows(champ_report, n)

    decision = _decide(
        challenger.version,
        chall_rows,
        champion.version if champion is not None else None,
        champ_rows,
        floor,
        min_gain_rows,
    )
    decision.update(n=n, threshold=threshold, gate_manifest_sha=sha, dry_run=dry_run)
    if decision["promoted"] and not dry_run:
        set_alias(client, model_name, CHAMPION, challenger.version)
        try:
            info = publish_champion_gguf(challenger, quant, client=client)
        except Exception as e:
            if champion is not None:
                set_alias(client, model_name, CHAMPION, champion.version)
            else:
                unset_alias(client, model_name, CHAMPION)
            decision.update(
                promoted=False,
                reason="publish_failed",
                line=(
                    f"ROLLED BACK v{challenger.version}: alias moved, GGUF copy failed "
                    f"({type(e).__name__}: {e}); @champion restored to "
                    + (f"v{champion.version}" if champion is not None else "unset")
                ),
            )
            print(decision["line"])
            raise
        decision["published"] = info
        decision["line"] += f"; served {quant} sha {info['sha256'][:12]} published"
    print(decision["line"] + ("  [dry-run: nothing moved]" if dry_run else ""))
    return decision


def _decide(
    chall_v: str,
    chall_rows: int,
    champ_v: str | None,
    champ_rows: int | None,
    floor: int,
    min_gain: int,
) -> dict:
    """The rule, pure: rows in, verdict out. One line per verdict for the CD log."""
    d = {
        "challenger": chall_v,
        "challenger_rows": chall_rows,
        "champion": champ_v,
        "champion_rows": champ_rows,
        "floor_rows": floor,
        "min_gain_rows": min_gain,
    }
    if chall_rows < floor:
        return {
            **d,
            "promoted": False,
            "reason": "below_floor",
            "line": f"REJECTED v{chall_v}: {chall_rows} < floor {floor}",
        }
    if champ_v is not None and champ_v == chall_v:
        return {
            **d,
            "promoted": False,
            "reason": "already_champion",
            "line": f"REJECTED v{chall_v}: already @champion",
        }
    if champ_rows is None:
        return {
            **d,
            "promoted": True,
            "reason": "no_incumbent",
            "line": f"PROMOTED v{chall_v}: {chall_rows} >= floor {floor}, no champion",
        }
    gain = chall_rows - champ_rows
    if gain < min_gain:
        return {
            **d,
            "promoted": False,
            "reason": "inside_noise_floor",
            "line": (
                f"REJECTED v{chall_v}: {chall_rows} vs champion v{champ_v} "
                f"{champ_rows} ({gain:+d} < {min_gain}, inside the noise floor)"
            ),
        }
    return {
        **d,
        "promoted": True,
        "reason": "beats_champion",
        "line": (
            f"PROMOTED v{chall_v}: {chall_rows} >= {champ_rows} + {min_gain} "
            f"(champion v{champ_v}, {gain:+d})"
        ),
    }


def _rows(report: dict, n: int) -> int:
    return round(report["exact_match"] * n)


def _champion_or_none(client: MlflowClient, model_name: str) -> ModelVersion | None:
    try:
        return get_version_by_alias(client, model_name, CHAMPION)
    except MlflowException as e:
        # Azure-tag fallback raises RESOURCE_DOES_NOT_EXIST; the native alias API
        # (sqlite/file stores) raises INVALID_PARAMETER_VALUE "alias ... not found".
        unset = (
            e.error_code == "RESOURCE_DOES_NOT_EXIST" or "not found" in str(e).lower()
        )
        if not unset:
            raise
        return None


def _report_on_current_gate(
    client: MlflowClient, mv: ModelVersion, manifest: dict, gate_manifest: str
) -> dict:
    """The version's report if it was scored on the current gate; otherwise
    re-score it (download ``model/``, run the gate) and re-attach the report."""
    report = get_report(client, mv)
    if report.get("gate_manifest_sha") == manifest["sha256"]:
        return report
    print(
        f"champion v{mv.version}: report is from another gate "
        f"({report.get('gate_manifest_sha', 'unstamped')!s:.8}); re-scoring on "
        f"{manifest['name']}_{manifest['version']} — this is the slow path"
    )
    chat = client.get_run(mv.run_id).data.params.get("chat_format", "True") == "True"
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = download_model(client, mv, tmp)
        report = eval_model(model_dir, fetch_local(gate_manifest), chat=chat)
    report["gate_manifest_sha"] = manifest["sha256"]
    client.log_dict(mv.run_id, report, REPORT_ARTIFACT)
    return report


def download_model(client: MlflowClient, mv: ModelVersion, dst: str) -> str:
    return mlflow.artifacts.download_artifacts(
        run_id=mv.run_id,
        artifact_path=MODEL_ARTIFACT,
        dst_path=dst,
        tracking_uri=client.tracking_uri,
    )


def eval_model(*args, **kwargs) -> dict:
    """Lazy door to tinyllm.model_eval: the torch stack loads only on the rare
    re-score path (and tests swap this function for a sentinel)."""
    from tinyllm.model_eval import eval_model as _eval_model

    return _eval_model(*args, **kwargs)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="gate exact-match floor, pre-registered in docs/decisions.md (0.595)",
    )
    parser.add_argument(
        "--min-gain-rows",
        type=int,
        default=NOISE_FLOOR_ROWS,
        help=f"rows a challenger must beat the champion by (default {NOISE_FLOOR_ROWS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the decision, move nothing"
    )
    parser.add_argument(
        "--quant", default=SERVED_QUANT, help="which GGUF to publish on promotion"
    )
    args = parser.parse_args()
    promote_if_better(
        args.threshold, args.min_gain_rows, dry_run=args.dry_run, quant=args.quant
    )


if __name__ == "__main__":
    main()
