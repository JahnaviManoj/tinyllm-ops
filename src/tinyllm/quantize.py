"""GGUF conversion + quantization (tutorial 4.1), wrapping llama.cpp.

The merged fp32 model needs a GPU-class box to run; converted to GGUF and
quantized it runs on a plain CPU. Two llama.cpp commands do that:

  python convert_hf_to_gguf.py <merged> --outfile f16.gguf --outtype f16 --no-nextn
  llama-quantize f16.gguf Q8_0.gguf Q8_0

``--no-nextn`` is not optional for Qwen3.5: the base checkpoint ships a
multi-token-prediction draft head whose weights a transformers save drops
while leaving ``mtp_num_hidden_layers`` in config.json, so without the flag
the converter declares one block too many and llama.cpp refuses the file
(decisions.md, 2026-09-29).

Files land in ``<merged>/../gguf/<quant>.gguf`` (``outputs/exp_111/gguf/Q8_0.gguf``);
in MLflow they are ``gguf/<quant>.gguf`` on the training run, which is what 4.6
downloads on promotion. The f16 stays local: it is reproducible from ``model/``.

  LLAMA_CPP_DIR=llama.cpp uv run python -m tinyllm.quantize outputs/exp_111/merged [--run-id ID]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# The checkout the probe ran on (llama.cpp master, 2026-09-29); serving/Dockerfile
# must build the same commit so the served kernels match the scored ones.
LLAMA_CPP_COMMIT = "7fee1784646b6196bdff9c144326217720099a3c"
DEFAULT_QUANTS = ("Q8_0", "Q4_K_M")
ARTIFACT_DIR = "gguf"


def quantize_to_gguf(
    merged_dir: str,
    out_dir: str | None = None,
    quants: tuple[str, ...] = DEFAULT_QUANTS,
    *,
    run_id: str | None = None,
    llama_cpp_dir: str | None = None,
) -> dict[str, str]:
    """Convert ``merged_dir`` to f16 GGUF and quantize it. Returns
    ``{"f16": path, "Q8_0": path, ...}``. Existing outputs are reused.
    With ``run_id``, each quant's byte size is logged as ``gguf_bytes_<quant>``
    and the quant files are attached under ``gguf/``."""
    out = Path(out_dir) if out_dir else Path(merged_dir).resolve().parent / ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    _llama: list[Path] = []  # resolved only if something must actually be built

    def llama_dir() -> Path:
        if not _llama:
            _llama.append(_llama_cpp_dir(llama_cpp_dir))
        return _llama[0]

    f16 = out / "f16.gguf"
    if not _present(f16):
        _run(
            [
                sys.executable,
                str(llama_dir() / "convert_hf_to_gguf.py"),
                merged_dir,
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
                "--no-nextn",
            ]
        )
    paths = {"f16": str(f16)}
    for q in quants:
        target = out / f"{q}.gguf"
        if not _present(target):
            _run(
                [
                    str(llama_dir() / "build" / "bin" / "llama-quantize"),
                    str(f16),
                    str(target),
                    q,
                ]
            )
        paths[q] = str(target)

    if run_id:
        _log_to_run(run_id, {q: paths[q] for q in quants})
    return paths


def _present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _llama_cpp_dir(explicit: str | None) -> Path:
    d = explicit or os.environ.get("LLAMA_CPP_DIR")
    if not d:
        raise OSError(
            "LLAMA_CPP_DIR is not set: point it at a built llama.cpp checkout "
            f"(commit {LLAMA_CPP_COMMIT[:12]}; `cmake -B build && cmake --build build -j`)"
        )
    p = Path(d)
    for needed in ("convert_hf_to_gguf.py", "build/bin/llama-quantize"):
        if not (p / needed).exists():
            raise FileNotFoundError(
                f"{p / needed} missing — is llama.cpp cloned and built?"
            )
    head = p / ".git" / "HEAD"
    if head.exists():
        commit = _git_head(p)
        if commit and commit != LLAMA_CPP_COMMIT:
            print(
                f"warning: llama.cpp at {commit[:12]}, pinned {LLAMA_CPP_COMMIT[:12]} — "
                "GGUFs may differ from the scored ones",
                flush=True,
            )
    return p


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _log_to_run(run_id: str, files: dict[str, str]) -> None:
    # azureml-mlflow gives each upload batch 300 s by default; a GGUF needs more.
    os.environ.setdefault("AZUREML_ARTIFACTS_DEFAULT_TIMEOUT", "14400")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    for q, path in files.items():
        client.log_metric(run_id, f"gguf_bytes_{q}", float(os.path.getsize(path)))
        client.log_artifact(run_id, path, ARTIFACT_DIR)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("merged_dir", help="e.g. outputs/exp_111/merged")
    parser.add_argument("--out-dir", help="default: <merged_dir>/../gguf")
    parser.add_argument("--quants", nargs="+", default=list(DEFAULT_QUANTS))
    parser.add_argument("--run-id", help="MLflow run to attach sizes + files to")
    args = parser.parse_args()
    paths = quantize_to_gguf(
        args.merged_dir, args.out_dir, tuple(args.quants), run_id=args.run_id
    )
    for q, p in paths.items():
        print(f"{q:8s} {os.path.getsize(p) / 1e9:.2f} GB  {p}")


if __name__ == "__main__":
    main()
