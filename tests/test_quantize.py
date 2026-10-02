"""quantize_to_gguf (4.1) against a fake llama.cpp: no model, no real conversion.

The fake converter refuses to run without --no-nextn (the Qwen3.5 MTP trap),
so the wrapper is pinned to the flag that made the real conversion loadable.
"""

import os
import pathlib
import stat

import mlflow
import pytest
from conftest import run_with
from mlflow.tracking import MlflowClient

from pipelines import training_pipeline as tp
from tinyllm import quantize
from tinyllm.quantize import quantize_to_gguf


@pytest.fixture
def fake_llama(tmp_path):
    d = tmp_path / "llama.cpp"
    (d / "build" / "bin").mkdir(parents=True)
    (d / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "a = sys.argv[1:]\n"
        "assert '--no-nextn' in a, 'MTP trap: --no-nextn missing'\n"
        "open(a[a.index('--outfile') + 1], 'wb').write(b'F16' * 100)\n"
    )
    q = d / "build" / "bin" / "llama-quantize"
    q.write_text('#!/bin/sh\nprintf "$3" > "$2"\n')  # writes the quant name as bytes
    q.chmod(q.stat().st_mode | stat.S_IEXEC)
    return d


@pytest.fixture
def merged(tmp_path):
    m = tmp_path / "outputs" / "exp_x" / "merged"
    m.mkdir(parents=True)
    return m


def test_converts_then_quantizes_next_to_the_merged_model(fake_llama, merged):
    paths = quantize_to_gguf(str(merged), llama_cpp_dir=str(fake_llama))

    assert set(paths) == {"f16", "Q8_0", "Q4_K_M"}
    assert paths["Q8_0"] == str(merged.parent / "gguf" / "Q8_0.gguf")
    assert pathlib.Path(paths["f16"]).read_bytes() == b"F16" * 100
    assert pathlib.Path(paths["Q4_K_M"]).read_text() == "Q4_K_M"


def test_existing_outputs_are_reused(fake_llama, merged):
    first = quantize_to_gguf(str(merged), llama_cpp_dir=str(fake_llama))
    os.utime(first["Q8_0"], (0, 0))  # tamper with mtime: a rerun must not rewrite

    quantize_to_gguf(str(merged), llama_cpp_dir=str(fake_llama))

    assert os.stat(first["Q8_0"]).st_mtime == 0


def test_missing_llama_cpp_dir_is_a_clear_error(merged, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    with pytest.raises(OSError, match="LLAMA_CPP_DIR"):
        quantize_to_gguf(str(merged))


def test_sizes_and_files_land_on_the_run(store, fake_llama, merged):
    run_id = run_with(store, "model")

    quantize_to_gguf(str(merged), llama_cpp_dir=str(fake_llama), run_id=run_id)

    client = MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert metrics["gguf_bytes_Q8_0"] == 4.0 and metrics["gguf_bytes_Q4_K_M"] == 6.0
    logged = {a.path for a in client.list_artifacts(run_id, "gguf")}
    assert logged == {"gguf/Q8_0.gguf", "gguf/Q4_K_M.gguf"}  # the f16 stays local
    mlflow.end_run()


def test_quantize_step_passes_the_dict_through(monkeypatch):
    monkeypatch.setattr(
        tp, "quantize_to_gguf", lambda d, run_id: {"Q8_0": f"{d}/gguf/Q8_0.gguf"}
    )
    assert tp.quantize_step.entrypoint("x/merged", "run", {"exact_match": 0.7}) == {
        "Q8_0": "x/merged/gguf/Q8_0.gguf"
    }


def test_pinned_commit_is_a_full_sha():
    assert len(quantize.LLAMA_CPP_COMMIT) == 40


def test_existing_outputs_need_no_llama_cpp(fake_llama, merged, monkeypatch):
    quantize_to_gguf(
        str(merged), llama_cpp_dir=str(fake_llama)
    )  # builds everything once
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    paths = quantize_to_gguf(str(merged))  # no llama.cpp anywhere: must just reuse

    assert set(paths) == {"f16", "Q8_0", "Q4_K_M"}
