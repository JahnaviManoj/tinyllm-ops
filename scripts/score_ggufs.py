"""4.2: score each GGUF on the gate set through llama-server, with the same harness.

For every quant: start llama-server on the file, wait for /health, run
eval_model(backend="llama") on gate_v2 with a rows file, read generation
throughput from the server log, stop the server. Writes a summary next to the
GGUFs and logs each run to MLflow as gguf-<quant>.

  LLAMA_CPP_DIR=llama.cpp uv run python scripts/score_ggufs.py --exp exp_111 \
      --quants f16 Q8_0 Q4_K_M
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinyllm.manifest import fetch_local
from tinyllm.model_eval import eval_model
from tinyllm.registry import GATE_MANIFEST

EVAL_LINE = re.compile(
    r"(?<!prompt )eval time =\s+([\d.]+) ms /\s+(\d+) (?:runs|tokens)"
)
PROMPT_LINE = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")


def throughput(log: Path) -> dict:
    text = log.read_text(errors="replace")
    gen_ms = gen_tok = pr_ms = pr_tok = 0.0
    for ms, tok in EVAL_LINE.findall(text):
        gen_ms += float(ms)
        gen_tok += int(tok)
    for ms, tok in PROMPT_LINE.findall(text):
        pr_ms += float(ms)
        pr_tok += int(tok)
    return {
        "gen_tok_per_s": round(gen_tok / gen_ms * 1000, 1) if gen_ms else None,
        "prompt_tok_per_s": round(pr_tok / pr_ms * 1000, 1) if pr_ms else None,
        "gen_tokens": int(gen_tok),
    }


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default="exp_111")
    ap.add_argument("--quants", nargs="+", default=["f16", "Q8_0", "Q4_K_M"])
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None, help="default: gate_v2 by manifest")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument(
        "--constrained",
        action="store_true",
        help="grammar-constrained decoding (as served); run names get a -grammar suffix",
    )
    args = ap.parse_args()

    llama = Path(os.environ.get("LLAMA_CPP_DIR", "llama.cpp"))
    server = llama / "build" / "bin" / "llama-server"
    gguf_dir = Path("outputs") / args.exp / "gguf"
    tokenizer_dir = str(Path("outputs") / args.exp / "merged")
    data = args.data or fetch_local(GATE_MANIFEST)
    with open(data) as f:
        n = sum(1 for _ in f)
    url = f"http://127.0.0.1:{args.port}"
    tag = "-grammar" if args.constrained else ""
    summary_path = gguf_dir / f"gate2_scores{tag}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for q in args.quants:
        gguf = gguf_dir / f"{q}.gguf"
        log = gguf_dir / f"server_{q}{tag}.log"
        print(f"\n=== {q}: {gguf} ({gguf.stat().st_size / 1e9:.2f} GB) ===", flush=True)
        proc = subprocess.Popen(
            [
                str(server),
                "-m",
                str(gguf),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "-c",
                str(args.ctx),
                "-t",
                str(args.threads),
                "-np",
                "1",
                "--ctx-checkpoints",
                "0",  # hybrid model: each checkpoint is a full recurrent-state copy
                "--cache-ram",
                "0",  # host prompt cache (default 8 GB) parks every finished request's state
                "--log-file",
                str(log),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_healthy(url, proc)
            t0 = time.time()
            report = eval_model(
                str(gguf),
                data,
                run_name=None if args.no_mlflow else f"gguf-{q}{tag}",
                chat=True,
                rows_out=str(gguf_dir / f"gate2_rows_{q}{tag}.json"),
                backend="llama",
                server_url=url,
                tokenizer_dir=tokenizer_dir,
                constrained=args.constrained,
            )
            wall = time.time() - t0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        summary[q] = {
            "rows": round(report["exact_match"] * n),
            "n": n,
            "exact_match": report["exact_match"],
            "bytes": gguf.stat().st_size,
            "wall_s": round(wall),
            **throughput(log),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"--- {q}: {summary[q]}", flush=True)

    print("\n== summary ==")
    for q, r in summary.items():
        print(
            f"{q:8s} {r['rows']:>3}/{r['n']}  {100 * r['exact_match']:.1f}%  "
            f"{r['bytes'] / 1e9:.2f} GB  gen {r['gen_tok_per_s']} tok/s  {r['wall_s']} s"
        )


if __name__ == "__main__":
    main()
