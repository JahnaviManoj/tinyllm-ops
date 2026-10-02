"""Latency probe (tutorial 4.7): 5 warm-ups, then 50 timed /parse calls with a
typical ~120-char SMS; prints p50 / p95 wall latency and the server's own
tokens/s (from the gateway's X-Gen-Tok-Per-S header). Run the container pinned
to 2 threads to mimic the cloud box:

  docker run -d -p 8080:8080 --cpus=2 -e LLAMA_THREADS=2 -e API_KEYS=dev \
      -v $PWD/outputs/exp_111/gguf/Q8_0.gguf:/models/model.gguf tinyllm-ops
  uv run python monitoring/latency_probe.py --url http://localhost:8080 --n 50
"""

import argparse
import json
import statistics
import time

import httpx

SMS = "Rs.1,249.00 debited from A/c XX4412 on 01-10-26 to VPA zomato.order@paytm via UPI. Ref 629301847. Not you? Call 18002586161 -HDFC Bank"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--api-key", default="dev")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    a = ap.parse_args()
    headers = {"x-api-key": a.api_key}
    lat, tps, gen = [], [], []
    with httpx.Client(timeout=120) as c:
        for i in range(a.warmup + a.n):
            t0 = time.perf_counter()
            r = c.post(f"{a.url}/parse", json={"sms": SMS}, headers=headers)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            if i >= a.warmup:
                lat.append(dt)
                tps.append(float(r.headers.get("x-gen-tok-per-s", 0)))
                gen.append(int(r.headers.get("x-gen-tokens", 0) or 0))
    lat_sorted = sorted(lat)
    summary = {
        "n": a.n,
        "sms_chars": len(SMS),
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(lat_sorted[int(0.95 * len(lat_sorted)) - 1], 2),
        "max_s": round(lat_sorted[-1], 2),
        "gen_tokens_mean": round(statistics.mean(gen), 1) if gen else None,
        "gen_tok_per_s_mean": round(statistics.mean(tps), 1) if tps else None,
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
