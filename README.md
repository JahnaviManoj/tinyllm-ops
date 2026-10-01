# TinyLLM Ops 💸📱

**Fine-tune a small language model to read bank transaction SMS and flag scams — and wrap
it in a production-grade MLOps platform that monitors itself and retrains itself when the
world changes. Total infrastructure cost: under $25.**

---

## The problem

Every bank payment in India generates an SMS — RBI mandates it, and UPI alone produces
billions of transactions a month. Expense-tracker apps parse these messages with
hand-written regex, one pattern per bank template, which silently breaks every time a bank
reformats its alerts. Meanwhile, scam texts imitating those same bank alerts are a genuine
consumer hazard. And because financial SMS is exactly the data you *shouldn't* ship to a
cloud API, the right solution has to be small enough to run cheaply — ideally on-device.

## The solution

Take a small Qwen3.5 model — the **0.8B**, after a five-run sweep showed the 2B could not
beat it beyond seed noise — and specialize it with **QLoRA fine-tuning** until it turns any
transaction SMS into a strict, schema-valid record:

```json
{
  "is_transaction": true,
  "txn_type": "debit",
  "amount": "450.00",
  "currency": "INR",
  "counterparty": "Swiggy",
  "account_tail": "1234",
  "channel": "UPI",
  "category": "food",
  "is_suspected_scam": false
}
```

The model is measured against an honest baseline ladder — from the regex parsers real
apps use, through zero/few-shot small models, up to a ~13×-bigger frontier-class model —
all on the same hand-reviewed gate set written by a *different* model than the training
teacher, with a frozen final set touched exactly once for the headline number:

| Model | Exact match, gate_v2 (356 rows)¹ | Serving cost |
|---|---|---|
| Per-template regex (industry baseline) | 0% | ~free |
| Qwen3.5-2B zero-shot (0.8B: also 0%) | 0% | CPU |
| Qwen3.5-0.8B few-shot | 6.5% | CPU |
| Qwen3.5-2B few-shot | 11.0% | CPU |
| Gemma 3 270M fine-tuned (Era-1 champion, template data) | 8.1% | CPU |
| Qwen3.5-2B fine-tuned (5 runs: 63.5–69.9%, tie-break loser) | 64.9% | CPU |
| **Qwen3.5-0.8B fine-tuned (this repo's champion, exp_111)** | **64.3%** | **CPU / on-device, ~$0/mo** |
| ↳ **the same model as served: Q8_0 GGUF on `llama-server`** | **63.8%** (227/356; f16 GGUF = 229, Q4_K_M = 209, rejected) | **0.81 GB, 23 tok/s on a laptop CPU** |
| Gemma 4 26B few-shot, pre-registered prompt (~30× bigger) | 27.0% | API $$ |
| Gemma 4 26B few-shot, told the output schema² | 72.2% | API $$ |

**Once-only sets, spent exactly once on the champion:** frozen final **59.9%** (106/177,
±7 pt) · out-of-distribution real SMS **44%** (11/25; the Era-1 champion scored 0% on the same
set) · real-scam holdout **91.7%** (55 of 60 real smishing texts flagged, none mistaken for a
transaction). The served Q8_0 quant is reported on the gate set only — the final set was spent
on the fp32 champion and is not reused (selection rule pre-registered 2026-09-30:
smallest quant within 7 rows of f16; 4-bit lost 20 rows, 8-bit lost 2).

¹ Strict metric: output must parse AND validate against the schema with *every* field
exactly right. The gate is 356 SMS written by a *different* teacher model than the training
teacher, reviewed row by row. Two seeds of the same recipe differ by 17 rows (4.8 pt), so
differences under that are noise — which is why the 0.8B and 2B are a tie, and the tie goes
to the smaller model by a rule fixed before any run. All runs in MLflow; every number's
provenance is in [docs/decisions.md](docs/decisions.md).

² The pre-registered 26B prompt named the fields but not their allowed values, so the model
answered `"channel": "Debit Card"` and failed the schema two thirds of the time. Told the
schema, it beats the champion by 28 rows. Both numbers are reported: **the 0.8B reaches 89%
of a schema-informed 26B's score, offline, at ~1/30th the parameters** — that is the honest
headline, not "beats the 26B". The one field every model stalls on (counterparty, ~75%)
traces to a train/test labelling inconsistency on ATM rows, diagnosed in the post-mortem and
left unfixed for this era by design.

The multi-size ladder also answers a question most fine-tuning projects skip:
**how much model does this task actually need?** Here: 0.8B, and no more.

## Why the engineering is the point

Training the model takes an afternoon. The project is the **platform around it** — a
closed loop that most fine-tuning demos never build:

```
 synthetic + curated data ──► validation gate ──► QLoRA fine-tune ──► eval + promotion gate
  (Gemini teacher, FinEE        (Pydantic, class-     (config-driven —      (challenger must beat
   relabeled, smishing            balance guards,       base model is         champion on the
   corpus, template shapes)       hash-manifested)      just a YAML field)    current gate set)
        ▲                                                                  │
        │                                              MLflow registry (Azure ML) ─► GGUF
   retraining trigger                                   champion alias        quantize + verify
        │                                                                  │
        │                                                                  ▼
 Evidently drift jobs ◄── logged traffic ◄── FastAPI + llama.cpp on Azure Container Apps
 + Grafana dashboards        (API-keyed,        (grammar-constrained JSON, scale-to-zero,
                              review-gated       GGUF pulled from Blob at startup)
                              before training)              │
                                                optional: the 0.8B sibling exported
                                                to LiteRT / WebGPU, fully offline
```

- **Reproducibility** — every run traceable to an exact git commit, config YAML, seed, and
  content-hashed dataset manifest; experiments tracked in MLflow on Azure ML's hosted
  tracking server. One versioning owner per layer: git for code, manifests for data,
  MLflow for models.
- **Orchestration** — the whole flow (`generate → validate → train → evaluate → quantize →
  register`) is one config-driven ZenML pipeline with a quality gate that blocks bad
  models from ever registering; the same DAG runs the 0.8B and the 2B, one YAML line apart.
- **Serving** — int-quantized GGUF on `llama-server` behind FastAPI, with
  **grammar-constrained decoding** so the model *cannot* emit schema-invalid JSON.
  Dockerized, deployed to Azure Container Apps with scale-to-zero CPU. API-keyed and
  rate-limited. Pre-committed trade-off rule: if the 2B's warm p95 exceeds budget on
  2 vCPU, the 0.8B ships and the 2B stays the quality reference — decided by measurement.
- **CI/CD** — GitHub Actions (keyless OIDC federated auth to Azure — no stored cloud
  credentials) runs lint/tests/smoke-train on every PR; on merge, the full pipeline
  retrains and registers a challenger; promotion to champion requires beating the
  incumbent on the gate set, then ships blue-green via immutable Container Apps revisions
  with post-deploy smoke tests and one-line traffic rollback.
- **Testing** — one rule: code that *judges* the model is tested like the model itself.
  The eval harness has unit tests against hand-computed fixture metrics (every gate and
  headline number flows through it), a CheckList-style behavioral/invariance suite
  (perturbed amounts, swapped merchants, injected promo noise) runs inside the promotion
  gate, the FastAPI gateway's auth/rate-limit edge cases are unit-tested against a mocked
  backend, and drift-job threshold logic is tested on fixture logs with known outcomes —
  on top of CI's smoke-train, the container integration test, post-deploy smoke tests,
  and k6 load-test thresholds wired in as failing gates.
- **Monitoring & the closed loop** — OpenTelemetry/OTLP metrics pushed to Grafana
  (latency, throughput, category distributions, scam-flag rate — push, because
  scale-to-zero leaves nothing to scrape), per-request LLM-observability traces with a
  schema-validity flag, nightly Evidently drift reports over logged traffic, a scheduled
  traffic simulator that keeps the demo environment live, and a drift alert that triggers
  retraining (Google MLOps maturity Level 1→2) — with logged data entering training only
  after validation **and** human review (data-poisoning is a threat model here, and it's
  documented, not ignored).
- **Optimization study** — measured accuracy/latency/size trade-offs across
  {0.8B, 2B} × {fp16, Q8, Q4}, load tests, $/1M tokens on CPU vs GPU, and the crossover
  point where GPU serving wins (written in Stage 7).

## Design decisions worth reading

Short ADRs live in [docs/decisions.md](docs/decisions.md), including: why the promotion
gate uses a rotating gate set plus a frozen final set (Goodhart's law is real); why the
training teacher and the test-set teacher are different models; why external datasets
contribute texts but not labels; why grammar constraints instead of retry loops; why CPU
serving beats GPU at this scale; why hash manifests instead of DVC; and the retraining
data-poisoning threat model.

## Cost

Runs on an Azure for Students subscription: Container Apps free grant + scale-to-zero,
hosted MLflow via Azure ML, ghcr.io images, Blob storage, Grafana Cloud free tier. Era 2
outgrew two free tiers: Colab Pro (L4; a 2B run is ~2.5 h, the 0.8B the same — the linear-
attention layers run a pure-PyTorch fallback) and a billing-enabled Gemini key (the free
tier's 20 requests/day was the binding constraint for test-set generation and the 26B
baseline). Expected total: **$25–40**.

## Quickstart

```bash
git clone <repo> && cd tinyllm-ops
uv sync          # everything; the Colab notebooks install `-e ".[colab]"` / `".[colab,gen]"`
python run_pipeline.py --config configs/smoke.yaml   # 50-example CPU smoke run (0.8B; entrypoint lands in Stage 3)
```

Live demo: `<endpoint URL>` (first request after idle may take ~20 s — scale-to-zero cold
start). Browser demo (fully offline, WebGPU): `<pages URL>`.

## License

Code: MIT. Fine-tuned weights derive from Qwen3.5 (Apache-2.0); the earlier Gemma 3 270M
adapter remains under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
