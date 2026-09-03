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

Take **Qwen3.5-2B** — a model small enough to serve on CPU (~1.2 GB at 4-bit) — and
specialize it with **QLoRA fine-tuning** until it turns any transaction SMS into a strict,
schema-valid record:

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

| Model | Exact match (gate set)¹ | Serving cost |
|---|---|---|
| Per-template regex (industry baseline) | X% | ~free |
| Qwen3.5-2B zero-shot | X% | CPU |
| Qwen3.5-0.8B few-shot | X% | CPU |
| Qwen3.5-2B few-shot | X% | CPU |
| Gemma 3 270M fine-tuned (smaller-student comparison) | X% | CPU |
| Qwen3.5-0.8B fine-tuned (efficiency comparison) | X% | CPU |
| **Qwen3.5-2B fine-tuned (this repo)** | **X%** | **CPU, ~$0/mo** |
| Gemma 4 26B few-shot (~13× bigger) | X% | API $$ |

¹ Strict metric: output must parse AND validate against the schema with *every* field
exactly right, on ~154 hand-reviewed messages. X% placeholders fill in as each stage
lands; all runs logged in MLflow — details in [docs/decisions.md](docs/decisions.md).
The multi-size ladder also answers a question most fine-tuning projects skip:
**how much model does this task actually need?**

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
hosted MLflow via Azure ML, ghcr.io images, Blob storage, Grafana Cloud free tier, Gemini
free tier for data generation, Colab free T4 for training (a 2B QLoRA fits a T4
comfortably). The only real-money item is optional GPU retraining on RunPod
(~$0.50–1/run). Expected total: **$10–25**.

## Quickstart

```bash
git clone <repo> && cd tinyllm-ops
uv sync
python run_pipeline.py --config configs/smoke.yaml   # 50-example CPU smoke run (0.8B)
```

Live demo: `<endpoint URL>` (first request after idle may take ~20 s — scale-to-zero cold
start). Browser demo (fully offline, WebGPU): `<pages URL>`.

## License

Code: MIT. Fine-tuned weights derive from Qwen3.5 (Apache-2.0); the earlier Gemma 3 270M
adapter remains under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
