# TinyLLM Ops 💸📱

**A 270-million-parameter language model, fine-tuned to read bank transaction SMS and flag
scams — wrapped in a production-grade MLOps platform that monitors itself and retrains itself
when the world changes. Total infrastructure cost: under $25.**

> 🚧 **Status: in development.** Built stage-by-stage per [PLAN.md](PLAN.md); this README's
> metric placeholders (X%, Y ms) are filled in as each stage lands.

---

## The problem

Every bank payment in India generates an SMS — RBI mandates it, and UPI alone produces
billions of transactions a month. Expense-tracker apps parse these messages with hand-written
regex, one pattern per bank template, which silently breaks every time a bank reformats its
alerts. Meanwhile, scam texts imitating those same bank alerts are a genuine consumer hazard.
And because financial SMS is exactly the data you *shouldn't* ship to a cloud API, the right
solution has to be small enough to run anywhere — ideally on the phone itself.

## The solution

Take Google's **Gemma 3 270M** — a model small enough to run on a CPU, or in your browser —
and specialize it with **QLoRA fine-tuning** until it turns any transaction SMS into a strict,
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

Specialization beats size on narrow tasks: the goal is a fine-tuned 270M that approaches
frontier-model accuracy on this one job at a fraction of the serving cost — measured against
an honest baseline ladder (regex parser → zero-shot 270M → few-shot 270M → few-shot frontier
model → fine-tuned 270M), evaluated both on held-out data and on a real-world
out-of-distribution set the training generator never saw.

| Model | Exact match (gate set)¹ | OOD accuracy² | Serving cost |
|---|---|---|---|
| Per-template regex (industry baseline) | 0% — matched 0/154 unseen formats³ | fails on unseen templates | ~free |
| Gemma 3 270M zero-shot | 0% (no schema-valid output at all) | ² | CPU |
| Gemma 3 270M few-shot | 1.3% | ² | CPU |
| Gemma 4 26B few-shot (~96× bigger) | 34% (scam P/R 1.0/1.0) | ² | API $$ |
| **Gemma 270M fine-tuned (this repo)** | **7.8%** (frozen final: 3.2%) | **0%** | **CPU, ~$0/mo** |

¹ Strict metric: output must parse AND validate against the schema with *every* field exactly
right, measured on 154 hand-reviewed messages written by a different model than the training
teacher. All runs logged in MLflow; details in [docs/decisions.md](docs/decisions.md).
² OOD column is filled exactly once, when the final model is evaluated — the eval-set
discipline (gate vs frozen vs OOD) exists so no number here can be quietly overfit to.
³ On formats it *has* patterns for, regex extracts amounts/directions near-perfectly — but it
matched none of the 154 out-of-template messages, and it can never infer categories. That
generalization cliff is the reason this project exists.

## Why the engineering is the point

Training the model takes minutes. The project is the **platform around it** — a closed loop
that most fine-tuning demos never build:

```
 synthetic data gen ──► validation gate ──► QLoRA fine-tune ──► eval + promotion gate
  (Gemini, seeded from      (Pydantic,          (Colab T4 /         (challenger must beat
   real template shapes,     hash-manifested)    RunPod, tracked      champion on gate set)
   + scam negatives)                             in MLflow)                │
        ▲                                                                  ▼
        │                                              MLflow registry (Azure ML) ─► GGUF
   retraining trigger                                   champion alias        quantize + verify
        │                                                                  │
        │                                                                  ▼
 Evidently drift jobs ◄── logged traffic ◄── FastAPI + llama.cpp on Azure Container Apps
 + Grafana dashboards        (API-keyed,        (grammar-constrained JSON, scale-to-zero)
                              review-gated                                 │
                              before training)              optional: same model exported
                                                            to LiteRT / WebGPU, fully offline
```

- **Reproducibility** — every run traceable to an exact git commit, config YAML, seed, and
  content-hashed dataset manifest; experiments tracked in MLflow on Azure ML's hosted
  tracking server. One versioning owner per layer: git for code, manifests for data, MLflow
  for models.
- **Orchestration** — the whole flow (`generate → validate → train → evaluate → quantize →
  register`) is one config-driven ZenML pipeline with a quality gate that blocks bad models
  from ever registering.
- **Serving** — int-quantized GGUF on `llama-server` behind FastAPI, with
  **grammar-constrained decoding** so the model *cannot* emit schema-invalid JSON.
  Dockerized, deployed to Azure Container Apps with scale-to-zero — a 270M model needs no
  serving GPU. API-keyed and rate-limited.
- **CI/CD** — GitHub Actions (keyless OIDC federated auth to Azure — no stored cloud
  credentials) runs lint/tests/smoke-train on every PR; on merge, the full pipeline retrains
  and registers a challenger; promotion to champion requires beating the incumbent on the
  gate set, then ships blue-green via immutable Container Apps revisions with post-deploy
  smoke tests and one-line traffic rollback.
- **Testing** — one rule: code that *judges* the model is tested like the model itself. The
  eval harness has unit tests against hand-computed fixture metrics (every gate and headline
  number flows through it), a CheckList-style behavioral/invariance suite (perturbed amounts,
  swapped merchants, injected promo noise) runs inside the promotion gate, the FastAPI
  gateway's auth/rate-limit edge cases are unit-tested against a mocked backend, and the
  drift-job threshold logic is tested on fixture logs with known outcomes — on top of CI's
  lint + unit tests + smoke-train, the container integration test, post-deploy smoke tests,
  and k6 load-test thresholds wired in as failing gates.
- **Monitoring & the closed loop** — OpenTelemetry/OTLP metrics pushed to Grafana (latency,
  throughput, category distributions, scam-flag rate — push, because scale-to-zero leaves
  nothing to scrape), per-request LLM-observability traces with a schema-validity flag,
  nightly Evidently drift reports over logged traffic, a scheduled traffic simulator that
  keeps the demo environment live, and a drift alert that triggers retraining (Google MLOps
  maturity Level 1→2) — with logged data entering training only after validation **and**
  human review (data-poisoning is a threat model here, and it's documented, not ignored).
- **Optimization study** — measured fp16 vs Q8 vs Q4 accuracy/latency/size trade-offs, load
  tests, and a llama.cpp-vs-vLLM cost crossover analysis in [docs/optimization.md]
  (written in Stage 7).

## Design decisions worth reading

Short ADRs live in [docs/decisions.md] (Stage 9), including: why a tiny specialized model
beats a big general one here; why CPU serving beats GPU at this scale; why the promotion gate
uses a rotating gate set plus a frozen final set (Goodhart's law is real); why grammar
constraints instead of retry loops; why hash manifests instead of DVC; and the retraining
data-poisoning threat model.

Full project docs: [PROJECT.md](PROJECT.md) (definition & architecture) ·
[PLAN.md](PLAN.md) (stage-by-stage build checklists) ·
[project_info/RETRAIN_PLAN.md](project_info/RETRAIN_PLAN.md) (the researched plan to close
the Stage 2 generalization gap) ·
[ALTERNATIVES.md](ALTERNATIVES.md) (every tool choice vs the industry-standard alternatives,
verified against 2025–26 practice, with the "why X over Y" for each). The design went through
an adversarial review before building — every finding's resolution is captured in the ADRs.

## Cost

Runs on an Azure for Students subscription: Container Apps free grant + scale-to-zero, hosted
MLflow via Azure ML, ghcr.io images, Blob storage, Grafana Cloud free tier, Gemini free tier
for data generation, Colab free T4 for training. The only real-money item is optional GPU
retraining on RunPod (~$0.10–0.50/run). Expected total: **$10–20**.

## Quickstart

```bash
git clone <repo> && cd tinyllm-ops
uv sync
python run_pipeline.py --config configs/smoke.yaml   # 50-example CPU smoke run
```

Live demo: `<endpoint URL>` (first request after idle may take ~20 s — scale-to-zero cold
start; see the trade-off note in docs). Browser demo (fully offline, WebGPU): `<pages URL>`.

## License

Code: MIT. Fine-tuned weights derive from Gemma 3 270M and are distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) with its prohibited-use policy.
