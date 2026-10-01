"""The gateway (tutorial 4.4): auth, rate limit, the training prompt, a grammar,
a schema check — llama-server only does inference.

  client → POST /parse {"sms": ...} → llama-server /completion (json_schema =
  ExpenseRecord, temperature 0) → Pydantic re-validate → optional rulebook → JSON

Grammar-constrained decoding: llama.cpp masks every token that would break the
JSON grammar compiled from ExpenseRecord's schema, so an invalid key or enum is
impossible by construction. Pydantic still validates the result (a 422 with the
raw text if it somehow fails — Stage 6 counts those).

Limitations, on purpose at this scale: API keys and the per-minute rate limit
live in memory, so they reset on restart and are not shared across replicas.
Documented rather than fixed with Redis.

  API_KEYS     comma-separated keys (required)
  LLAMA_URL    llama-server base URL (default http://127.0.0.1:8081)
  RULEBOOK     "1" → apply tinyllm.rulebook to the validated record (reported as
               its own column in 4.4 step 6; never silently folded in)
  RATE_LIMIT   requests per minute per key (default 30)
  REQUEST_LOG  path of a JSONL file to append one line per request (feeds Stage 6)

  API_KEYS=dev uv run uvicorn serving.app:app --port 8080
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ValidationError

from serving.prompting import render
from tinyllm.baselines import extract_json
from tinyllm.rulebook import apply_rulebook
from tinyllm.schema import ExpenseRecord

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8081")
API_KEYS = {k for k in os.environ.get("API_KEYS", "").split(",") if k}
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "30"))
WINDOW_S = 60
RULEBOOK = os.environ.get("RULEBOOK", "0") == "1"
REQUEST_LOG = os.environ.get("REQUEST_LOG")
N_PREDICT = 200
LLAMA_TIMEOUT_S = 60.0
SCHEMA = ExpenseRecord.model_json_schema()

app = FastAPI(title="tinyllm-ops SMS parser", version="0.1.0")
_requests: dict[str, list[float]] = defaultdict(list)
stats = {
    "requests": 0,
    "ok": 0,
    "schema_fallback": 0,
    "upstream_error": 0,
    "latency_ms_sum": 0.0,
}


class ParseRequest(BaseModel):
    sms: str


def check_key(key: str | None = Header(None, alias="x-api-key")) -> str:
    """Auth + per-key quota, as a dependency so it runs before body validation:
    a bad key is a 401 even when the body is garbage."""
    if key not in API_KEYS:
        raise HTTPException(401, "Invalid API key")
    now = time.time()
    _requests[key] = [t for t in _requests[key] if now - t < WINDOW_S]
    if len(_requests[key]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded")
    _requests[key].append(now)
    return key


async def call_llama(prompt: str) -> str:
    """One completion from llama-server, grammar-constrained to the schema.
    cache_prompt stays off: on this recurrent architecture the server's prompt
    cache snapshots the full state per request (decisions.md 2026-10-01)."""
    async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT_S) as client:
        r = await client.post(
            f"{LLAMA_URL}/completion",
            json={
                "prompt": prompt,
                "json_schema": SCHEMA,
                "temperature": 0,
                "n_predict": N_PREDICT,
                "cache_prompt": False,
            },
        )
        r.raise_for_status()
        return r.json()["content"]


def log_request(entry: dict) -> None:
    if REQUEST_LOG:
        with open(REQUEST_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "rulebook": RULEBOOK}


@app.get("/stats")
def get_stats():
    n = stats["requests"] or 1
    return {**stats, "latency_ms_mean": round(stats["latency_ms_sum"] / n, 1)}


@app.post("/parse", response_model=ExpenseRecord, dependencies=[Depends(check_key)])
async def parse(body: ParseRequest, response: Response):
    stats["requests"] += 1
    t0 = time.time()
    try:
        raw = await call_llama(render(body.sms))
    except httpx.HTTPError as e:
        stats["upstream_error"] += 1
        raise HTTPException(
            502, f"model backend unavailable: {type(e).__name__}"
        ) from e
    try:
        record = ExpenseRecord.model_validate_json(extract_json(raw))
    except ValidationError as e:
        stats["schema_fallback"] += 1
        log_request({"sms": body.sms, "raw": raw, "ok": False, "error": str(e)[:200]})
        raise HTTPException(
            422, {"error": "model output failed schema validation", "raw": raw}
        ) from e
    applied = False
    if RULEBOOK:
        ex, applied = apply_rulebook(
            {"sms": body.sms, "label": record.model_dump(mode="json")}
        )
        record = ExpenseRecord.model_validate(ex["label"])
    latency_ms = (time.time() - t0) * 1000
    stats["ok"] += 1
    stats["latency_ms_sum"] += latency_ms
    response.headers["X-Latency-Ms"] = f"{latency_ms:.0f}"
    response.headers["X-Rulebook-Applied"] = str(applied).lower()
    log_request(
        {
            "sms": body.sms,
            "record": record.model_dump(mode="json"),
            "ok": True,
            "latency_ms": round(latency_ms),
            "rulebook_applied": applied,
        }
    )
    return record
