"""The gateway's plumbing with the model faked (tutorial 4.7, laptop-fast):
wrong key → 401, quota → 429, bad body → 422, bad model output → 422, backend
down → clean 502, happy path → schema-valid record, rulebook flag → applied."""

import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from tinyllm.schema import ExpenseRecord

KEY = "test-key"
GOOD = json.dumps(
    {
        "is_transaction": True,
        "txn_type": "debit",
        "amount": "450.00",
        "currency": "INR",
        "counterparty": "swiggy@ybl",
        "account_tail": "1234",
        "channel": "UPI",
        "category": "other",
        "is_suspected_scam": False,
    }
)


@pytest.fixture
def gateway(monkeypatch):
    """A fresh app with env read at import, the prompt renderer and llama call faked."""
    monkeypatch.setenv("API_KEYS", f"{KEY},other")
    monkeypatch.setenv("RATE_LIMIT", "5")
    monkeypatch.delenv("RULEBOOK", raising=False)
    monkeypatch.delenv("REQUEST_LOG", raising=False)
    import serving.app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "render", lambda sms: f"<prompt>{sms}")

    async def fake_llama(prompt):
        return app_module.FAKE_REPLY

    app_module.FAKE_REPLY = GOOD
    monkeypatch.setattr(app_module, "call_llama", fake_llama)
    return app_module


def client(mod):
    return TestClient(mod.app)


def test_missing_or_wrong_api_key_is_401(gateway):
    c = client(gateway)
    assert c.post("/parse", json={"sms": "hi"}).status_code == 401
    assert (
        c.post("/parse", json={"sms": "hi"}, headers={"x-api-key": "nope"}).status_code
        == 401
    )


def test_happy_path_returns_a_schema_valid_record(gateway):
    r = client(gateway).post(
        "/parse", json={"sms": "Rs.450 debited"}, headers={"x-api-key": KEY}
    )

    assert r.status_code == 200
    rec = ExpenseRecord.model_validate(r.json())
    assert rec.amount == "450.00" and rec.channel.value == "UPI"
    assert r.headers["x-rulebook-applied"] == "false"
    assert int(r.headers["x-latency-ms"]) >= 0


def test_rate_limit_is_429_after_the_quota(gateway):
    c = client(gateway)
    for _ in range(5):
        assert (
            c.post("/parse", json={"sms": "hi"}, headers={"x-api-key": KEY}).status_code
            == 200
        )
    assert (
        c.post("/parse", json={"sms": "hi"}, headers={"x-api-key": KEY}).status_code
        == 429
    )
    # another key has its own quota
    assert (
        c.post("/parse", json={"sms": "hi"}, headers={"x-api-key": "other"}).status_code
        == 200
    )


def test_malformed_body_is_422(gateway):
    r = client(gateway).post(
        "/parse", json={"wrong_field": 1}, headers={"x-api-key": KEY}
    )
    assert r.status_code == 422


def test_model_output_failing_the_schema_is_422_with_raw_text(gateway):
    gateway.FAKE_REPLY = '{"is_transaction": "yes", "category": "crypto"}'
    r = client(gateway).post("/parse", json={"sms": "hi"}, headers={"x-api-key": KEY})

    assert r.status_code == 422
    assert r.json()["detail"]["raw"].startswith('{"is_transaction": "yes"')
    assert gateway.stats["schema_fallback"] == 1


def test_backend_down_is_a_clean_502_not_a_hang(gateway, monkeypatch):
    async def down(prompt):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(gateway, "call_llama", down)
    r = client(gateway).post("/parse", json={"sms": "hi"}, headers={"x-api-key": KEY})

    assert r.status_code == 502
    assert gateway.stats["upstream_error"] == 1


def test_rulebook_flag_corrects_category_and_says_so(monkeypatch, gateway):
    monkeypatch.setenv("RULEBOOK", "1")
    import serving.app as app_module

    mod = importlib.reload(app_module)
    monkeypatch.setattr(mod, "render", lambda sms: sms)

    async def fake(prompt):
        return GOOD  # counterparty swiggy@ybl, category "other"

    monkeypatch.setattr(mod, "call_llama", fake)
    r = client(mod).post("/parse", json={"sms": "x"}, headers={"x-api-key": KEY})

    assert r.status_code == 200
    assert r.json()["category"] == "food"  # swiggy → food
    assert r.headers["x-rulebook-applied"] == "true"


def test_healthz_and_stats(gateway):
    c = client(gateway)
    assert c.get("/healthz").json()["status"] == "ok"
    c.post("/parse", json={"sms": "hi"}, headers={"x-api-key": KEY})
    s = c.get("/stats").json()
    assert s["requests"] == 1 and s["ok"] == 1


def test_bad_key_wins_over_bad_body(gateway):
    r = client(gateway).post(
        "/parse", content=b"not json", headers={"x-api-key": "nope"}
    )
    assert r.status_code == 401
