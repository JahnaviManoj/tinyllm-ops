"""The happy path against the REAL container (tutorial 4.7): 20 gate SMS through
the running service, every answer a 200 that validates against the schema.

  docker run -d -p 8080:8080 -e API_KEYS=dev -v $PWD/outputs/exp_111/gguf/Q8_0.gguf:/models/model.gguf tinyllm-ops
  uv run pytest tests/test_integration.py -m integration --url http://localhost:8080
"""

import json
import os

import httpx
import pytest

from tinyllm.registry import GATE_MANIFEST
from tinyllm.schema import ExpenseRecord

pytestmark = pytest.mark.integration
N = 20


@pytest.fixture(scope="module")
def gate_sms():
    local = "data/generated/test_gate_v2.jsonl"
    if not os.path.exists(local) and not os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING"
    ):
        pytest.skip("gate set not local and AZURE_STORAGE_CONNECTION_STRING unset")
    from tinyllm.manifest import fetch_local

    with open(fetch_local(GATE_MANIFEST)) as f:
        rows = [json.loads(line) for line in f]
    return [r["sms"] for r in rows[:N]]


@pytest.fixture(scope="module")
def gateway(request):
    url = request.config.getoption("--url")
    try:
        assert httpx.get(f"{url}/healthz", timeout=5).status_code == 200
    except (httpx.HTTPError, AssertionError) as e:
        pytest.skip(f"no gateway at {url}: {e}")
    return url, {"x-api-key": request.config.getoption("--api-key")}


def test_every_gate_sms_parses_to_a_schema_valid_record(gateway, gate_sms):
    url, headers = gateway
    with httpx.Client(timeout=120) as c:
        for sms in gate_sms:
            r = c.post(f"{url}/parse", json={"sms": sms}, headers=headers)
            assert r.status_code == 200, (sms, r.text)
            ExpenseRecord.model_validate(r.json())  # schema-valid, every time
            assert "x-gen-tok-per-s" in r.headers


def test_wrong_key_is_rejected_by_the_real_service(gateway):
    url, _ = gateway
    assert (
        httpx.post(
            f"{url}/parse", json={"sms": "hi"}, headers={"x-api-key": "nope"}
        ).status_code
        == 401
    )
