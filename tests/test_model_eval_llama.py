"""The llama-server backend (4.2): same prompt bytes and post-processing as the
HF backend, text fetched over HTTP. A throwaway local HTTP server stands in for
llama-server; no model is loaded."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tinyllm import model_eval
from tinyllm.prompt import build_prompt


@pytest.fixture
def fake_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            reply = json.dumps({"content": 'Sure:\n{"is_transaction": false}\nbye'})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(reply.encode())

        def log_message(self, *_):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_raw_prompt_is_posted_and_json_extracted(fake_server):
    url, seen = fake_server

    out = model_eval.predict_llama_server(url, ["Rs.1 debited", "Rs.2 debited"])

    assert out == ['{"is_transaction": false}'] * 2
    path, body = seen[0]
    assert path == "/completion"
    assert body["prompt"] == build_prompt("Rs.1 debited")
    assert body["temperature"] == 0 and body["n_predict"] == 200


def test_chat_prompt_comes_from_the_tokenizer_template(fake_server, monkeypatch):
    url, seen = fake_server

    class FakeTok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert msgs[0]["role"] == "user" and add_generation_prompt
            return "<chat>" + msgs[0]["content"] + "<gen>"

    monkeypatch.setattr(model_eval, "load_tokenizer", lambda d: FakeTok())

    model_eval.predict_llama_server(url, ["Rs.1 debited"], chat=True, tokenizer_dir="x")

    assert (
        seen[0][1]["prompt"]
        == "<chat>" + build_prompt("Rs.1 debited").rstrip() + "<gen>"
    )


def test_chat_without_tokenizer_dir_is_refused(fake_server):
    with pytest.raises(ValueError, match="tokenizer_dir"):
        model_eval.predict_llama_server(fake_server[0], ["x"], chat=True)


def test_eval_model_llama_backend_uses_the_same_scoring(
    tmp_path, fake_server, monkeypatch
):
    url, _ = fake_server
    data = tmp_path / "gate.jsonl"
    data.write_text(
        json.dumps({"sms": "hi", "label": {"is_transaction": False}})
        + "\n"
        + json.dumps(
            {"sms": "Rs.5 debited", "label": {"is_transaction": True, "amount": "5.00"}}
        )
        + "\n"
    )
    monkeypatch.setattr(
        model_eval, "predict_merged", lambda *a, **k: pytest.fail("hf backend used")
    )

    report = model_eval.eval_model(
        "model.gguf",
        str(data),
        backend="llama",
        server_url=url,
        rows_out=str(tmp_path / "rows.json"),
    )

    assert report["exact_match"] == 0.5  # row 1 matches, row 2 does not
    rows = json.loads((tmp_path / "rows.json").read_text())
    assert [r["correct"] for r in rows] == [True, False]


def test_unknown_backend_is_refused(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"sms": "x", "label": {"is_transaction": False}}) + "\n")
    with pytest.raises(ValueError, match="backend"):
        model_eval.eval_model("m", str(data), backend="vllm")
