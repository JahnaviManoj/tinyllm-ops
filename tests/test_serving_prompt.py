"""serving.prompting.render must be byte-identical to the training prompt
(tutorial 4.4): train/serve skew now includes the chat template."""

import os

import pytest

from tinyllm.prompt import chat_prompt

LOCAL = "outputs/exp_111/merged"
SMS = [
    "Rs.450.00 debited from A/c XX1234 to VPA swiggy@ybl via UPI. Ref 6234. -HDFC Bank",
    "Dear customer,\nINR 12,500.00 credited to A/c 9876\nSALARY SEP. -SBI",
    "₹1,999 spent on Card 4412 at AMAZON PAY on 29-09-26. Not you? Call 1800. -ICICI",
]


@pytest.fixture(scope="module")
def tok(monkeypatch_module=None):
    if os.path.isdir(LOCAL):
        os.environ["TOKENIZER"] = LOCAL
    try:
        from serving.prompting import tokenizer

        return tokenizer()
    except OSError as e:  # no local merged dir and no HF cache / network
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.parametrize("sms", SMS)
def test_render_equals_training_prompt_byte_for_byte(tok, sms):
    from serving.prompting import render

    assert render(sms) == chat_prompt(tok, sms)


def test_render_ends_with_the_assistant_header(tok):
    from serving.prompting import render

    out = render(SMS[0])
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Parse this bank SMS into JSON." in out
