"""THE prompt format — single source of truth (tutorial 2.7 / 4.4).

Training (train.py) and serving (Stage 4's FastAPI gateway) must present the
model with byte-identical prompt text. Change the wording, drop the newline,
add a space — and accuracy quietly craters, because the model has only ever
seen one shape: train/serve prompt skew is the LLM version of feature skew.
That is why this string lives here and ONLY here — never inline it elsewhere.
"""

import json

PROMPT = "Parse this bank SMS into JSON.\nSMS: {sms}\nJSON: "


def build_prompt(sms: str) -> str:
    return PROMPT.format(sms=sms)


def format_example(ex: dict) -> dict:
    """SFTTrainer wants one 'text' string per example: prompt + completion.
    ensure_ascii=False so ₹ in labels stays a real character — the serving
    side parses model output as UTF-8, not \\u escapes."""
    return {
        "text": build_prompt(ex["sms"]) + json.dumps(ex["label"], ensure_ascii=False)
    }


def chat_prompt(tok, sms: str) -> str:
    """Chat-template the SAME task text used since era 1 (single source of truth).

    add_generation_prompt=True appends the assistant header INCLUDING Qwen3.5's
    empty <think></think> block — completions start after it. Never hand-write
    ChatML markers; the tokenizer owns its own scaffolding. Train (train.py),
    eval (model_eval.py --chat) and serving must all call this — byte-identical
    prompts or the number measures skew, not the model."""
    return tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(sms).rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
