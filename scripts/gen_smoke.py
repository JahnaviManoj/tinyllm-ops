"""Half-precision sanity check (tutorial §2.2) — run on the training GPU BEFORE any sweep.

Loads the base model exactly as train.py will (NF4 + the given compute dtype),
generates on 5 SMS through chat_prompt(), and prints the raw text. YOU eyeball it.
Garbage / empty / repeating output ⇒ that dtype is unusable on this GPU: set
compute_dtype: fp32 in the configs and accept ~2× the time. (Era 1's scar: Gemma on
a pre-Ampere card in fp16 produced NaN logits that only LOOKING caught.)

    python scripts/gen_smoke.py data/generated/train_v3.jsonl            # 2B, fp16
    python scripts/gen_smoke.py data/generated/train_v3.jsonl --model Qwen/Qwen3.5-0.8B \
        --revision 2fc06364715b967f1860aea9cf38778875588b17
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tinyllm.prompt import chat_prompt

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--model", default="Qwen/Qwen3.5-2B")
p.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
p.add_argument("--n", type=int, default=5)
a = p.parse_args()

dtype = torch.float16 if a.dtype == "fp16" else torch.float32
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForCausalLM.from_pretrained(
    a.model,
    revision=a.revision,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
    ),
)
print(
    f"{a.model} @ {a.revision[:8]} · NF4 + {a.dtype} · {torch.cuda.get_device_name(0)}"
)
for r in [json.loads(line) for line in open(a.data)][: a.n]:
    ids = tok(chat_prompt(tok, r["sms"]), return_tensors="pt", add_special_tokens=False)
    ids = ids.to(model.device)
    out = model.generate(**ids, max_new_tokens=200, do_sample=False)
    print("SMS:", r["sms"][:80])
    print(
        "OUT:",
        tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True),
        "\n",
    )
