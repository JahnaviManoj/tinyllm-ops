"""Prove where the loss lands (tutorial §2.3) — MANDATORY once per model family.

Rebuilds the training rows exactly as train.py does (load_from_manifest, chat=True)
and tokenizes them the way TRL 1.12's SFTTrainer does for prompt/completion data:
prompt alone, then prompt+completion, completion mask = everything after the prompt's
token count. If the tokenizer merges tokens across that boundary the mask shifts and
TRL only WARNS (sft_trainer.py, "Mismatch between tokenized prompt…"). This script
makes it loud, and prints the decoded loss region so you can see it is JSON + EOS only.

    python scripts/check_masking.py configs/exp_101.yaml
"""

import sys

from transformers import AutoTokenizer

from tinyllm.config import load_config
from tinyllm.train import load_from_manifest

cfg = load_config(sys.argv[1])
assert cfg.chat_format, "check_masking is for chat_format: true configs"
tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
ds = load_from_manifest(
    cfg.dataset_manifest, limit=200, eos=tok.eos_token, tok=tok, chat=True
)

bad = 0
for i, ex in enumerate(ds):
    p_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    pc_ids = tok(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]
    if pc_ids[: len(p_ids)] != p_ids:
        bad += 1
    if i == 0:
        print("PROMPT (masked, no loss):", repr(ex["prompt"]))
        print("LOSS LANDS ON:", repr(tok.decode(pc_ids[len(p_ids) :])))
        print("  → must be ONLY the JSON followed by", repr(tok.eos_token))
print(f"{len(ds)} rows checked · boundary mismatches: {bad}")
if bad:
    raise SystemExit("MASK SHIFT — do not train until this is 0")
