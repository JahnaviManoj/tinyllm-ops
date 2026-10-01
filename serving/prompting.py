"""The served prompt, byte-identical to training (tutorial 4.4).

exp_111 trains with ``chat_format: true``, so the served bytes must equal
``tinyllm.prompt.chat_prompt(tok, sms)`` — the chat template is part of the
prompt now, and train/serve skew includes it. The tokenizer that owns the
template is loaded once; nothing else from transformers is used, so the
container needs no torch and (with HF_HUB_OFFLINE=1 and baked files) no network.

  TOKENIZER      HF id or local dir (default: the champion's pinned base)
  TOKENIZER_REV  revision for an HF id (ignored for a local dir)
"""

from __future__ import annotations

import os
from functools import lru_cache

from tinyllm.prompt import chat_prompt

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"  # exp_111's base pin


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    name = os.environ.get("TOKENIZER", DEFAULT_TOKENIZER)
    kw = (
        {}
        if os.path.isdir(name)
        else {"revision": os.environ.get("TOKENIZER_REV", DEFAULT_REVISION)}
    )
    return AutoTokenizer.from_pretrained(name, **kw)


def render(sms: str) -> str:
    """The exact prompt string the model saw in training, for this SMS."""
    return chat_prompt(tokenizer(), sms)
