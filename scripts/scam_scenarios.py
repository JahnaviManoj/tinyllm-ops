"""~200 Indian-idiom scam SMS via the scam-capable teacher (TUTORIAL_V2 C3).

Mendeley is UK-flavoured. The train teacher REFUSES scam generation (the 1.4 precedent in
decisions.md) — only SCAM_TEACHER_MODEL complies. Labels are by construction (every row is
a scam, not a transaction), so the weaker teacher is safe here. Rows carry
source: scenario-in. Candidates still pass dedupe + validation before touching train.

    uv run python scripts/scam_scenarios.py
"""

import json
import random
import re

from dotenv import load_dotenv

from tinyllm.data_gen import _call_teacher, _looks_like_sms, fix_doubled_currency
from tinyllm.test_gen import SCAM_TEACHER_MODEL

KINDS = [
    "fake KYC-expiry SMS threatening account suspension, demanding a link click",
    "fake UPI collect-request / 'money credited by mistake, return it' scam",
    "fake reward-points-expiring SMS for an Indian bank card with a phishing link",
    "fake electricity-disconnection-tonight SMS demanding immediate payment",
    "fake income-tax-refund SMS asking to verify bank details on a link",
    "lottery/lucky-draw win SMS demanding a processing fee via UPI",
]
PROMPT = (
    "Write {n} distinct realistic Indian scam SMS of this kind: {kind}. "
    "One per line, no numbering, no commentary, under 300 chars each. "
    "Use Indian idiom (Rs./INR, UPI, KYC, Indian brand styles, bit.ly-style links)."
)
# Second pass: the link-bearing prompt above yields ~90% bit.ly rows, a shortcut the model
# could learn instead of the scam content (no real transaction SMS carries one). Real Indian
# scams often have no link — the hook is a call-back number, a UPI ID, or "reply YES".
PROMPT_NOLINK = (
    "Write {n} distinct realistic Indian scam SMS of this kind: {kind}. "
    "One per line, no numbering, no commentary, under 300 chars each. "
    "Use Indian idiom (Rs./INR, UPI, KYC, Indian brand styles). Do NOT include any URL, "
    "link or domain — the hook must be a phone number to call back, a UPI ID to pay, "
    "an OTP to share, or an instruction to reply."
)
PASSES = [(PROMPT, 25), (PROMPT_NOLINK, 12)]
OUT = "data/scam/scenario_scam.jsonl"


_MASKED = re.compile(r"X{3,}")


def fill_masked_digits(sms: str, rng: random.Random) -> str:
    """The teacher masks phone numbers as 95555XXXXX in some rows. No real SMS looks like
    that, so it is a learnable shortcut; replace each masked run with random digits."""
    return _MASKED.sub(
        lambda m: "".join(rng.choice("0123456789") for _ in m.group()), sms
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    load_dotenv()  # GEMINI_API_KEY — every teacher call needs it
    rows, seen, rejected = [], set(), 0
    rng = random.Random(1006)
    for prompt, n in PASSES:
        for kind in KINDS:
            before = len(rows)
            text = _call_teacher(
                prompt.format(n=n, kind=kind), model=SCAM_TEACHER_MODEL
            )
            for line in text.splitlines():
                line = line.strip().strip("`").lstrip("-*• ").strip()
                line = re.sub(r"^\d+[.)]\s*", "", line)  # teacher numbered anyway
                if not line:
                    continue
                if not _looks_like_sms(line):
                    rejected += 1
                    continue
                if _norm(line) in seen:
                    continue
                seen.add(_norm(line))
                rows.append(
                    {
                        "sms": fill_masked_digits(fix_doubled_currency(line), rng),
                        "source": "scenario-in",
                        "label": {"is_transaction": False, "is_suspected_scam": True},
                    }
                )
            tag = "link" if prompt is PROMPT else "no-link"
            print(
                f"{tag:7} {kind[:40]:40} +{len(rows) - before:3} → total {len(rows)}",
                flush=True,
            )
    print(f"rejected by _looks_like_sms (refusals/meta/placeholders): {rejected}")
    assert len(rows) > 60, (
        "teacher refused or filter ate everything — inspect before proceeding"
    )
    with open(OUT, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(OUT, len(rows))


if __name__ == "__main__":
    main()
