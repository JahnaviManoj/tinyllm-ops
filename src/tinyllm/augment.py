"""Training-time augmentation (2.9 improvement iteration).

Real Indian SMS arrive with DLT sender-ID prefixes ("VM-HDFCBK:", "JD-SBIINB")
— two route letters, a dash, and a 6-char sender code. The 1.2 templates never
encoded them, which is one reason the first fine-tune misread real-format
negatives (an OTP with a sender prefix became a "transaction"). Prepending
realistic prefixes to a seeded fraction of training SMS teaches the model that
the prefix is routing metadata, not transaction content.

Applied on the fly in train.py (config: augment.sender_id_frac), never to the
stored datasets — manifests stay byte-stable and the augmentation is a
reproducible, logged hyperparameter instead of a hidden data edit.
"""

import random

ROUTES = ["VM", "VK", "JD", "AX", "AD", "BP", "CP", "DM", "JM", "TM"]
SENDERS = [
    "HDFCBK",
    "ICICIB",
    "SBIINB",
    "AXISBK",
    "KOTAKB",
    "IDFCFB",
    "PAYTMB",
    "PHONPE",
    "CANBNK",
    "UNIONB",
    "SBICRD",
    "INDUSB",
    "YESBNK",
    "FEDBNK",
]


def add_sender_id(sms: str, rng: random.Random) -> str:
    prefix = f"{rng.choice(ROUTES)}-{rng.choice(SENDERS)}"
    sep = rng.choice([": ", " : ", ": ", " - "])
    return prefix + sep + sms
