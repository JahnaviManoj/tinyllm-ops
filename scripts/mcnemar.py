"""Tie-rule arbiter (tutorial §2.7): exact two-sided McNemar on two --rows-out files.

python scripts/mcnemar.py outputs/exp_101/gate2_rows.json outputs/exp_103/gate2_rows.json
"""

import json
import sys
from math import comb

a = {r["sms"]: r["correct"] for r in json.load(open(sys.argv[1]))}
b = {r["sms"]: r["correct"] for r in json.load(open(sys.argv[2]))}
keys = a.keys() & b.keys()
x = sum(1 for k in keys if a[k] and not b[k])  # A right, B wrong
y = sum(1 for k in keys if b[k] and not a[k])  # B right, A wrong
n = x + y
p = min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2**n) if n else 1.0
print(
    f"paired rows {len(keys)} · A-only {x} · B-only {y} · margin {abs(x - y)} rows · exact McNemar p≈{p:.3f}"
)
print(
    "TIE (margin < 10 rows)"
    if abs(x - y) < 10
    else "margin ≥ 10 rows — a real difference"
)
