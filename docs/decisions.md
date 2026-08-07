# Decisions

## 2026-07-26 — Template sources vs. OOD eval split (tutorial 1.2)

Whatever seeds the templates in `src/tinyllm/templates.py` becomes
in-distribution for the synthetic generator (1.3), so it can never double as
the "unseen real-world" OOD eval in 1.5. The split is fixed as:

**Template sources (in-distribution, used in 1.2):**

- `akhilnarang/bank-sms-parser` test fixtures (~110 sanitized SMS, 11 Indian banks)
- `YuvrajDube/Bank-SMS-Parser` `samples.json` (~25 SMS, fake values)
- Hand-written shapes: scam/phishing, Hinglish, ATM, wallet
- (Later, if available: my own inbox)

**Reserved for the 1.5 OOD eval — do NOT open until then:**

- `saurabhgupta050890/transaction-sms-parser`
- the Dart `transaction_sms_parser` package
- `SmsParser` (Kotlin)

**Kaggle `engreemali/bank-transactions-sms-datasetss`:** downloaded and
reviewed, but **not used** for templates — its messages are UAE-bank (AED)
shapes and this project targets Indian bank SMS only. It stays gitignored in
`data/raw_shapes/kaggle/` and no message from it may ever be committed
(real user data, unvetted license). Since it did not seed any template, it
also does not contaminate the OOD split.

## 2026-08-03 — Teacher model for synthetic data: Gemini free tier (tutorial 1.3)

Teacher = Gemini free tier (`gemini-3.5-flash-lite`, pinned — a pinned model
id, not "-latest", keeps the dataset reproducible from its manifest). Two
models were ruled out empirically first: the tutorial's `gemini-2.5-flash` is
closed to new accounts as of 2026-08, and `gemini-3.6-flash`'s free tier is
capped at 20 requests/day (a full stratified run needs ~620 calls — that cap
was discovered by the first run dying on 429 RESOURCE_EXHAUSTED). The lite
tier's free quota is 500 requests/day (also learned from a 429), so the full
run spans two calendar days via the generator's per-template checkpoint and
`--resume`; the weaker teacher is acceptable
because generation is template-constrained and schema validation drops bad
rows at ingest.

**Doubled-currency repair (2026-08-03):** the teacher sometimes obeyed the
"amount as INR 1,234.56" style by writing the marker into the amount, doubling
the template's own prefix ("Rs.INR 450") — 847/2895 (29%) of early rows across
all marker pairs (Rs/INR/₹). Fixed three ways: the prompt now forbids doubled
markers, `fix_doubled_currency()` collapses any survivors at ingest (keeping
the template's marker; labels were never affected), and the checkpointed rows
were repaired in place with the same function. `python -m tinyllm.validate`
counts the pattern so a regression shows up in the report.

Train and val are generated as ONE pool and split after dedupe (12% val,
seeded shuffle) rather than as two runs: halves the API calls, and val's job
is overfitting detection on the same distribution. The differently-sourced
sets are 1.4 (different-teacher test set) and 1.5 (real-world OOD) — those
must NOT come from this pool. ToS caveats, noted
consciously: Google may use free-tier prompts for service improvement, and
the free tier carries a competing-model clause. Acceptable for a portfolio
project generating synthetic template-based SMS (no real user data ever hits
the API); revisit before any commercial use.

A framework like **distilabel** (also Bespoke Curator, NVIDIA NeMo Curator)
packages exactly this generate-then-validate pipeline. Hand-rolled here
because for one dataset on one task the framework is more code than the ~100
lines it replaces — the loop in `data_gen.py` mirrors the same standard
pattern (stratified generation → schema validation → MinHash dedup).

## 2026-08-03 — Test set teacher & decorrelation (tutorial 1.4)

Test set teacher = `gemini-3.5-flash` (pinned), deliberately different from
the train teacher (`gemini-3.5-flash-lite`) so teacher quirks don't sit on
both sides of the evaluation. The prompt is also structurally different:
train generation shows the teacher template shapes; test generation shows NO
templates and asks for scenario-driven SMS per channel × direction, so the
test set contains formats the template list never encoded. Test examples
near-duplicating anything in the train pool (MinHash, same threshold as
train dedupe) are dropped — a test item that near-matches a training row is
a freebie, not a measurement.

`test_gen.py` writes `test_unreviewed.jsonl` and refuses to write
`test.jsonl`: that name is reserved for the hand-reviewed set, per the
tutorial's rule that every ground-truth example gets human eyes. The 1.4
teacher also inherits `fix_doubled_currency` ingest repair and the schema
gate; both are data hygiene, not teacher style, so they don't weaken the
different-teacher claim.

## 2026-08-03 — OOD eval source: reserved repos turned out empty (tutorial 1.5)

The plan reserved three parser repos as the OOD source. Fetched and audited
today, they do not deliver: `saurabhgupta050890/transaction-sms-parser` (MIT)
does not commit its real `testCases.json` — only a placeholder xlsx with one
fake row; the Dart `transaction_sms_parser` (MIT) has ~12 short sample strings
in README/example; `KapilYadav-dev/SmsParser` (Kotlin) has NO license file
(tutorial claimed MIT) and no committed fixtures at all, so it is unusable
regardless. A wider search (other repos, gists) surfaces only ~20 more
sanitized shapes with no license. Net: the "hand-label 200-300 fixture
messages" plan is not executable from public fixtures.

**Decision (2026-08-04):** no inbox export is available, so the OOD set is
built from the public scraps by `python -m tinyllm.ood_build` (fetch at eval
time — nothing committed): 27 rows after dedupe, real sender formats with
sanitized values; lowercase-x placeholder digits are filled deterministically
(seed 1005), uppercase masks kept. **Known limitation, stated honestly:** this
is a small format-OOD smoke eval (~27 rows, synthetic values, some literal
MERCHANT placeholders), not the 200-300-message real-inbox eval the plan
wanted. OOD numbers must be reported with that caveat, and upgrading to a
real inbox export remains the standing improvement if one becomes available.
Rows carry label:null so the 1.6 gate rejects the sheet until it is
hand-labeled into data/ood/ood.jsonl. Never train on it.

## 2026-08-04 — Scam spec teacher exception + junk filter (tutorial 1.4)

`gemini-3.5-flash` refused to generate scam SMS under three framings
(defensive-purpose, concrete-details, no-brand-names) — it returned smishing
explainers or [Bank]-placeholder text instead, and the original negative-line
parser happily stored that prose as data (caught by eyeballing the test set:
all 24 "scam" rows were refusal text). Two fixes: `_looks_like_sms()` now
rejects refusals/markdown/meta-prose/placeholder brackets in every negative
path (data_gen and test_gen), and the scam spec alone uses
`gemini-3.1-flash-lite` (SCAM_TEACHER_MODEL), which complies. The exception is
safe for teacher-decorrelation: negative labels are by-construction
(is_transaction=false, is_suspected_scam from the spec), so nothing
label-quality-critical is delegated to the weaker model. Lesson recorded:
count-per-slice checks aren't enough — glance at actual rows per slice.

## 2026-08-04 — AI-assisted hand-review + currency normalization (tutorial 1.4)

Six independent AI audit passes re-derived labels for all 308 test rows and
flagged disagreements. Systematic find: 115 test labels (and 204 train labels)
carried the SMS's literal marker ("Rs.", "₹") as `currency` instead of INR.
Deterministic, so fixed mechanically: `normalize_currency()` at ingest in both
generators, bulk repair of existing files, and a non-ISO-currency note in the
validate report. The remaining 31 judgment flags (category calls, counterparty
= bank/rails/own-wallet instead of the other party, one not-yet-moved refund)
go to human review via `python -m tinyllm.review` — flagged rows first with
proposed diffs, every row still gets human eyes, output is test.jsonl.
AI audit assists the review; it does not replace the human verdict.

**Amount convention (2026-08-04, owner's call):** amounts are decimal strings
with exactly two places ("2000.00"), regardless of how the SMS wrote them.
Enforced by `normalize_amount()` at ingest in both generators, bulk-applied
to existing files (1509 train + 45 test labels reformatted), and checked by
a note in the validate report. Chosen over exact-SMS-digits because the eval
comparator string-matches amounts and normalized form is what a downstream
expense tracker would store.

## 2026-08-05 — Stage 1 Definition of Done met (tutorial 1.7)

train_v1 (3230), val_v1 (440) and test_v1 (308) uploaded to Blob at
content-addressed paths; manifests committed in manifests/. Round-trip
proven: local train.jsonl deleted, re-fetched purely from its manifest,
SHA-256 verified, byte-identical to the original (cmp). Validation report:
100% schema-valid across train/val/test/ood with per-class counts. The OOD
set (25 rows) stays local-only by design (unlicensed-gist rows; see the 1.5
entry). MLflow dataset logging happens per-training-run in Stage 2.

## 2026-08-05 — Gate set vs frozen final set (tutorial 2.3, Goodhart defense)

The 308-row hand-reviewed test set is split 154/154 into **test_gate_v1** and
**test_final_v1** (stratified on is_transaction × is_suspected_scam × channel
× txn_type; seed 23; both published with manifests). Policy:

- **Gate set** (`test_gate.jsonl`, kept locally): the only eval set that
  promotion decisions, prompt tweaks, and model comparisons may use — in
  Stage 2 sweeps and the Stage 3+/6 promotion gate. It is EXPECTED to wear
  out and gets refreshed (gate_v2 from a new different-teacher batch)
  whenever retraining data refreshes.
- **Frozen final set** (test_final_v1): exists only in Blob + manifest; the
  local file is deleted so nothing can evaluate against it casually. Fetched
  and run as close to once as possible — at Stage 2.9 for the headline
  number, and again only at major milestones. Its results are never used to
  choose between models; by the time it runs, the winner is already picked
  on the gate set.
- **OOD set** (`data/ood/ood.jsonl`): orthogonal third axis (distribution
  shift), reported as its own column with its known-limitation caveat;
  touch-rarely by the same logic.
- The parent test_v1 remains published for provenance; its local copy is
  also removed so "the test set" cannot be run as one 308-row blob by
  accident.

Rationale: val protects against the MODEL overfitting during training; this
split protects against the PIPELINE (and its operators) overfitting through
repeated promote/reject decisions across retraining cycles. The split
predates the first baseline number on purpose — it is only credible if it
exists before there is anything to game.

## 2026-08-05 — Regex baseline (tutorial 2.4)

`regex_baseline.py` derives one regex per shape in templates.py automatically
(literals escaped with whitespace-flexible runs, placeholders as capture
groups) — i.e. it IS the per-bank template list a real expense app maintains,
kept honest by construction. Results: **val (seen shapes) 63% match rate** —
and on matched rows amounts/txn_type are ~100% correct, counterparty ~96% —
but category macro-F1 ≈ 0.01 (a regex cannot know SWIGGY is food) and the
36% it misses are the style variations (truncation, Hinglish, reformatted
amounts). **Gate set (unseen real-world formats): 0% match — total failure.**
That generalization cliff is the motivating number for fine-tuning; the
model's job is to beat 63%→0% with graceful degradation instead.

## 2026-08-06 — Zero/few-shot baselines (tutorial 2.5), gate set, 154 rows

Ladder so far (all logged to MLflow experiment "baselines", local mlruns
until 2.8 re-points tracking at Azure ML):

| baseline                       | schema-valid | exact match | scam P/R  |
|--------------------------------|--------------|-------------|-----------|
| regex (2.4)                    | 0% matched   | 0%          | —         |
| zero-shot gemma-3-270m-it      | 0%           | 0%          | 0/0       |
| few-shot gemma-3-270m-it       | 15%          | 1.3%        | 0.33/0.20 |
| few-shot Gemma-4-26B (API)     | 43%          | 34%         | 1.0/1.0   |

Notes: the 270M runs local fp32 (fp16 overflows Gemma on pre-Ampere GPUs —
all-pad outputs; caught in smoke test). Big model is Gemma-4-26B via Gemini
API — deliberately NOT the gemini-3.5-flash family that wrote the test set,
so the big baseline isn't grading its own homework. Few-shot examples come
from train.jsonl only. "schema-valid" is strict: JSON that parses AND
validates against ExpenseRecord (amounts as strings etc.) — the same bar
every ladder rung including the fine-tune is held to. The 2.7 fine-tune's
target: beat 34% exact from a model ~96x its size.
