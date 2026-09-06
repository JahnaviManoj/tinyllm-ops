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

## 2026-08-07 — MLflow wired to Azure ML (tutorial 2.8)

Tracking URI comes from MLFLOW_TRACKING_URI in .env (MLflow reads the env var
natively) rather than a set_tracking_uri() call with IDs hardcoded in Python.
Auth at runtime is DefaultAzureCredential via the azureml-mlflow plugin, i.e.
whatever `az login` session exists — nothing stored in the repo. Verified by
round-trip: a smoke run logged and read back from the workspace. The three
2.5 baseline runs were re-logged into the workspace with a `relogged_from`
provenance param (metrics identical to the local originals) so the 2.9 sweep
compares against baselines in one system of record. Pre-2.8 local stores
(mlflow.db / mlruns) remain untouched as originals; new runs go to Azure.

## 2026-08-08 — Sweep result, gate-set reality check, and improvement iteration (tutorial 2.9)

**What happened:** the 9-run LoRA sweep completed (rank {8,16,32} x LR
{5e-5,1e-4,2e-4} x epochs {2,4}; all in MLflow). Winner on validation loss:
exp_009 (r32, lr 2e-4, 4 epochs) at 0.416 / 89% token accuracy. Val-loss
lessons: rank and epochs both helped monotonically; lr 2e-4 dominated.

**The reality check:** on the gate set the top-3 scored 7.8% / 3.9% / 0%
exact match — far below the few-shot 26B baseline (34%). Val metrics and
gate metrics disagreed by an order of magnitude.

**Why (from reading raw generations, not guessing):**
1. *Enum indiscipline* — the model invents values like channel="bill
   payment"; schema validation rejects the whole record (~40% of outputs).
2. *Semantic misreads on unfamiliar phrasing* — "Credit Card ... charged"
   labeled credit (it is a debit); the words "Credit Card" pattern-matched.
3. *Negative collapse on realistic formats* — an OTP with a sender-ID prefix
   ("VK-CREAPP: 293810 is your OTP...") became a Rs 293,810 transaction.
   Training SMS never carried sender-ID prefixes; gate SMS (different
   teacher, scenario-driven) do, as do real Indian SMS.
4. *Mechanical bug:* training texts had NO EOS token appended (verified by
   tokenizing a formatted example), so stopping was learned only implicitly.

**The meaning:** this is the val/test split doing exactly its job. Val shares
the training distribution (same teacher, same templates), so val loss
measures "can the student mimic its own teacher" — the answer was yes, and
it was hiding the real question. The gate set, generated by a different
model with no templates, measures generalization — and exposed that the
student memorized template style. The 26B baseline generalizes through
pretraining; a 270M knows only what its 3,230 examples show it. Token
accuracy (89%) was also misleading: getting 9 of 10 tokens right still fails
exact-match when the 10th is a digit or an enum. Proxy metrics select the
shortlist; task metrics pick winners.

**The iteration (gate set used as designed — diagnose, improve, re-gate;
frozen final still untouched):**
1. EOS appended to every training text (train.py).
2. Sender-ID augmentation: a seeded fraction of training SMS get realistic
   "VM-HDFCBK:"-style prefixes at load time (config-driven, on-the-fly, so
   manifests stay pure and the augmentation is a logged hyperparameter).
3. Scenario top-up: ~500 non-template examples + ~100 realistic negatives
   generated by the TRAIN teacher (flash-lite — test-teacher decorrelation
   preserved) with a scenario prompt written to be deliberately different
   from test_gen's wording. Deduped against existing train AND against the
   gate + frozen final sets (the final set is fetched into the dedupe filter
   only — contamination prevention, not evaluation). Published as train_v2;
   val_v1 unchanged.
4. Retrain the winning config on train_v2 as exp_010, re-evaluate on the
   gate set, and only if improved, spend the once-only final + OOD eval.

## 2026-08-09 — Improvement iteration result: exp_010 REGRESSED; ablation running

exp_010 (winning recipe + train_v2 + EOS + sender-ID augmentation) scored
3.2% exact on the gate — WORSE than exp_009's 7.8%, despite a slightly
better val loss (0.411 vs 0.416; val remains blind to what matters).
Autopsy of raw outputs: the model began inventing JSON keys that exist
NOWHERE in any training data ("type_of_query"), dropping required keys, and
using invalid enums — a structural-discipline regression. Two hypotheses
were tested and REJECTED with data: (a) scenario-label noise — scenario
counterparty labels are clean (12/200 bank-name issues), and (b) key-shape
inconsistency — both datasets carry identical 9-key/2-key label shapes.
Attribution is therefore confounded across three simultaneous changes
(new data, EOS, augmentation) — a process lesson in changing one variable
at a time, recorded as such.

**Ablation exp_011** (running): identical recipe + EOS + augmentation but
ORIGINAL train_v1 — isolates the scenario-data variable. Champion will be
argmax(exp_009, exp_010, exp_011) on the gate set; the once-only final+OOD
eval spends only after that. Also noted for perspective: Stage 4 serving
uses grammar-constrained decoding, which makes invalid enums/keys
IMPOSSIBLE at serving time — failure mode #1 is mechanically eliminated in
production regardless of which champion wins; the raw eval numbers here are
deliberately unassisted.

## 2026-08-09 — Stage 2 close: champion, once-only final numbers, and the honest story

Ablation verdict: exp_011 (train_v1 + EOS + augmentation) scored 1.3% — the
EOS-string hypothesis was tested and disproven (the appended string DOES
tokenize to the real EOS id), leaving no mechanical bug. Read honestly,
exp_009 (7.8%), exp_010 (3.2%) and exp_011 (1.3%) differ by a handful of
rows on a 154-row gate: all variants sit in one low band and the iteration
did not produce a better model. Chasing further attribution would be
seed-noise archaeology at ~1 GPU-hour per data point.

Champion by the pre-agreed rule (argmax on gate): **exp_009**. Once-only
frozen-final eval: **3.2% exact** (57% schema-valid; amounts 47%,
is_transaction 49% field-exact). OOD (25 rows, known-limitation set):
**0% exact** (64% schema-valid). The final < gate delta (3.2 vs 7.8) also
says the gate number itself carried variance.

**The honest story, recorded for the README and Stage 9:** the hoped-for
punchline ("fine-tuned 270M beats a 96x model") did NOT materialize — the
26B few-shot baseline (34% on gate) remains far ahead. What this stage
actually proved: (1) the eval discipline WORKED — val loss and token
accuracy said "great model" while gate/final said otherwise, and the split
caught it before anything shipped; (2) a 270M trained on template-derived
synthetic data learns format, not task, and generalization needs
qualitatively better data (real-style diversity), not more epochs or rank;
(3) the improvement path is the platform being built anyway — Stage 4's
grammar-constrained decoding mechanically eliminates the invalid-enum/key
failures (the single largest bucket), and Stage 6's retraining loop with
the promotion gate is exactly the vehicle for data iterations, with 7.8%
as the number every challenger must beat. Champion's merged weights are in
MLflow (run champion-exp_009-artifact).

**Champion artifact note (2026-08-09):** the 1.1GB merged-model upload to the
Azure ML artifact store timed out (free-tier uplink). The artifact of record
is the champion's LoRA adapter (MLflow run champion-exp_009, a few MB) plus
the reconstruction recipe logged as a run param: base gemma-3-270m-it fp32 +
adapter + merge_and_unload reproduces the merged model bit-for-bit given the
pinned base. Proper registry upload happens at the Stage 3 registry step.

## 2026-08-11 — RETRAIN_PLAN Phase 1: FinEE audit (PASS, two adaptations)

Audited Ranjit0034/finee-dataset (Apache 2.0) before spending any quota.
Delivers: 152,519 rows in chat format; after stripping instruction wrappers
(seven phrasing variants, stripped generically) and exact-dedupe, **128,201
unique clean SMS** — ~79% Latin/English + ~21% Hindi/Tamil/Telugu scripts,
sane lengths (p50 127 chars), 0% DLT sender prefixes (our augmentation stays
useful). Cached to data/finee/all_sms.jsonl (gitignored).

Adaptation 1: the card's "2,419 real ICICI rows" are NOT marked in the data
(only field is `messages`), so the planned real-row quarantine + real-eval
reserve is not executable. The main value (text diversity at scale) stands.
Adaptation 2: FinEE's own labels use a different philosophy (an EMI-due
REMINDER is labeled type=debit; under our schema no money moved →
is_transaction=false) — confirms the plan to use their TEXTS only and have
Gemma-4-26B write labels in OUR schema. Their labels are kept per-row as a
cross-check signal, never as training truth.

## 2026-08-11 — RETRAIN Phase 2 decisions: teacher labels over FinEE labels; Latin-script only

**Why the 26B teacher relabels FinEE texts instead of using FinEE's own labels
(or adopting FinEE's schema wholesale):**
1. *Schema gap* — FinEE has no `is_transaction`, no `is_suspected_scam` (a
   headline product feature), no `channel`; amounts are floats; its category
   taxonomy differs (grocery/bills/healthcare/emi/cashback...). Three of our
   nine fields cannot be converted because they were never recorded.
2. *Convention conflict* — FinEE labels an EMI due REMINDER as type=debit;
   under our contract no money moved (is_transaction=false). Training on
   their conventions would institutionalize the exact false-positive class
   our hand-review corrected in the gold sets. Mixed label conventions are
   the exp_010 lesson.
3. *Unvetted quality* — their synthetic labels are grammar-generated and
   nobody has sampled them against anything we trust; teacher labels pass
   our schema gate at ingest and a 100-row human spot-check (Phase 4 gate).
4. *Switching schemas is the expensive path, not the cheap one* — the schema
   is the product contract: adopting theirs would invalidate the eval
   harness + its tests, the hand-reviewed test/gate/final sets (human
   re-review), the OOD labels, every baseline number incl. the 7.8%
   incumbent, and the serve-time grammar design (their "include only if
   found" style = variable-key JSON, literally our failure mode). All that
   to skip ~500 labeling calls — the cheapest step in the plan.
FinEE labels are still used where they overlap (amount, debit/credit) as a
free AGREEMENT cross-check on teacher labels — two imperfect labelers
agreeing beats either alone; disagreements get flagged/dropped.

**Latin-script only (English/Hinglish):** the entire eval suite is
English/Hinglish; training Devanagari/Tamil/Telugu spends label quota on
skills nothing measures and rows the owner cannot spot-check. ~101K
Latin-script rows remain — 10x our need. Multilingual = future work behind
multilingual eval sets.

## 2026-08-12 — RETRAIN Phase 4: spot-check PASSED at 1% error

Owner reviewed all 100 sampled rows (disagreement rows prioritized): 1
teacher error (row dropped), 99 accepted — a 1% error rate against the <10%
gate. In all 16 rows where FinEE's labels disagreed with the teacher on both
amount and type, the teacher was right (FinEE's synthetic labels grab
balances / mislabel reminders — vindicating the relabel decision).
Convention settled during review: for cashback/rewards credits, counterparty
= the granting merchant/provider as named in the SMS (e.g. "Paytm"), not
null. Teacher labels are cleared for training use.

## 2026-08-12 — v3 poisoning post-mortem: 63 bank STATEMENTS in the SMS pool

First v3 sweep run (exp_012) collapsed on the gate: 12% parse, 0% exact,
hallucinated keys ("is_statement"!), pretty-printed JSON — while showing 85%
teacher-forced val accuracy. Diagnosis ruled out, in order: the merge
(unmerged adapter equally broken), TRL's processed sequences (decoded and
verified byte-correct incl. single EOS — also retroactively exonerating the
2.9 EOS hypothesis), label key shapes and key ORDER (uniform in v1 and v3),
and overtraining (checkpoint-500 already broken). The tell was the adapter
emitting degenerate junk on its own training base: **63 FinEE rows are not
SMS but full ASCII-art ACCOUNT STATEMENTS** (400+ chars, ====== runs) that
passed the harvest filters (which checked scripts and minimum length, never
maximum or repetition). Degenerate repeated-token sequences are gradient
poison for a small LoRA; 63 rows x 4 epochs sufficed. The hallucinated
"is_statement" key was literally the model blending the statement documents
into the schema.

Fix: assemble_v3 junk guard — SMS must be <=320 chars with no 10+ repeated-
char run (drops the 63). v3/val_v2 rebuilt and republished (same manifest
names, new hashes; the poisoned build never produced a champion). Lesson
appended to the running list: validate INPUTS against the physical medium
("an SMS is short by definition"), not just labels against the schema.

## 2026-08-13 — Iteration-1 verdict: v3 fixes mechanics, not judgment; paused for iteration 2

Cleaned-v3 sweep results (gate): exp_012 3.9% exact (parse 75%, amounts 62%,
is_transaction 62% — all majorly up from exp_009's 60/48/49) but category 19%
/ counterparty 34% unchanged and scam recall 0 (94 scam rows = 0.8% of train,
class starvation). exp_013 (lr 1e-4) 1.9% — undertrained, 2e-4 dominance
reconfirmed. exp_014 (r16) stopped at ~45% by owner call: lowest-information
arm. **exp_009 (7.8%) remains champion; the frozen final was NOT touched.**
The iteration-2 data-quality plan (scam injection from verified smishing
corpora, real negative texture, merchant→category rulebook, class-balance
guards in validate.py, val rebalance) is in RETRAIN_PLAN.md. One-shot
pipeline scripts (finee_select/finee_label/spotcheck/assemble_v3) deleted in
cleanup; their intermediate DATA is kept in data/finee/ for iteration 2.
Process lesson, learned twice and now generalized (1.4 scam refusals; FinEE):
**search for existing datasets before engineering around generation.**

## 2026-09-03 — Era 2 reboot (R0): plan v2, and Qwen3.5 checkpoints verified & pinned

**The reboot decision:** Stage-2 evidence says the 270M's failures are capability
failures (judgment fields flat across three data iterations), so Era 2 swaps the
student for Qwen3.5-2B (main) + Qwen3.5-0.8B (efficiency comparison) and keeps
everything else — schema, data pipeline, eval discipline, platform stages. Full
plan in project_info/PLAN_V2.md; a ChatGPT-proposed from-scratch restart was
evaluated and rejected (mostly re-derived this design; its dataset suggestions
were worse than our audited sources). Era-1 docs archived to project_info/archive/.
Eval-era rule: gate_v1 is worn (~15 decisions) and final_v1/OOD are spent, so Era 2
builds gate_v2/final_v2 and never mixes numbers across gate versions.

**R0.3 checkpoint verification (empirical, transformers 5.16.1 from the lockfile):**

- Pins (from the HF API, 2026-09-03): `Qwen/Qwen3.5-2B` @
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, `Qwen/Qwen3.5-0.8B` @
  `2fc06364715b967f1860aea9cf38778875588b17` (both last modified 2026-03-02).
  License apache-2.0, ungated — no HF license-acceptance step (unlike Gemma).
- **There is no text-only instruct trim at these sizes.** The post-trained
  checkpoints ARE multimodal (`Qwen3_5ForConditionalGeneration`,
  pipeline image-text-to-text; the -Base repos are multimodal too). Non-issue in
  practice, verified by loading: `AutoModelForCausalLM` maps model_type `qwen3_5`
  → `Qwen3_5ForCausalLM`, i.e. the text tower only — train.py's existing loader
  works unchanged and vision weights are never touched.
- Chat template present (ChatML), EOS `<|im_end|>`. The template pre-fills an
  EMPTY `<think>…</think>` block at the start of the assistant turn (non-thinking
  default). Consequence: training prompts must be built with
  `apply_chat_template(add_generation_prompt=True)` and the completion starts
  AFTER the think block — prompt bytes now come from the tokenizer, not a literal
  string, and train/eval/serve must all go through the same helper.
- Hybrid attention: only every 4th layer is full attention (`q/k/v/o_proj`);
  the rest are GatedDeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_a`,
  `in_proj_b`, `out_proj`); MLP is `gate/up/down_proj`. Verified from the
  modeling source. Era-1's Gemma target list would silently LoRA only every 4th
  layer — target_modules must use the 12-name list.
- GGUF/llama.cpp support confirmed viable: official-community Qwen3.5-2B GGUFs
  exist with 100Ks of downloads (bartowski, unsloth, lmstudio-community) —
  Stage 4's serving path is derisked.
- Dependency floors already in pyproject (transformers>=5.14.1, trl>=1.9.0,
  peft>=0.19.1, bitsandbytes>=0.49.2, accelerate>=1.14.0); pyproject/uv.lock
  changes pending the owner's commit.

## 2026-09-03 — Era-2 adversarial review: pre-registered decision rules

Three independent adversarial review passes (code-correctness vs repo+installed
libs, ML methodology, cross-document consistency) ran against PLAN_V2/TUTORIAL_V2;
both docs corrected same-day. The binding methodology rules, pre-registered here
BEFORE any Era-2 training or baseline run:

1. **Tie rule:** on the gate set, a difference of <10 rows is a tie (≈3pp on a
   ~300-row gate — inside binomial noise; Era 1 argmax'd differences of a handful
   of rows). Ties break by: fewer epochs → lower lr → smaller rank → smaller
   model. model_eval stores per-row correctness; scripts/mcnemar.py reports the
   exact p-value with any claimed win. Success bar: 2B-ft beats 26B few-shot by
   ≥10 rows on gate_v2.
2. **Gate sizing:** ~450–500 generated rows, split 2:1 gate:final (≈300/≈150);
   scam spec raised 20→~60 so scam recall stops moving in 10pp steps (gate_v1 had
   10 scam rows). Review budget ~3–4 h.
3. **Real-scam holdout:** ~60 finance-flavored Mendeley smishing rows reserved as
   an eval-only slice (scam_holdout), never trained — otherwise every real scam
   text is spent on training and scam recall is only ever measured on synthetic
   scams. Spent-once alongside final_v2/OOD.
4. **Rulebook ordering + scope:** rulebook.py committed BEFORE the first gate_v2
   row is reviewed (git history enforces "never mined from gate/final"); patterns
   word-bounded both sides (the draft's \blic matched "license"); corrections
   apply only when the matched merchant IS the labeled counterparty; at Stage 4
   the guard is reported as a separate column, never silently folded in.
5. **Sweep:** all 4 primary runs gate-evaluated (no val-loss shortlisting — val
   was proven blind in Era 1); one refinement run is an exact seed replicate of
   the leader, sizing the era's noise floor for rule 1.
6. **Pre-registration of the bar:** the 26B few-shot config (shot count, exemplar
   pool=train_v4, exemplar seed) and one prompt policy per rung get recorded here
   before Stage-2 runs; not tunable afterwards.
7. **Audit-model rule:** gate_v2's AI label audits use a model family disjoint
   from the gate teacher (Gemini) and all ladder models (Gemma/Qwen), on a
   no-training endpoint.
8. **Known circularity, stated up front:** the 26B baseline also wrote most of
   train_v4's labels; every ladder rung therefore reports the field breakdown
   (schema-valid vs judgment fields) next to exact-match so the win mechanism is
   visible. final_v2 is same-distribution as gate_v2; generalization claims rest
   on OOD + scam_holdout.
9. **Dedupe correction:** train_v3 ⊉ train_v1 (445 v1 texts absent — verified);
   the gate_v2 dedupe pool lists train_v1, val_v1, val_v2, train_v3 and all
   candidate files explicitly. Corpus-sourced (UK) rows are exempt from sender-ID
   augmentation via a source field.
