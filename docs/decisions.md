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

## 2026-09-06 — R0.4 done: Colab path proven end-to-end (L4); gen_smoke verdict = fp16 viable

**Smoke run.** `configs/exp_100_smoke.yaml` (Qwen3.5-0.8B, 20 steps, 50 rows) completed
on a Colab L4 through the full path: clone → manifest fetch (hash-verified) → QLoRA →
adapter saved + logged → fp32 re-merge. Run `afe1e1e8` (name `exp_100_smoke.yaml`) in `tinyllm-finetune`;
the `git_commit` param is the receipt for which code ran. A FAILED twin `3285025e` is the
torchao merge-step crash (adapter logged, merge died) — kept as evidence, never a result. Not a result; never in a table.

**gen_smoke verdict (Part B cell 4).** Qwen3.5-2B @ 15852e8c, NF4 + fp16, NVIDIA L4:
5/5 generations coherent — well-formed JSON in the model's own schema (```json fences,
its own keys, amounts as floats). No NaN / empty / repeating output. **fp16 is viable
for this model.** Decision: L4/A100 sessions use `compute_dtype: auto` (→ native bf16,
no GradScaler workaround); T4 sessions use `fp16`. fp16 numerics are GPU-independent so
the verdict should transfer, but the pre-registered rule is "verify on the GPU you train
on" — re-run cell 4 (~2 min) if a T4 session ever happens.

What the untuned output shows training must override (useful for reading Stage-2
failures later): markdown fences, free-form keys, float amounts. Our schema wants bare
JSON with `is_transaction`, `txn_type`, `amount` as a decimal *string*, `counterparty`,
`category`, `is_suspected_scam`.

**Traps closed this session (all now enforced by notebook cell 1):**
- Nothing from Era 2 had been pushed; Colab cloned `f481ade` (no `scripts/`, no
  `[colab]` extra, duplicate `compute_dtype` MLflow param) — Trap 9 in practice. Cell 1
  now refuses to continue if the Era-2 files are missing from the clone.
- `AZURE_CLIENT_SECRET` pasted 36/40 chars (the value ends in a period — easy to lose).
  Cell 1 now acquires a real token with `ClientSecretCredential` before anything is spent,
  and the MLflow smoke runs in a subprocess so a bad credential is never cached in-kernel.
- Colab preinstalls `torchao 0.10`; peft ≥ 0.20's availability check *raises* on < 0.16,
  hit only at the merge step (the bnb dispatcher wins during training). Training and the
  adapter upload succeed, then the merge dies. torchao is not in `uv.lock`; cell 1
  uninstalls it. If this ever recurs, the adapter is already safe in MLflow.
- `HF_TOKEN` secret unset → "unauthenticated requests" warning only; 4.55 GB at ~300 MB/s.

**Next:** Part C (Stage 1 data), local, no GPU. C1 (rulebook commit) first — before any
gate_v2 row exists. The notebook idles until train_v4 / val_v3 / test_gate_v2 are published.

## 2026-09-07 — C4: gate_v2/final_v2 candidate batch generated (536 rows, unreviewed)

**What.** `data/generated/gate2_unreviewed.jsonl` — 536 rows from `tinyllm.test_gen`
(teacher `gemini-3.5-flash`, scam spec `gemini-3.1-flash-lite`), deduped (MinHash 0.85)
against an explicit 12-file pool of 21,963 rows: train v1/v2/v3, val v1/v2, test/test_gate/
test_final v1, and the four `data/scam/` candidate files. 0 collisions. Composition: 411
transaction rows (10 channel × type specs), 65 non-scam negatives (OTP/promo/informational),
60 scams. Above the pre-registered ~450–500 because top-ups over-delivered; nothing is
trimmed before review — every row gets a verdict (C5), the 2:1 split happens after (C6).

**Deviations from the plan, all pre-review (they add rows, never select them):**
- `NEG_PROMPT` now says INDIA explicitly. The first pass produced 75% US texts in the promo
  and informational specs (Chase, Citi, Capital One, $ and AED) — the positive prompt named
  Indian banks, the negative prompt never did. Those three specs were discarded and
  regenerated; the scam spec was Indian by description and kept.
- Under-yielding specs were topped up with `scripts/topup_specs.py` (re-asks a spec, appends
  to the `.raw` checkpoint under the same spec_id; `--resume` then re-runs dedupe). First-pass
  shortfalls (ATM × credit 10/36) were flaky teacher responses, not hard specs: every top-up
  came back at ~100%.
- Rs.INR doubled-currency artifacts: 0. Amounts: all decimal strings.

**Trap: free-tier Gemini = 20 `gemini-3.5-flash` requests per DAY** (quotaId
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), and failed 503 attempts count. The
old `_call_teacher` retried 6× with backoff, so one "high demand" storm burned a whole day
twice. Now: a daily-quota 429 exits immediately with the reset time; transient errors get
≤4 attempts at 30/60/120 s; the SDK layer is pinned to 1 attempt. Finished on a
billing-enabled key (project needed the Gemini API enabled first — 403 SERVICE_DISABLED).

**Next:** C5 — AI audit with a family-disjoint model (rule 7), then the human verdict on
all 536 rows.

## 2026-09-07 — Rulebook fix before any gate_v2 verdict: cab rides are `transport`, not `travel`

The committed starter rulebook (ddd5ef0) mapped `ola`, `uber`, `rapido` → `travel`. The
project's own hand-reviewed gold (gate_v1 + final_v1) labels them `transport` 15/15 and the
train_v3 teacher 247/1; `travel` is reserved for intercity — IRCTC, MakeMyTrip, redBus,
airlines (gold 10/11). Left as-is, C7 assembly would have re-labelled ~250 train rows into
`travel` against the gold convention and the gate would have punished the model for
learning the right thing. Fixed to `transport`. Rule 4 (rulebook never mined from
gate/final rows) holds: the discrepancy surfaced as a mechanical-audit flag count, but the
evidence and the decision rest on Era-1 gold and train data only, recorded here before any
gate_v2 row received a human verdict. Taxonomy note for reviewers: transport = intra-city
(cab, auto, metro, fuel); travel = intercity / flights / trains / hotels.

## 2026-09-07 — C5 audit of gate2_unreviewed (536 rows): 78 flagged, two convention calls pending

**Auditor.** Claude (Anthropic) — family-disjoint from the gate teacher (Gemini) and every
ladder model (Gemma/Qwen), per rule 7. Eight parallel passes of ~67 rows, each given the
schema and the conventions settled in Era-1 gold (money-moved test; scam = not-transaction;
transport vs travel; refunds/cashback are credits with the merchant as counterparty; ATM
category = stated purpose else other). Plus a mechanical pass: schema validity, amount /
account_tail present in the SMS, channel cues, debit/credit wording, rulebook agreement.
Flags are proposals; the human verdict on every row is the deliverable (`tinyllm.review`).

**Result:** 78 rows flagged (15%), in `data/generated/gate2_flags.json` (gitignored; ⚑ in
the review tool). Clear errors: 7 — one is_transaction miss (a credit-card payment received,
row 534), two counterparties not in the SMS, four "counterparty is the sending bank" on
salary/PF credits. Everything else is category judgment.

**Two systematic teacher artifacts, to settle ONCE before/while reviewing:**
1. **ATM withdrawals/deposits/reversals with no stated purpose** — 45 rows, categories
   assigned round-robin by the teacher (groceries, health, travel, food… in sequence).
   Convention candidate: `other` unless the SMS states the purpose ("for Rent" → rent).
2. **Salary/PF credits naming no employer** — 5 rows where counterparty = the sending bank
   or the word "Salary". Era-1 gold is inconsistent here (it has "YES BANK" as a salary
   counterparty once). Candidate: counterparty = employer if named, else null.

Also visible: FASTag / toll / parking split across transport, fees, bills_utilities, other
(8 rows) — the transport definition (intra-city incl. tolls, parking, fuel) covers them.

## 2026-09-10 — C5 closed: test_v2 = 533 rows; 175 human verdicts, 358 audit-accepted (deviation recorded)

**What happened.** The owner reviewed rows 1–175 with `tinyllm.review` and then chose to
close the review by applying the C5 audit's proposals to the rest. Recorded plainly: **175 of
533 rows carry a human verdict; 358 were accepted on teacher label + Claude audit.** This
departs from the plan's "your verdict on every row". Why it was judged acceptable: on the
175 reviewed rows the owner made 4 edits, all on rows the audit had already flagged with the
same fix (66 food, 97 other, 118 fees, 141 transport), and none on rows the audit had passed.
`data/generated/test_v2.provenance.json` lists the human-verdict indices, the audit-accepted
indices, every edit and every drop, so the two populations can be compared later.

**Applied (86 rows edited, 3 dropped):**
- ATM withdrawals / deposits / reversals with no stated purpose → `category: other` (45 rows).
  The teacher had rotated categories in sequence with nothing in the SMS to justify them.
  Convention, now settled: category = the stated purpose, else `other`.
- Wallet rows (26) had a wallet name in `account_tail` ("Paytm Wallet") → null. Convention:
  `account_tail` is digits or null.
- Salary / PF credits naming no employer → `counterparty: null` (5 rows). Convention:
  employer if named, else null (Era-1 gold was inconsistent; settled here).
- Tolls, FASTag, parking, cab rides → `transport` (7 rows; transport = intra-city incl.
  tolls/parking/fuel; travel = intercity).
- Category/counterparty fixes on 8 rows (kirana → groceries, cafe/tea stall → food, school
  fee refund → education, Airtel cashback → Airtel / bills_utilities, Amazon Pay cashback →
  shopping, ZEPTOLAB → entertainment, unnamed refund merchant → null / other, "SBI ATM" → "ATM").
- Kept against the rulebook: a Swiggy weekly payout labelled `salary` (gig payout, not food)
  — the rulebook's counterparty-only rule will mislabel this class at C7; noted for assembly.
- Dropped as ambiguous gold: a scam row that reads as a genuine UPI collect warning (326),
  a card-used SMS with "OTP: xxxx" (393), a card payment received whose channel and
  counterparty are unknowable (534).

**Composition:** 410 transactions · 64 non-scam negatives · 59 scams. Next: C6 split 2:1.

## 2026-09-10 — C6: gate_v2 / final_v2 published; gate_v1 retired

`scripts/split_gate_final.py` (seed 24, stratified on is_transaction × is_suspected_scam ×
channel × txn_type) split test_v2 (533) into **test_gate_v2 = 356** and **test_final_v2 = 177**,
disjoint, channels balanced (gate: ATM 55 / UPI 58 / card 57 / netbanking 48 / wallet 56).
Both validate with zero failures. Published by hash: `manifests/test_gate_v2.json`
(sha 0043f34d…) and `manifests/test_final_v2.json` (sha e89636cf…). Frozen-final rule
applied: the local `test_final_v2.jsonl` and the unreviewed working files were deleted —
final_v2 exists only in Blob until the once-only ritual (D5). `test_gate.jsonl` (v1) renamed
`test_gate_v1.retired.jsonl`; gate_v1 is no longer an evaluation target. Provenance of every
gate row is in `data/generated/test_v2.provenance.json` (gitignored, backed by this log).
Order check for rule 4: rulebook commit ddd5ef0 (2026-09-06) predates every gate_v2 file.

## 2026-09-10 — C7/C8: train_v4 + val_v3 assembled and published; Part C complete

**Rulebook guards added before assembly** (from a dry run over train_v3, not from gate rows):
the naive rulebook would have changed 170 train rows, 57 of them for the worse — "SALARY-INDIGO"
→ travel, "SIP in DMart Supermarket folio" → groceries, "Amazon Prime" → shopping. Now:
`PROTECTED_CATEGORIES = {salary, investment}` are never overridden; `amazon prime` →
entertainment (specific rule first); `amazon pay` is a payment rail and is skipped. 7 unit
tests in `tests/test_rulebook.py` pin these plus word boundaries and counterparty-only matching.
Result: 108 corrections, 79 of them shopping → groceries for DMart / JioMart / BigBasket.

**Assembly** (`scripts/assemble_v4.py`, seed 1007): train_v3 (11,682) + Mendeley scam (254) +
scenario scam (219) + Mendeley ham (294) → near-dedup (MinHash 0.85) against train_v3 AND
gate_v2 + final_v2 (fetched into the filter only, then deleted) + OOD + scam holdout → 673 of
767 new rows kept → rulebook → 10% val carve. **train_v4 = 11,120 (scam 4.0%), val_v3 = 1,235.**
Both validate with zero failures and no WARN lines.

**Published (Part C DoD — six manifests):** test_gate_v2 (356), test_final_v2 (177), train_v4
(11,120), val_v3 (1,235), ood_v1, scam_holdout_v1 (60). Rulebook commit ddd5ef0 predates every
gate_v2 artefact. Known caveat carried into Stage 2: Swiggy/Zomato gig-worker payouts are
labelled salary and now protected from the food rule; refunds/cashback from those merchants
still map to food by design.

Publishing note: the 3.4 MB train_v4 upload timed out twice as a single PUT; `manifest.py` now uploads in 1 MB blocks with 120 s connection/read timeouts (fetch likewise). Round-trip fetch of train_v4 is byte-identical.

**Next:** Part D — D1 pre-registration entry (26B few-shot config, prompt policy per rung,
tie rule, seed replicate, success bar, compute_dtype auto on L4) BEFORE exp_101–104 configs.

## 2026-09-10 — D1: Stage-2 (Era 2) pre-registration — written before any config or run exists

Everything below is fixed now so nothing can be tuned to the results. Any departure gets its
own dated entry with the reason, before the departing run.

**Eval set and comparator.** `manifests/test_gate_v2.json` (356 rows, sha 0043f34d…) for every
Stage-2 decision. Metric = exact-match on the full label, with the field breakdown
(schema-valid / amount / txn_type / channel / counterparty / category / scam) reported beside
it so the win mechanism is visible (rule 8). `model_eval --rows-out` for every run; per-row
files logged to MLflow. final_v2 / ood_v1 / scam_holdout_v1 are spent-once: the champion, once.

**Tie rule (rule 1).** A gate difference of < 10 rows is a tie. Ties break by fewer epochs →
lower lr → smaller rank → smaller model. `scripts/mcnemar.py` exact two-sided p reported for
the top pair and for any claimed win.

**Success bar.** 2B fine-tuned beats the 26B few-shot baseline by ≥ 10 rows on gate_v2,
McNemar p reported. Era-1 floor for context: exp_009 (270M) 7.8% gate_v1 exact; 26B few-shot
34% gate_v1 — both to be re-measured on gate_v2 (D3/D4), not carried over.

**Ladder and one prompt policy per rung.**
| rung | model | prompt policy |
|---|---|---|
| regex | `tinyllm.regex_baseline` | as is |
| 2B zero-shot | Qwen/Qwen3.5-2B @ 15852e8c | `baselines.PROMPT` in the model's chat template |
| 0.8B few-shot | Qwen/Qwen3.5-0.8B @ 2fc06364 | same + `FEW_SHOT_SPEC` block |
| 2B few-shot | Qwen/Qwen3.5-2B @ 15852e8c | same + `FEW_SHOT_SPEC` block |
| 270M-ft (Era-1 floor) | exp_009 rebuilt from its MLflow adapter | Era-1 raw prompt, **no** `--chat` (how it was trained) |
| 0.8B-ft / 2B-ft | this sweep | `chat_prompt` via `model_eval --chat` |
| 26B few-shot | `gemma-4-26b-a4b-it` via Gemini API (family-disjoint from the gate teacher) | same as few-local |

**26B / few-shot config (rule 6).** 4 exemplars, pool = `data/generated/train_v4.jsonl`
(fetched by manifest, `--train-pool` passed explicitly), policy = first row matching each
`FEW_SHOT_SPEC` kind in order: (debit, UPI), (credit, salary), (non-transaction non-scam),
(scam); exemplar SMS < 200 chars. Deterministic — no exemplar seed exists.

**Sweep (rule 5).** exp_101–104 on the 2B; all four gate-evaluated; val loss logged but
eliminates nobody. Shared config: `chat_format: true`, `dataset_manifest: train_v4`,
`val_manifest: val_v3`, seed 42, alpha = 2r, target_modules = the 12 Qwen3.5 projections,
epochs 2, batch 4 × accumulation 4 (effective 16, held identical across every run incl. the
seed replicate), `augment.sender_id_frac: 0.3` (corpus rows exempt), gradient checkpointing
on, max_length 640. **`compute_dtype: auto`** — sessions run on an L4 (bf16 native); the
fp16 verdict of 2026-09-06 applies only if a T4 session ever happens.
| run | r | lr |
|---|---|---|
| exp_101 | 16 | 1e-4 |
| exp_102 | 16 | 2e-4 |
| exp_103 | 32 | 1e-4 |
| exp_104 | 32 | 2e-4 |
Refinement: ≤ 2 runs. One is an **exact seed replicate of the leader** (seed 43, nothing
else changed) — |Δgate| between seed-twins is this era's measured noise floor. The other
probes one variable: a 3rd epoch, or lr 5e-5 if 2e-4 diverges. Then the 0.8B with the
winning recipe as exp_111 (and at most exp_112) — a datapoint, not a champion.

**Protocol.** One training run per Colab session, gate-evaluated in the same session, `git_commit`
param as the receipt; adapter always logged; merged model logged (`--log-merged`) for the
champion only. Champion = best gate_v2 subject to the tie rule; then the once-only ritual
(final_v2, ood_v1, scam_holdout_v1) in one session, reported with the ~±7pp error bar on
177 rows, win or lose.
