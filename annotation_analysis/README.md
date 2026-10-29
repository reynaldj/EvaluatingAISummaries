# Thesis analysis pipeline

## ROUGE, multilingual BERTScore, and UniEval comparisons

Install `requirements-baselines.txt` in a Python environment with working
PyTorch, then run:

```powershell
python annotation_analysis/reference_baselines.py
```

On this workstation, the isolated environment uses the working system PyTorch:

```powershell
annotation_analysis/.baseline_venv/Scripts/python.exe annotation_analysis/reference_baselines.py
```

Outputs are in `analysis_output/reference_baselines/`:

- `REPORT.md`: scoring conventions, limitations, and baseline correlations.
- `paired_metric_comparisons.csv`: baseline and proposed metric Spearman/Kendall
  correlations on **identical summaries**, with paired transcript-bootstrap
  confidence intervals for the difference (proposed minus baseline).
- `baseline_metric_validity.csv`: baseline correlations with each human dimension.
- `baseline_scores.csv`: individual scores, ROUGE/BERTScore P/R/F1, input lengths,
  and truncation flags for BERTScore and each UniEval dimension.
- `reference_match_audit.csv`: matching/exclusion status for every human summary.
- `proposed_join_audit.csv`: missing/conflicting saved evaluator scores. These are
  excluded instead of averaged or silently choosing one.
- `baseline_dataset.csv`, `paired_comparison_records.csv`: auditable input and
  comparison cohorts; includes source/reference text and human ratings.
- `baseline_table.tex`: baseline-only correlation table (requires `booktabs`).
- `run_config.json`, `bertscore_cache.json`, `unieval_cache.json`: settings,
  package/model versions, content fingerprints, and resumable scores.

The reference corpus is `resources/summarization_corpus_20230815.csv`. Match on
transcript ID **and** whitespace-normalized source text to disambiguate repeated
IDs. Currently all 67 annotated short transcripts match (201 summaries); the 14
long transcripts have no verified reference in that corpus and are excluded.
No fallback reference is generated. Existing full-dataset evaluator correlations
are left untouched and are **not** compared against these smaller-cohort scores.

ROUGE-1/2/L use Unicode word tokenization, lowercase, no stemming. BERTScore uses
`bert-base-multilingual-cased`, layer 9, `lang=de`, no IDF or baseline rescaling.
UniEval uses `MingZhong/unieval-sum`, original questions and Yes/(Yes+No) scoring,
with German NLTK sentence splitting. Consistency maps to factuality. Relevance
uses the reference; coherence/consistency use the transcript; fluency uses the
summary alone. It is **exploratory German transfer**, not a validated German
UniEval checkpoint; see the English limitation in the original paper.

Default model truncation is recorded rather than hidden: UniEval right-truncates
at 1,024 input tokens, and BERTScore uses its encoder limit. Compare `all_scored`
with `untruncated` sensitivity rows. Longer inputs can lose evidence; removing
them also changes the cohort. No translation or custom retrieval is introduced.

Useful switches: `--prepare-only` (audit without models), `--metrics rouge`
(CPU lexical baseline), `--score-only` (cache without bootstrap analysis),
`--device cpu` / `--device cuda`, `--batch-size 4`, `--bootstrap-samples 1000`.
Separate experiments should use `--output-dir` to retain their reports. Reruns
reuse neural scores when dataset/settings fingerprints agree. First neural runs
download public models and NLTK data; all summary scoring is local, without a paid
LLM API. Tests:

```powershell
python -m unittest discover -s annotation_analysis -p test_reference_baselines.py -v
```

To audit saved LLM token usage across the current coherence, fluency, factuality,
and relevance result folders, run:

```powershell
python annotation_analysis/token_usage_analysis.py
```

Open [the token usage report](analysis_output/token_usage/REPORT.md) for model and
method totals. The CSVs include prompt/completion counts, per-record averages,
missing-count flags, and a separate keypoint-extraction breakdown. This covers
all saved inputs in the selected runs, not just annotated inputs. Factuality uses
cumulative counters and requires reconstruction; missing/zero counters and
duplicate records are documented in the report. These are recorded usage totals,
not a complete historical bill. No inference is run by this analysis.

For thesis tables restricted to completed human annotations, including the pilot,
with transcript counts, annotator distribution, overlaps, sources and text lengths, run:

```powershell
python annotation_analysis/data_inventory.py
```

Open [the inventory report](data_inventory/REPORT.md) or
[the HTML tables](data_inventory/report.html). Spreadsheet-compatible CSVs and
per-transcript/per-summary metadata are in `data_inventory/`. `overview.csv` counts
transcripts only; table captions explain the three summaries per transcript.
The script uses the Python standard library and checks summary text identity
independently of shuffled slots. Assigned but unannotated material is excluded.

Run these commands from the repository root after adding completed Label Studio
CSV exports to `main_annotation/Results/`.

```powershell
python annotation_analysis/human_analysis.py
python annotation_analysis/automatic_analysis.py
python annotation_analysis/span_keypoint_analysis.py
python annotation_analysis/pairwise_logit_consensus.py
```

All derived files are written to `annotation_analysis/analysis_output/`. Source
annotations and automatic evaluator results are never modified.

## Scripts

- `human_analysis.py` normalizes human ratings, error spans, and transcript
  importance spans; enriches model/profile metadata from assignment JSON;
  calculates rating IAA; builds consensus scores and ranks; and writes human
  descriptive statistics.
- `automatic_analysis.py` ingests the selected factuality, coverage, coherence,
  and fluency outputs. It joins them to summaries using filename, profile, and
  where available model/degradation metadata. Ambiguous or missing joins are
  preserved in `human_automatic_merged.csv` through `join_status` and never
  silently assigned.
- `span_keypoint_analysis.py` compares human error spans to automatic FP
  statements and human transcript-important spans to automatic keypoints. It
  is a token-overlap baseline; inspect `span_keypoint_alignment_details.csv`
  before presenting the result as semantic alignment.
- `pairwise_logit_consensus.py` is the final-analysis companion to
  `pairwise_logit_analysis.py`. It uses consensus ranks derived only from
  explicit human rankings and fits a cluster-robust pairwise logit.

## Primary outputs

| File | Purpose |
|---|---|
| `dataset_composition.csv` | transcript, summary, annotation, and overlap counts |
| `rating_iaa.csv` | ordinal Krippendorff alpha, exact/adjacent agreement |
| `human_scores_aggregated.csv` | one row per transcript and AI summary |
| `human_descriptives.csv` | human scores by source and generation model |
| `metric_validity.csv` | correlations, transcript-cluster-bootstrap CIs and Wald p-values, MAE, and bias; IID p-values are retained for reference |
| `pairwise_metric_agreement.csv` | automatic metric agreement with consensus preference |
| `span_keypoint_alignment_summary.csv` | direct error-span/keypoint comparison summary |
| `pairwise_logit_consensus_results.csv` | odds ratios for drivers of human preference |

## Automatic evaluator inputs

`automatic_analysis.py` discovers the selected factuality, coverage,
coherence, and fluency result files in `seval-ex/`, `relevance/`, `coherence/`,
and `fluency/`. It groups the results by evaluator model; coherence and fluency
are also grouped by prompting strategy, and coverage by keypoint-extraction
strategy. Coherence and fluency use the detailed JSONL files when available so
the generating-model metadata is retained for matching to human scores.

The repository includes the per-summary result files consumed by this analysis,
as well as the aggregate tables. The raw input corpora, human annotation exports,
and generated summaries are not included. To regenerate the tables, provide the
human-analysis input described above and run the evaluator analysis from the
repository root. Model inference is not run by `automatic_analysis.py`.

For publication-quality results, manually audit the rows marked `match=True`
and uncertain/non-matching rows in `span_keypoint_alignment_details.csv`.

## Fuzzy localization and positional agreement

**Audit finding:** the initial results are exploratory, not yet validated final
agreement estimates. An AI-assisted audit found residual verdict explanations in
factual-error queries, missed paraphrases and some incorrect source locations or
boundaries. See [the audit report](analysis_output/fuzzy_position_agreement/localization_audit/REPORT.md)
and its per-item review. **The factual-error parser has now been corrected** to
remove any `VERDICT:` suffix (including multiline reasoning and CONTRADICTED),
plus leading bracketed item numbers. It preserves the raw item in `raw_statement`
when loading factuality data. The current fuzzy-position outputs use the corrected
parser; prior scores and the historical audit inputs are preserved in
`analysis_output/fuzzy_position_agreement_before_parser_fix/`. The audit script
uses that archive when present. Other previously generated analysis directories
are not regenerated automatically. The paraphrase and boundary limitations remain.

This separate analysis locates each automatic keypoint in the **original transcript**
and each automatic FP claim in the **original summary**, then compares source token
positions with human highlights. Human annotations are never used to locate items.
There is no exact-match gate, semantic model, LLM call, or Hungarian span pairing.

From the repository root, use a Python environment with pandas, numpy, scipy,
and RapidFuzz. The initial full run used system `python` because Windows blocked
a SciPy DLL in this machine's `.venv`:

```powershell
python -m pip install -r annotation_analysis/requirements-fuzzy.txt
python annotation_analysis/fuzzy_position_agreement.py
```

The default output is `analysis_output/fuzzy_position_agreement/`. Open `report.html`
to inspect the score tables and examples with human/automatic source highlights.

To change the exploratory settings, write a separate run directory:

```powershell
python annotation_analysis/fuzzy_position_agreement.py --min-similarity 70 --min-coverage 0.70 --ambiguity-margin 3 --output-dir annotation_analysis/analysis_output/fuzzy_position_70
```

`--evaluators llama-70b` restricts the run to an evaluator. `--limit-sets 4` makes a
small smoke run (not a representative result). The default `--ambiguity-policy best`
retains one highest-scoring qualifying location per item, even if other occurrences
have similar scores. Positions are unioned across items, so overlaps never count twice.
`--ambiguity-policy all` and `exclude` reproduce the earlier multiple-location or
ambiguity-exclusion policies. No policy consults human highlights. The preceding corrected-parser results
are archived in `analysis_output/fuzzy_position_agreement_before_multiple_locations/`.

### Localization definition

- Unicode word tokens retain original character offsets. Casefolded words joined
  with spaces form the fuzzy-search representation; punctuation is ignored for
  searching but the returned quotation is always sliced from the original text.
- RapidFuzz `partial_ratio_alignment` searches overlapping windows of three times
  the query's token length (minimum 12 words), stepping half the query length.
  This crosses sentence boundaries. It is a heuristic search, not an exhaustive
  guarantee of the global best span; very long paraphrases may exceed a window.
- Candidate boundaries expand to full source words. The complete query is rescored
  with `fuzz.ratio` against the candidate, so a matching query fragment cannot alone
  obtain a perfect final similarity. Similarity is on a 0–100 scale.
- **Query character coverage** is the longest common subsequence length of the
  query and candidate's casefolded non-space word characters, divided by the query's
  character count. It measures shared in-order characters, not semantic coverage or
  the proportion of important facts. It is related to, not independent of, similarity.
- Alternate boundaries overlapping at least half of the shorter span are grouped
  as one occurrence. Only the highest-scoring qualifying occurrence is retained by
  default. This does not discover complementary evidence fragments that fail thresholds
  individually, nor does it guarantee that repeated passages express identical meanings.
- Defaults are similarity **65**, coverage **0.65**, and selection policy **best**.
  The ambiguity margin **3** applies only to the optional `all`/`exclude` policies. These
  are starting settings, not validated thresholds. Audit localization quality before
  reporting findings; do not tune thresholds to maximize human-highlight agreement.

### Scoring and inputs

The source texts come directly from `main_annotation/Results/*.csv`, preserving
annotation coordinates. Every human span must reproduce its quoted text; verified
UTF-16 offsets are converted when necessary. Any invalid span excludes its entire
document/annotator comparison rather than silently dropping that highlight.

Summary A/B/C slots are shuffled across some annotator exports. Documents are
therefore identified by transcript identity and **actual summary text**, not by slot.
Model/profile metadata is resolved against the same text in assignment files; import
copies without metadata are ignored. Missing or conflicting metadata remains visible
in `human_document_metadata.csv`. Original export slots remain in document provenance.
Exact text identity is used only to join documents and metadata, never as a gate for
localizing extracted keypoints or claims.

The automatic configurations follow `span_evaluator_agreement.py`: four factuality
evaluators and narrow/wide/statistical relevance evaluations. Coverage uses **all
`key_items[].item`**, including uncovered items, from the entailment results. It
does not read arbitrary `keypoints_*.json` files or measure summary completeness.
FP claims come from `verdict_reasoning.FP` through the existing loader. Joins use
filename, profile, generating model, and long/short source. Join failures remain in
`automatic_join_audit.csv`; missing item lists are reported in `input_issues.csv`.

Identical keypoint sets reused across summaries count once per transcript,
evaluator, and keypoint type. Distinct sets are retained with their provenance.
Each set is compared with each available human annotator independently.

Each source word with positive overlap with a selected character span is marked.
Overlapping automatic spans are unioned, as are human spans. For those position sets
`A` and `H`: precision = `|A & H| / |A|`, recall = `|A & H| / |H|`, and F1 =
`2 * |A & H| / (|A| + |H|)`. Empty denominators give missing precision/recall;
both-empty F1 is missing, while one-sided-empty F1 is zero. Pooled micro scores use
token totals. Macro F1 first averages sets/annotators per transcript, then averages
transcripts; seeded bootstrap CIs resample those transcript means.

**Unlocated items have no positions and cannot contribute token false positives.**
Always report localization rate beside positional F1. These are agreement scores
over accepted localized text, not an end-to-end extraction-accuracy score. Empty
human highlights in exported annotation units are treated as no selected tokens.

| Output | Contents |
|---|---|
| `report.html` | Summary tables and examples with original-text highlights |
| `automatic_localizations.csv` | Every item, best candidate, exact offsets/quote, scores, acceptance and alternatives |
| `localization_summary.csv` | Accepted, multiple-location and weak-match counts/rates |
| `position_agreement_by_unit.csv` | Position metrics per extraction set and annotator |
| `position_agreement_summary.csv` | Per-evaluator/type pooled scores and transcript-averaged F1 with CIs |
| `human_span_validation.csv` | Validation of original human coordinates |
| `human_document_metadata.csv` | Text-based metadata joins and original annotator slots |
| `extraction_sets.csv` | Deduplicated extraction sets and evaluation provenance |
| `source_documents.jsonl` | Original texts, document identifiers, SHA-256 hashes |
| `automatic_join_audit.csv`, `input_issues.csv` | Input coverage and ingestion issues |
| `run_config.json` | Thresholds, algorithm, dependency version and input fingerprints |

Synthetic coordinate, fuzzy-match, ambiguity and scoring checks:

```powershell
python -m unittest discover -s annotation_analysis -p test_fuzzy*.py -v
```

For multiple-location items, `automatic_localizations.csv` retains the best candidate
in the scalar `start`, `end`, and `quote` columns for inspection. **All scored locations**
are in `selected_spans_json`; `selected_span_count` gives their count. Do not reconstruct
the new scores from the scalar best-candidate columns alone. `accepted_items` counts
items, while `accepted_spans` in the per-unit table counts their selected spans before
the union of word positions. The HTML report displays each retained location.

## Fuzzy localization with semantic fallback

Run the existing fuzzy analysis first, then extend its saved results:

```powershell
python -m pip install -r annotation_analysis/requirements-semantic.txt
python annotation_analysis/semantic_fallback_agreement.py
```

The runner reads `analysis_output/fuzzy_position_agreement` and writes to the
separate `analysis_output/fuzzy_semantic_position_agreement` directory. It refuses
to overwrite a completed output run. For another threshold or model, pass a new
`--output-dir`. The saved fuzzy baseline files are preserved. With the default
`--location-policy best`, saved multiple-location items are first reduced to their
highest-scoring location. Every accepted fuzzy item remains accepted, and its best
location is retained unchanged during semantic fallback. The fuzzy comparison
scores are also recomputed with this single-location rule.
Before embedding any text, the script reconstructs and verifies the baseline's
per-unit token counts and scores using its saved source documents and human offsets.

Only items with fuzzy status `below_threshold` and no accepted spans are searched
semantically. Empty items and ambiguities deliberately excluded by a legacy fuzzy
policy are not rescued. All rejected queries for one source document are batched;
the same source passages are encoded once per document. No human annotation enters
the localization module, and no LLM/API calls are made.

The default is the locally cached
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
model, consistent with the project's existing embedding configuration. Its inputs
use `query: ` for extracted items and `passage: ` for source candidates, including
German text. Prefix tokens count toward the model's 512-token input budget, but
never enter source quotations or offsets. Use `--model PATH_OR_ID` and optionally
`--model-revision COMMIT` for another compatible SentenceTransformer. The default
uses CPU and four threads; `--device`, `--threads`, and `--batch-size` are configurable.
Inference defaults to float32. On a supported CPU/GPU, `--dtype bfloat16` enables
reduced-precision inference (recorded in the run configuration and cache key).
Cosine computations use float32 arrays. Small score differences near thresholds
are possible; different precision settings never share cached localizations.
If the model is not cached, `--allow-model-download` downloads it into a model cache
inside the output folder. Otherwise missing weights cause an error; the runner
never silently substitutes TF-IDF or another lexical representation. Multilingual
E5 names automatically select retrieval prefixes; other model names use empty
prefixes. For a renamed local model directory, supply `--query-prefix 'query: '`,
`--passage-prefix 'passage: '`, and an explicit threshold. A model's implicit default
prompt is disabled to avoid adding the prefix twice.

Per-document results are checkpointed in `localization_cache/`. Interrupted runs
can resume using the same command. Cache keys include the source, extracted items,
model/commit, prefixes, threshold, margin, token limit, dependency versions, and
locator code hash. Completed runs still require a new output folder.

### Semantic candidate selection

- Candidates are heuristic sentences, clauses separated by punctuation, and pairs
  of adjacent sentences. Their boundaries snap to words in the original source.
  The splitter is not a linguistic German sentence parser; abbreviations can split.
- Passages longer than the model's tokenizer budget are split into overlapping
  word windows until they fit. Overlength queries are rejected explicitly as
  `semantic_query_too_long`. Nothing is silently truncated before embedding.
- Normalized embeddings are compared by cosine similarity. Candidates overlapping
  at least half of the shorter passage are treated as alternative boundaries of
  the same occurrence, retaining the highest score (shorter passage breaks ties).
- The exploratory acceptance threshold for E5 is **0.85 cosine similarity**. By
  default, only the highest-scoring qualifying occurrence is retained. The optional
  `--location-policy all` uses a **0.03 cosine margin** to retain additional near-best
  qualifying occurrences. Neither setting is calibrated. These scores are not probabilities
  and are not interchangeable with fuzzy similarity or character coverage. E5
  cosine scores are commonly concentrated around 0.7--1.0, so the previous MiniLM
  threshold was not transferred unchanged. The value 0.85 remains provisional,
  not an independently calibrated acceptance rule. Non-E5 models retain the old
  exploratory default of 0.75; `--min-semantic-similarity` overrides either default.
- Each accepted item contributes one selected occurrence by default, with positions
  unioned across items. Items without a qualifying location remain unlocated.
- Fuzzy lexical coverage is not imposed on semantic candidates: doing so would
  reject the paraphrases this fallback is intended to recover.

The same valid document/annotator comparisons, scoring formulas, and human-span
exclusions are used for baseline and hybrid results. However, semantic candidates
often have broader boundaries than fuzzy candidates. Higher recall/F1 alone is
not evidence of better localization, and accepted fuzzy mistakes are not corrected.
Review locations independently, including rejected items, and choose thresholds
on a development sample before reporting confirmatory thesis results.

### Hybrid outputs

For thesis reporting, statistical keypoint extraction is a shared method, not an
evaluator-specific extraction. Run `python annotation_analysis/consolidate_statistical_agreement.py`
to produce `thesis_position_agreement_summary.csv` with one statistical row.
The script matches canonical `statistical_kp_dedup.json` records by filename and
source-transcript hash, reuses identical saved item locations, and localizes any
missing canonical items with the same fuzzy thresholds. It stops if a new item
requires semantic fallback. Each document/annotator pair counts once. Separate
`statistical_shared_*` files record the inputs, localizations, comparisons and
audit; historical evaluator-specific results remain available. This consolidation
does not change evaluator-specific summary-rating correlations.

The active `analysis_output/fuzzy_semantic_position_agreement/` results use one
best location per item. The previous E5 multiple-location results are archived in
`analysis_output/fuzzy_semantic_position_agreement_multiple_locations/`. The saved
fuzzy-only folder retains its historical policy; the active hybrid folder includes
`fuzzy_best_position_agreement_summary.csv` for a comparison using the same
single-location policy on both sides.

To rescore a saved multiple-location run without repeating embedding inference:

```powershell
python annotation_analysis/select_best_localizations.py --source-dir annotation_analysis/analysis_output/fuzzy_semantic_position_agreement_multiple_locations --output-dir annotation_analysis/analysis_output/fuzzy_semantic_position_agreement_best_recomputed
```

This verifies the original scores, retains each accepted item's best saved
candidate, verifies source offsets, and recomputes position scores. Acceptance and
localization rates remain unchanged. `verification.json` records these checks;
`run_config.json` records source hashes and the original embedding configuration.

The usual `position_agreement_summary.csv`, `position_agreement_by_unit.csv`,
`localization_summary.csv`, and `report.html` describe the combined method.
Additional audit information:

- `position_agreement_comparison.csv`: fuzzy and hybrid scores side by side, with
  absolute deltas, using the same evaluation units.
- `automatic_localizations.csv`: `localization_method` (`fuzzy`, `semantic`, or
  `unlocated`), `semantic_attempted`, `semantic_status`, cosine score, all selected
  spans, and original fuzzy status/candidates. Fuzzy `similarity` and
  `query_character_coverage` are blank for semantic candidates rather than mixing
  incompatible score types. `selected_spans_json` is authoritative for scoring.
- `localization_summary.csv`: separate counts of accepted fuzzy items, semantic
  attempts, and accepted semantic items. Each extracted item counts once even
  when it has multiple selected spans.
- `semantic_localization_review.csv`: up to five accepted and five rejected
  fallback items per evaluator/configuration, sampled with seed 42. Includes source
  text, proposed passage, scores, and blank review fields, but no human highlights.
  This is an unreviewed sample, not an automatic accuracy assessment.
- `run_config.json`: model/commit, thresholds, dependency versions, original run
  settings, and baseline input hashes.

Tests (synthetic embeddings; no downloads):

```powershell
python -m unittest discover -s annotation_analysis -p test_semantic_fallback.py -v
```
