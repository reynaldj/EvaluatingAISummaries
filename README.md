# Evaluating AI Summaries

Code and selected aggregate results for the German AI-summary evaluation
project. The repository keeps the scripts used to generate summaries and
evaluator scores, run the human and automatic analyses, and produce the
thesis figures.

## Main scripts

- `prompt/process_short_transcripts_with_profiles.py` and
  `prompt/process_long_transcripts_with_profiles.py` generate summaries with
  degradation profiles. The PowerShell runners in `prompt/` run the three
  generation models used in the experiments.
- `create_summaries_strict_9.py` selects the core annotated subset from earlier
  summary runs.
- `seval-ex/german_seval_academic.py`, `relevance/`, `coherence/`, and
  `fluency/` contain the evaluator implementations used to produce factuality,
  coverage, coherence, and fluency scores.
- `annotation_analysis/` creates the human and human--automatic analysis tables.
- `evaluation_analysis/automatic_evaluator_analysis.py` creates evaluator
  comparisons, summary tables, and figures.

The analysis commands and their input layouts are documented in
[`annotation_analysis/README.md`](annotation_analysis/README.md) and
[`evaluation_analysis/README.md`](evaluation_analysis/README.md).

## Setup

Use Python 3.10 or newer. Install `requirements-project.txt`; optional
reference-baseline, fuzzy matching, and semantic alignment analyses have extra
dependencies listed in `annotation_analysis/requirements-*.txt`.

Model-generation and hosted-evaluator scripts require an Academic Cloud API key.
Set `ACADEMIC_CLOUD_API_KEY` in the environment; `.env.example` documents the
variable name. Never commit a real key.

## Data and results

Large input corpora, annotation exports, model checkpoints, and full generated
transcripts/summaries are kept out of Git. Supply the local inputs described in
the analysis READMEs before rerunning model-generation or annotation pipelines.

The repository includes the selected per-summary evaluator result files used by
the current automatic analysis for factuality, coverage, coherence, and fluency,
along with the aggregate analysis tables and figures. It excludes exploratory
and pilot runs, caches, and raw annotation exports. The selected evaluator
results are already-produced model outputs; reproducing inference still requires
the relevant model access and source data.
