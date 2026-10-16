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

Large input corpora, annotation exports, evaluator run files, model checkpoints,
and generated per-summary text are kept out of Git. Supply the local inputs
described in the analysis READMEs before rerunning those pipelines.

Selected results are included: aggregate human and automatic analysis tables,
the regenerated human-analysis PNG/SVG figures, and evaluator-analysis outputs.
Row-level exports containing transcripts, summaries, annotation text, or
extracted evidence are excluded. The generated figures and aggregate tables are
ready to inspect without rerunning model inference.
