# Automatic evaluation analysis

This folder contains analysis of the automatic evaluators themselves.  It is
separate from `annotation_analysis/`, which contains human annotation analysis
and human--automatic metric validation.

Run from the repository root:

```powershell
python evaluation_analysis\automatic_evaluator_analysis.py
```

Outputs are written to `evaluation_analysis/output/`.

## Inputs

- `summaries_strict_9/`: summary metadata and the paired source transcripts
  summarized by GPT-OSS-120B, Mistral Large, and Qwen.
- `seval-ex/`: factuality evaluator outputs.
- `relevance/`: legacy input directory containing entailment/keypoint coverage evaluator outputs.
- `coherence/`: coherence evaluator outputs.
- `fluency/`: fluency evaluator outputs.

The script selects the `*_auto_cot` prompt output for coherence and fluency
where a file contains multiple prompt variants.

## Outputs

| File | Contents |
|---|---|
| `automatic_score_panel.csv` | Canonical row-level panel: metric dimension, evaluator, generator, transcript, source type, profile, degradation, and score. |
| `rating_evaluator_output_availability.csv` | Availability of the four coherence and four fluency evaluator folders, including numeric-score counts and unusable outputs. |
| `automatic_scores_by_generation_model.csv` | Mean, SD, median, bootstrap CI, range, and tie rate by evaluator, generation model, and short/long source. |
| `generation_model_adjusted_comparisons.csv` | Generator-model score differences adjusted for degradation profile and source type; cluster-robust inference by transcript and BH-adjusted p-values. |
| `evaluator_model_agreement.csv` | Evaluator-pair Spearman and Kendall association, mean absolute pairwise score difference, signed bias, and BH-adjusted p-values. |
| `degradation_sensitivity.csv` | Score change when each degradation type is present, including a standardized difference. |
| `degradation_sensitivity_heatmap.svg` | Heatmap of standardized degradation effects. Negative blue cells indicate lower scores when a degradation is present. |
| `automatic_metric_spearman_correlation.csv` | Spearman correlation matrix among selected automatic dimensions. |

## Important interpretation note

The generation models summarize the same transcripts, but their assigned
degradation profiles are often different for a given transcript.  Therefore,
the adjusted generation-model comparison controls for profile and source type;
it should be preferred over a naive direct comparison of unadjusted mean
scores.  Results from incomplete evaluator output folders must be interpreted
with their reported `n` values.
