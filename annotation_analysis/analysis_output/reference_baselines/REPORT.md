# Reference baseline comparisons

Reference audit: {'matched': 67, 'reference_unavailable': 14}. Main cohort: 201 summaries from 67 transcripts.

References require matching transcript ID AND whitespace-normalized transcript text. No references are fabricated for other transcripts. Human ratings come from the existing consensus file. All proposed-metric correlations in paired_metric_comparisons.csv are recomputed on precisely the same summaries as that baseline. Original full-dataset correlations are not used as comparators.

ROUGE uses Unicode word tokenization without stemming; main scores are ROUGE-1/2/L F1. BERTScore uses bert-base-multilingual-cased layer 9 by default, German, no IDF, no rescaling; the exact model hash and settings are stored in the cache. Precision and recall are also saved.

UniEval uses MingZhong/unieval-sum, original English question templates on untranslated German text, German NLTK sentence segmentation, and a mean sentence score for factuality (consistency) and fluency. Coherence and relevance use full-summary inputs. This is exploratory transfer: the original model was developed for English, so poor German results do not establish inferiority on its intended domain.

UniEval uses the official right-truncation policy at 1024 input tokens. BERTScore also has a model input limit. Lengths and truncation flags are saved per summary/dimension; both all_scored and untruncated sensitivity comparisons are reported. Truncation can omit source evidence.

Confidence intervals and cluster-adjusted p-values use 1000 transcript-cluster bootstrap draws (seed 42). Paired correlation differences (proposed minus baseline) share identical resamples. P-values use two-sided Wald tests with the cluster-bootstrap standard error and a t reference distribution with degrees of freedom equal to the number of transcripts minus one; IID Spearman/Kendall p-values are retained in separate *_p_iid columns. P-values are not adjusted for multiple comparisons. Constant/insufficient samples have undefined correlations.

Main file: paired_metric_comparisons.csv. Baseline-only correlations: baseline_metric_validity.csv. Inspect n_summaries and n_transcripts: different proposed methods may have different availability. Reference similarity scores are not direct factual-error detectors or span-localization metrics.

| Dimension | Baseline | n | Spearman | Kendall | Truncated |
|---|---|---:|---:|---:|---:|
| coherence | bertscore_f1 | 201 | 0.424 | 0.318 | 2 |
| coherence | rouge1_f1 | 201 | 0.361 | 0.266 | 0 |
| coherence | rouge2_f1 | 201 | 0.404 | 0.311 | 0 |
| coherence | rougeL_f1 | 201 | 0.400 | 0.294 | 0 |
| coherence | unieval_coherence | 201 | 0.462 | 0.335 | 155 |
| factuality | bertscore_f1 | 201 | 0.232 | 0.165 | 2 |
| factuality | rouge1_f1 | 201 | 0.243 | 0.178 | 0 |
| factuality | rouge2_f1 | 201 | 0.250 | 0.190 | 0 |
| factuality | rougeL_f1 | 201 | 0.263 | 0.194 | 0 |
| factuality | unieval_factuality | 201 | 0.624 | 0.471 | 117 |
| fluency | bertscore_f1 | 201 | 0.475 | 0.369 | 2 |
| fluency | rouge1_f1 | 201 | 0.381 | 0.286 | 0 |
| fluency | rouge2_f1 | 201 | 0.450 | 0.350 | 0 |
| fluency | rougeL_f1 | 201 | 0.440 | 0.333 | 0 |
| fluency | unieval_fluency | 201 | 0.369 | 0.277 | 0 |
| relevance | bertscore_f1 | 201 | 0.343 | 0.248 | 2 |
| relevance | rouge1_f1 | 201 | 0.301 | 0.216 | 0 |
| relevance | rouge2_f1 | 201 | 0.372 | 0.277 | 0 |
| relevance | rougeL_f1 | 201 | 0.323 | 0.229 | 0 |
| relevance | unieval_relevance | 201 | 0.341 | 0.244 | 0 |

## Sources

- https://github.com/google-research/google-research/tree/master/rouge
- https://github.com/Tiiiger/bert_score
- https://github.com/maszhongming/UniEval
- https://aclanthology.org/2022.emnlp-main.131/ (including English-language limitation)
