"""Compare automatic evaluator spans/keypoints with human annotations by evaluator.

Factuality compares evaluator false-positive claims with human error spans in
the same summary. Coverage compares evaluator-extracted source keypoints with
human transcript spans. It measures extraction agreement, not summary coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - semantic dependency is optional outside this analysis
    SentenceTransformer = None

from automatic_analysis import attach_to_humans, load_factuality, load_relevance


FACTUALITY_EVALUATORS = {
    "results-factuality-evaluation-llama8b": "llama-3.1-8b",
    "results-factuality-evaluation-llama70b-correct": "llama-70b",
    "results-factuality-evaluation-gemma-12b": "gemma-12b",
    "results-factuality-evaluation-gemma-31b": "gemma-31b",
}
COVERAGE_EVALUATORS = {
    "gemma12b": "gemma-12b",
    "gemma31b": "gemma-31b",
    "llama-3.1-8b": "llama-3.1-8b",
    "llama-70b": "llama-70b",
}
TOKEN_RE = re.compile(r"[\wÀ-ÿ]+")


def token_iou(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(str(left).lower()))
    right_tokens = set(TOKEN_RE.findall(str(right).lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def match_spans(automatic_texts, human_texts, threshold):
    """Maximize qualifying one-to-one matches, then their total token IoU."""
    if not automatic_texts or not human_texts:
        return []
    similarities = np.array([
        [token_iou(automatic, human) for human in human_texts]
        for automatic in automatic_texts
    ])
    return match_similarity_matrix(similarities, threshold)


def match_similarity_matrix(similarities, threshold):
    if similarities.size == 0:
        return []
    automatic_count, human_count = similarities.shape
    size = automatic_count + human_count
    costs = np.zeros((size, size), dtype=float)
    costs[:automatic_count, :human_count] = 1_000.0
    eligible = similarities >= threshold
    costs[:automatic_count, :human_count][eligible] = -2.0 - similarities[eligible]
    row_indices, column_indices = linear_sum_assignment(costs)
    return [
        (int(auto_index), int(human_index), float(similarities[auto_index, human_index]))
        for auto_index, human_index in zip(row_indices, column_indices)
        if auto_index < automatic_count
        and human_index < human_count
        and eligible[auto_index, human_index]
    ]


class SemanticEncoder:
    """Cache normalized sentence embeddings to avoid re-encoding repeated text."""

    def __init__(self, model_name):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for semantic span agreement")
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.embeddings = {}

    def preload(self, texts, batch_size=128):
        unique_texts = list(dict.fromkeys(str(text) for text in texts if str(text).strip()))
        missing = [text for text in unique_texts if text not in self.embeddings]
        if not missing:
            return
        vectors = self.model.encode(
            missing,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embeddings.update(zip(missing, vectors))

    def similarity_matrix(self, automatic_texts, human_texts):
        texts = list(dict.fromkeys([*automatic_texts, *human_texts]))
        missing = [text for text in texts if text not in self.embeddings]
        if missing:
            vectors = self.model.encode(
                missing,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=128,
                show_progress_bar=False,
            )
            self.embeddings.update(zip(missing, vectors))
        automatic_vectors = np.stack([self.embeddings[text] for text in automatic_texts])
        human_vectors = np.stack([self.embeddings[text] for text in human_texts])
        return cosine_similarity(automatic_vectors, human_vectors)


def match_semantic_spans(automatic_texts, human_texts, threshold, encoder):
    if not automatic_texts or not human_texts:
        return []
    similarities = encoder.similarity_matrix(automatic_texts, human_texts)
    return match_similarity_matrix(similarities, threshold)


def metric_scores(automatic_count, human_count, matches):
    matched_count = len(matches)
    precision = matched_count / automatic_count if automatic_count else np.nan
    recall = matched_count / human_count if human_count else np.nan
    if automatic_count == 0 and human_count == 0:
        f1 = np.nan
    elif precision == 0 or recall == 0 or not np.isfinite([precision, recall]).all():
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluator_rows(factuality_root, coverage_root, human):
    factuality_metrics, factuality_statements = [], []
    for folder_name, evaluator in FACTUALITY_EVALUATORS.items():
        folder = factuality_root / folder_name
        if not folder.is_dir():
            continue
        metrics, statements = load_factuality(folder)
        metrics["evaluator_model"] = evaluator
        statements["evaluator_model"] = evaluator
        factuality_metrics.append(metrics)
        factuality_statements.append(statements)

    relevance_metrics, coverage_keypoints = load_relevance(coverage_root)
    if not relevance_metrics.empty:
        relevance_metrics["evaluator_model"] = relevance_metrics["metric_file"].map(
            lambda value: COVERAGE_EVALUATORS.get(Path(value).parent.parent.name)
        )
        coverage_keypoints["evaluator_model"] = coverage_keypoints["metric_file"].map(
            lambda value: COVERAGE_EVALUATORS.get(Path(value).parent.parent.name)
        )
        relevance_metrics = relevance_metrics[
            relevance_metrics["evaluator_model"].notna()
            & relevance_metrics["keypoint_type"].isin(("narrow", "wide", "statistical"))
        ]
        coverage_keypoints = coverage_keypoints[
            coverage_keypoints["evaluator_model"].notna()
            & coverage_keypoints["keypoint_type"].isin(("narrow", "wide", "statistical"))
        ]

    factuality_metrics = pd.concat(factuality_metrics, ignore_index=True) if factuality_metrics else pd.DataFrame()
    factuality_statements = pd.concat(factuality_statements, ignore_index=True) if factuality_statements else pd.DataFrame()
    if not factuality_metrics.empty:
        factuality_metrics = attach_to_humans(factuality_metrics, human)
        factuality_statements = attach_to_humans(factuality_statements, human)
    if not relevance_metrics.empty:
        relevance_metrics = attach_to_humans(relevance_metrics, human)
        coverage_keypoints = attach_to_humans(coverage_keypoints, human)
    return factuality_metrics, factuality_statements, relevance_metrics, coverage_keypoints


def comparison_rows(metrics, automatic_units, human_ratings, human_spans, *, analysis, text_column, threshold,
                    matcher=match_spans, score_name="token_iou", similarity_label="mean_matched_token_iou",
                    encoder=None):
    rows, matches_out = [], []
    available = metrics.dropna(subset=["item_key", "summary_id", "evaluator_model"])
    for _, evaluation in available.drop_duplicates(
        ["evaluator_model", "item_key", "summary_id", "keypoint_type", "summary_model"]
    ).iterrows():
        item_key = evaluation["item_key"]
        summary_id = evaluation["summary_id"]
        evaluator = evaluation["evaluator_model"]
        keypoint_type = evaluation.get("keypoint_type", "not_applicable")
        summary_model = evaluation.get("summary_model", "")
        annotators = human_ratings.loc[
            human_ratings["item_key"].eq(item_key)
            & human_ratings["summary_id"].eq(summary_id), "annotator"
        ].dropna().unique()
        auto_group = automatic_units.loc[
            automatic_units["evaluator_model"].eq(evaluator)
            & automatic_units["item_key"].eq(item_key)
            & automatic_units["summary_id"].eq(summary_id)
            & automatic_units["keypoint_type"].eq(keypoint_type)
        ]
        if summary_model and "summary_model" in auto_group:
            auto_group = auto_group[auto_group["summary_model"].eq(summary_model)]
        auto_texts = auto_group[text_column].dropna().drop_duplicates().astype(str).tolist()
        for annotator in annotators:
            if analysis == "factuality_error_spans":
                human_group = human_spans.loc[
                    human_spans["item_key"].eq(item_key)
                    & human_spans["summary_id"].eq(summary_id)
                    & human_spans["annotator"].eq(annotator)
                ]
            else:
                human_group = human_spans.loc[
                    human_spans["item_key"].eq(item_key)
                    & human_spans["annotator"].eq(annotator)
                ]
            human_texts = human_group["span_text"].dropna().drop_duplicates().astype(str).tolist()
            if matcher is match_semantic_spans:
                matched = matcher(auto_texts, human_texts, threshold, encoder)
            else:
                matched = matcher(auto_texts, human_texts, threshold)
            precision, recall, f1 = metric_scores(len(auto_texts), len(human_texts), matched)
            row = {
                "analysis": analysis,
                "evaluator_model": evaluator,
                "keypoint_type": keypoint_type,
                "generation_model": summary_model,
                "item_key": item_key,
                "summary_id": summary_id,
                "annotator": annotator,
                "automatic_units": len(auto_texts),
                "human_units": len(human_texts),
                "matched_units": len(matched),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                similarity_label: np.mean([match[2] for match in matched]) if matched else np.nan,
                "threshold": threshold,
            }
            rows.append(row)
            for auto_index, human_index, similarity in matched:
                matches_out.append({
                    **{key: row[key] for key in (
                        "analysis", "evaluator_model", "keypoint_type", "generation_model",
                        "item_key", "summary_id", "annotator", "threshold",
                    )},
                    "auto_index": auto_index,
                    "human_index": human_index,
                    "auto_text": auto_texts[auto_index],
                    "human_span": human_texts[human_index],
                    score_name: similarity,
                })
    return pd.DataFrame(rows), pd.DataFrame(matches_out)


def clustered_ci(frame, n_resamples=1000, seed=42):
    data = frame.dropna(subset=["f1"])
    transcript_keys = data["item_key"].dropna().unique()
    if len(transcript_keys) < 2:
        return np.nan, np.nan
    by_key = {key: group["f1"].to_numpy(float) for key, group in data.groupby("item_key")}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for index in range(n_resamples):
        selected = rng.choice(transcript_keys, size=len(transcript_keys), replace=True)
        estimates[index] = np.concatenate([by_key[key] for key in selected]).mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def summarize(comparisons, similarity_label="mean_matched_token_iou"):
    if comparisons.empty:
        return pd.DataFrame()
    output = []
    group_columns = ["analysis", "evaluator_model", "keypoint_type", "threshold"]
    for keys, group in comparisons.groupby(group_columns, dropna=False):
        auto_total = group["automatic_units"].sum()
        human_total = group["human_units"].sum()
        matched_total = group["matched_units"].sum()
        precision = matched_total / auto_total if auto_total else np.nan
        recall = matched_total / human_total if human_total else np.nan
        micro_f1 = (2 * precision * recall / (precision + recall)
                    if np.isfinite([precision, recall]).all() and precision + recall
                    else 0.0 if auto_total or human_total else np.nan)
        ci_low, ci_high = clustered_ci(group)
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,))) | {
            "generation_model": "ALL",
            "aggregation_level": "all_generation_models",
            "n_comparisons": len(group),
            "n_transcripts": group["item_key"].nunique(),
            "automatic_units": int(auto_total),
            "human_units": int(human_total),
            "matched_units": int(matched_total),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": micro_f1,
            "macro_precision": group["precision"].mean(),
            "macro_recall": group["recall"].mean(),
            "macro_f1": group["f1"].mean(),
            "macro_f1_ci_95_low": ci_low,
            "macro_f1_ci_95_high": ci_high,
            similarity_label: group[similarity_label].mean(),
        }
        output.append(row)
    return pd.DataFrame(output)


def evaluator_model_summary(summary, similarity_label="mean_matched_token_iou"):
    """Keep only pooled evaluator rows in a compact evaluator comparison table."""
    if summary.empty:
        return summary
    pooled = summary.loc[
        summary["aggregation_level"].eq("all_generation_models")
    ].copy()
    columns = [
        "analysis", "evaluator_model", "keypoint_type", "threshold",
        "n_comparisons", "n_transcripts", "automatic_units", "human_units",
        "matched_units", "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
        "macro_f1_ci_95_low", "macro_f1_ci_95_high", similarity_label,
    ]
    return pooled[columns].sort_values(
        ["analysis", "keypoint_type", "threshold", "macro_f1"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path, default=Path("annotation_analysis/analysis_output/human_analysis"))
    parser.add_argument("--factuality-root", type=Path, default=Path("seval-ex"))
    parser.add_argument("--coverage-root", type=Path, default=Path("relevance"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotation_analysis/analysis_output/span_evaluator_agreement"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--semantic-thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80])
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human = pd.read_csv(args.human_dir / "human_scores_aggregated.csv", dtype=str)
    human_ratings = pd.read_csv(args.human_dir / "human_ratings_raw.csv", dtype=str)
    human_errors = pd.read_csv(args.human_dir / "human_error_spans_raw.csv", dtype=str)
    human_transcript = pd.read_csv(args.human_dir / "human_transcript_labels_raw.csv", dtype=str)
    for frame in (human, human_ratings, human_errors, human_transcript):
        for column in ("item_key", "summary_id", "annotator"):
            if column in frame:
                frame[column] = frame[column].fillna("").astype(str)
    human_errors = human_errors.drop_duplicates(["item_key", "summary_id", "annotator", "span_start", "span_end", "span_text"])
    human_transcript = human_transcript.drop_duplicates(["item_key", "annotator", "span_start", "span_end", "span_text"])

    fact_metrics, fact_statements, coverage_metrics, coverage_keypoints = evaluator_rows(
        args.factuality_root, args.coverage_root, human
    )
    outputs, matches = [], []
    for threshold in args.thresholds:
        fact_comparisons, fact_matches = comparison_rows(
            fact_metrics, fact_statements, human_ratings, human_errors,
            analysis="factuality_error_spans", text_column="statement", threshold=threshold,
        )
        coverage_comparisons, coverage_matches = comparison_rows(
            coverage_metrics, coverage_keypoints, human_ratings, human_transcript,
            analysis="coverage_keypoints", text_column="keypoint", threshold=threshold,
        )
        outputs.extend([fact_comparisons, coverage_comparisons])
        matches.extend([fact_matches, coverage_matches])

    comparisons = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    match_details = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    summary = summarize(comparisons)
    evaluator_summary = evaluator_model_summary(summary)
    comparisons.to_csv(args.output_dir / "evaluator_span_agreement_by_unit.csv", index=False)
    summary.to_csv(args.output_dir / "evaluator_span_agreement_summary.csv", index=False)
    evaluator_summary.to_csv(args.output_dir / "evaluator_model_span_f1_summary.csv", index=False)
    match_details.to_csv(args.output_dir / "evaluator_span_agreement_matches.csv", index=False)

    semantic_encoder = SemanticEncoder(args.semantic_model)
    semantic_texts = []
    for frame, column in (
        (fact_statements, "statement"),
        (coverage_keypoints, "keypoint"),
        (human_errors, "span_text"),
        (human_transcript, "span_text"),
    ):
        if column in frame:
            semantic_texts.extend(frame[column].dropna().astype(str).tolist())
    print("Encoding unique automatic and human span texts...")
    semantic_encoder.preload(semantic_texts)
    semantic_outputs, semantic_matches = [], []
    for threshold in args.semantic_thresholds:
        fact_comparisons, fact_matches = comparison_rows(
            fact_metrics, fact_statements, human_ratings, human_errors,
            analysis="factuality_error_spans", text_column="statement", threshold=threshold,
            matcher=match_semantic_spans, score_name="semantic_similarity",
            similarity_label="mean_matched_semantic_similarity", encoder=semantic_encoder,
        )
        coverage_comparisons, coverage_matches = comparison_rows(
            coverage_metrics, coverage_keypoints, human_ratings, human_transcript,
            analysis="coverage_keypoints", text_column="keypoint", threshold=threshold,
            matcher=match_semantic_spans, score_name="semantic_similarity",
            similarity_label="mean_matched_semantic_similarity", encoder=semantic_encoder,
        )
        semantic_outputs.extend([fact_comparisons, coverage_comparisons])
        semantic_matches.extend([fact_matches, coverage_matches])

    semantic_comparisons = pd.concat(semantic_outputs, ignore_index=True) if semantic_outputs else pd.DataFrame()
    semantic_match_details = pd.concat(semantic_matches, ignore_index=True) if semantic_matches else pd.DataFrame()
    semantic_summary = summarize(semantic_comparisons, "mean_matched_semantic_similarity")
    semantic_evaluator_summary = evaluator_model_summary(
        semantic_summary, "mean_matched_semantic_similarity"
    )
    semantic_comparisons.to_csv(args.output_dir / "semantic_evaluator_span_agreement_by_unit.csv", index=False)
    semantic_summary.to_csv(args.output_dir / "semantic_evaluator_span_agreement_summary.csv", index=False)
    semantic_evaluator_summary.to_csv(
        args.output_dir / "semantic_evaluator_model_span_f1_summary.csv", index=False
    )
    semantic_match_details.to_csv(
        args.output_dir / "semantic_evaluator_span_agreement_matches.csv", index=False
    )
    print(f"Wrote {len(comparisons)} evaluator/annotator comparisons and {len(match_details)} matched span pairs.")
    print("Pooled evaluator-level summary:")
    print(evaluator_summary.to_string(index=False))
    print(f"Wrote {len(semantic_comparisons)} semantic comparisons and {len(semantic_match_details)} matched pairs.")
    print("Semantic evaluator-level summary:")
    print(semantic_evaluator_summary.to_string(index=False))


if __name__ == "__main__":
    main()