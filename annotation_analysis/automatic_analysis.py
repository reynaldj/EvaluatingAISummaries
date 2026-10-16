"""Ingest selected automatic metrics and compare them with human judgments.

The defaults use the final factuality and coverage configurations selected for
this thesis.  Coherence and fluency use their Llama-3.1-8B detailed outputs.
All inputs remain read-only; derived tables are written to analysis_output.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from human_analysis import DIMENSIONS, normalized_filename


DEFAULTS = {
    "factuality": Path("seval-ex/results-factuality-evaluation-llama8b"),
    "relevance": Path("relevance"),
    "coherence": Path("coherence/coherence_geval_ablation_llama3.1-8b-instruct/coherence_ablation_detailed.csv"),
    "fluency": Path("fluency/fluency_evaluation_results_llama-3.1-8b/fluency_evaluation_detailed.csv"),
}


def source_from_name(path: Path):
    name = path.name.lower()
    if "_long_" in name or "long_" in name:
        return "long"
    if "_short_" in name or "short_" in name:
        return "short"
    return ""


def text_key(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def human_join_key(filename, profile):
    return normalized_filename(filename), text_key(profile)


def model_key(value):
    key = text_key(value)
    if key in ("", "nan", "none"):
        return ""
    return key.replace("meta-llama/", "").replace("-instruct", "")


def model_hint_from_filename(path: Path):
    """Extract the generating-summary model embedded in factuality filenames."""
    name = path.stem
    match = re.search(r"degraded_(.+?)(?:__|_meta-|_gemma-|$)", name, flags=re.I)
    return match.group(1) if match else ""


def keypoint_type_from_path(path: Path):
    match = re.search(r"(?:narrow|wide|statistical)_keypoints", path.as_posix().lower())
    return match.group(0).removesuffix("_keypoints") if match else "not_applicable"


def factuality_claim_text(raw):
    """Extract the claim from an FP-list entry, irrespective of verdict subtype.

    CONTRADICTED/NOT_SUPPORTED explanations describe the evaluation, not the
    summary passage. Split before the marker even when reasoning spans lines.
    Bracketed leading item indices are output formatting, not claim content.
    """
    if not isinstance(raw, str):
        return ""
    claim = re.split(r"\bVERDICT\s*:", raw, maxsplit=1, flags=re.I)[0].strip()
    return re.sub(r"^\s*\[\d+\]\s*", "", claim).strip()


def load_factuality(directory):
    rows, statements = [], []
    for path in sorted(directory.glob("*.json")):
        source = source_from_name(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data:
            metrics = entry.get("metrics", {})
            rows.append({
                "dimension": "factuality", "filename": entry.get("filename", ""),
                "source_hint": source, "profile": entry.get("profile", ""),
                "automatic_model": entry.get("model", ""), "automatic_score": metrics.get("precision"),
                "summary_model_hint": model_hint_from_filename(path),
                "degraded_hint": "",
                "keypoint_type": "not_applicable",
                "method_variant": "default",
                "tp_count": metrics.get("tp_count"), "fp_count": metrics.get("fp_count"),
                "metric_file": str(path),
            })
            for raw in entry.get("verdict_reasoning", {}).get("FP", []):
                statement = factuality_claim_text(raw)
                if statement:
                    statements.append({"filename": entry.get("filename", ""), "source_hint": source,
                                       "profile": entry.get("profile", ""), "statement": statement,
                                       "raw_statement": raw,
                                       "summary_model_hint": model_hint_from_filename(path),
                                       "keypoint_type": "not_applicable",
                                       "method_variant": "default",
                                       "verdict": "FP", "metric_file": str(path)})
    return pd.DataFrame(rows), pd.DataFrame(statements)


def load_relevance(directory):
    rows, keypoints = [], []
    if directory.name.startswith("entailment_evaluation_results_"):
        result_directories = [directory]
    else:
        result_directories = sorted(directory.glob("*/entailment_evaluation_results_*_log_reconsider_*_keypoints"))
        result_directories += sorted(directory.glob("entailment_evaluation_results_*_log_reconsider_*_keypoints"))
        result_directories = [path for path in result_directories if path.is_dir()]
    for result_directory in result_directories:
      keypoint_type = keypoint_type_from_path(result_directory)
      for path in sorted(result_directory.glob("*.json")):
        source = source_from_name(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data:
            rows.append({
                "dimension": "relevance", "filename": entry.get("filename", ""),
                "source_hint": source, "profile": entry.get("profile", ""),
                "automatic_model": entry.get("summary_model", ""),
                "summary_model_hint": entry.get("summary_model", ""),
                "degraded_hint": "",
                "keypoint_type": keypoint_type,
                "method_variant": keypoint_type,
                "automatic_score": entry.get("completeness"),
                "total_count": entry.get("total_count"), "covered_count": entry.get("covered_count"),
                "metric_file": str(path),
            })
            for point in entry.get("key_items", []):
                keypoints.append({"filename": entry.get("filename", ""), "source_hint": source,
                                  "profile": entry.get("profile", ""), "keypoint": point.get("item", ""),
                                  "summary_model_hint": entry.get("summary_model", ""),
                                  "keypoint_type": keypoint_type,
                                  "method_variant": keypoint_type,
                                  "covered": point.get("covered"), "score": point.get("score"),
                                  "metric_file": str(path)})
    return pd.DataFrame(rows), pd.DataFrame(keypoints)


def load_rating_csv(path, dimension):
    frame = pd.read_csv(path, dtype=str)
    prompt_name = frame.get("prompt_name", "")
    prompt_text = (prompt_name.fillna("")).astype(str)
    if dimension in {"coherence", "fluency"}:
        is_auto_cot = prompt_text.str.contains("auto_cot", case=False, na=False)
        is_non_cot = prompt_text.str.contains("explain.txt|rate_explain.txt", case=False, na=False, regex=True)
        method_variant = np.where(is_auto_cot, "auto_cot", np.where(is_non_cot, "no_auto_cot", "unknown"))
    else:
        method_variant = "default"
    return pd.DataFrame({
        "dimension": dimension,
        "filename": frame["filename"],
        "source_hint": frame["filename"].map(lambda x: ""),
        "profile": frame["profile"],
        "automatic_model": frame.get("summary_model", ""),
        "summary_model_hint": frame.get("summary_model", ""),
        "degraded_hint": frame.get("degraded", ""),
        "keypoint_type": "not_applicable",
        "method_variant": method_variant,
        "automatic_score": pd.to_numeric(frame["expected_rating"], errors="coerce"),
        "metric_file": str(path),
        "prompt_name": prompt_text,
    })


def attach_to_humans(metrics, human):
    """Join on filename/profile/model, retaining ambiguous automatic rows visibly."""
    if metrics.empty:
        return metrics
    metrics = metrics.copy()
    human = human.copy()
    metrics["join_filename"] = metrics["filename"].map(normalized_filename)
    metrics["join_profile"] = metrics["profile"].map(text_key)
    human["join_filename"] = human["filename"].map(normalized_filename)
    human["join_profile"] = human["degradation_profile"].map(text_key)
    human["join_model"] = human["summary_model"].map(model_key)
    human["join_degraded"] = human["degraded_dimension"].map(text_key)
    human_groups = {key: group for key, group in human.groupby(["join_filename", "join_profile"])}
    human_columns = ["item_key", "source", "summary_id", "summary_text", "summary_model",
                     "degradation_profile", "consensus_rank", "n_annotators", *DIMENSIONS]
    matched_rows = []
    for _, metric in metrics.iterrows():
        row = metric.to_dict()
        candidates = human_groups.get((row["join_filename"], row["join_profile"]))
        if candidates is None:
            row["join_status"] = "no_human_match"
        else:
            hint = model_key(row.get("summary_model_hint", ""))
            if hint:
                candidates = candidates[candidates["join_model"] == hint]
            degraded_hint = text_key(row.get("degraded_hint", ""))
            if len(candidates) > 1 and degraded_hint:
                candidates = candidates[candidates["join_degraded"] == degraded_hint]
            if len(candidates) == 1:
                row.update(candidates.iloc[0][human_columns].to_dict())
                row["join_status"] = "matched"
            elif len(candidates) == 0:
                row["join_status"] = "no_model_match"
            else:
                row["join_status"] = "ambiguous_human_match"
        matched_rows.append(row)
    return pd.DataFrame(matched_rows)


def bootstrap_cluster(frame, statistic, n_resamples=1000, seed=42):
    keys = frame["item_key"].dropna().unique()
    if len(keys) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = []
    by_key = {key: group for key, group in frame.groupby("item_key")}
    for _ in range(n_resamples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([by_key[key] for key in selected], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return np.nan, np.nan
    return np.quantile(samples, [0.025, 0.975])


def metric_validity(merged):
    rows = []
    for (dimension, method_variant), group in merged.groupby(["dimension", "method_variant"], dropna=False):
        data = group.dropna(subset=["automatic_score", dimension, "item_key"]).copy()
        if len(data) < 3:
            continue
        auto, human = data["automatic_score"].astype(float), data[dimension].astype(float)
        rho, rho_p = spearmanr(auto, human)
        tau, tau_p = kendalltau(auto, human)
        if dimension in ("factuality", "relevance"):
            comparable_auto = 1 + 4 * auto
        else:
            comparable_auto = auto
        mae = np.mean(np.abs(comparable_auto - human))
        bias = np.mean(comparable_auto - human)
        rho_ci = bootstrap_cluster(data, lambda d: spearmanr(d["automatic_score"], d[dimension]).statistic)
        tau_ci = bootstrap_cluster(data, lambda d: kendalltau(d["automatic_score"], d[dimension]).statistic)
        rows.append({"dimension": dimension, "method_variant": method_variant,
                     "n_summaries": len(data), "spearman_rho": rho,
                     "spearman_p": rho_p, "rho_ci_95_low": rho_ci[0], "rho_ci_95_high": rho_ci[1],
                     "kendall_tau_b": tau, "kendall_p": tau_p, "tau_ci_95_low": tau_ci[0],
                     "tau_ci_95_high": tau_ci[1], "mae_1_to_5": mae, "bias_1_to_5": bias})
    return pd.DataFrame(rows)


def pairwise_metric_agreement(merged):
    rows = []
    for (dimension, method_variant), group in merged.groupby(["dimension", "method_variant"], dropna=False):
        comparisons = []
        for item_key, item in group.dropna(subset=["consensus_rank", "automatic_score"]).groupby("item_key"):
            for _, left, right in ((None, *pair) for pair in combinations(item.index, 2)):
                a, b = item.loc[left], item.loc[right]
                human_diff = a["consensus_rank"] - b["consensus_rank"]
                auto_diff = a["automatic_score"] - b["automatic_score"]
                if human_diff == 0:
                    continue
                comparisons.append({"item_key": item_key, "correct": (auto_diff * human_diff) < 0,
                                    "auto_tie": auto_diff == 0})
        pairs = pd.DataFrame(comparisons)
        if pairs.empty:
            continue
        ci = bootstrap_cluster(pairs, lambda d: d["correct"].mean())
        rows.append({"dimension": dimension, "method_variant": method_variant, "n_pairs": len(pairs), "accuracy": pairs["correct"].mean(),
                     "metric_tie_rate": pairs["auto_tie"].mean(), "ci_95_low": ci[0], "ci_95_high": ci[1]})
    return pd.DataFrame(rows)


def method_agreement_summary(validity, pairwise):
    if validity.empty and pairwise.empty:
        return pd.DataFrame(columns=["dimension", "method_variant", "n_summaries", "n_pairs",
                                     "spearman_rho", "kendall_tau_b", "mae_1_to_5", "bias_1_to_5",
                                     "accuracy", "metric_tie_rate", "agreement_rank"])
    combined = validity.merge(pairwise, on=["dimension", "method_variant"], how="outer")
    combined = combined.sort_values(
        by=["dimension", "spearman_rho", "accuracy", "mae_1_to_5"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    combined["agreement_rank"] = combined.groupby("dimension").cumcount() + 1
    return combined[["dimension", "method_variant", "n_summaries", "n_pairs",
                     "spearman_rho", "kendall_tau_b", "mae_1_to_5", "bias_1_to_5",
                     "accuracy", "metric_tie_rate", "agreement_rank"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-output-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/human_analysis"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/automatic_analysis"))
    parser.add_argument("--factuality-dir", type=Path, default=DEFAULTS["factuality"])
    parser.add_argument("--relevance-dir", type=Path, default=DEFAULTS["relevance"])
    parser.add_argument("--coherence-file", type=Path, default=DEFAULTS["coherence"])
    parser.add_argument("--fluency-file", type=Path, default=DEFAULTS["fluency"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(args.human_output_dir / "human_scores_aggregated.csv", dtype=str)
    for column in [*DIMENSIONS, "consensus_rank", "n_annotators"]:
        human[column] = pd.to_numeric(human[column], errors="coerce")

    factuality, statements = load_factuality(args.factuality_dir)
    relevance, keypoints = load_relevance(args.relevance_dir)
    coherence = load_rating_csv(args.coherence_file, "coherence")
    fluency = load_rating_csv(args.fluency_file, "fluency")
    metrics = pd.concat([factuality, relevance, coherence, fluency], ignore_index=True)
    merged = attach_to_humans(metrics, human)
    metrics.to_csv(args.output_dir / "automatic_metrics_raw.csv", index=False)
    merged.to_csv(args.output_dir / "human_automatic_merged.csv", index=False)
    validity = metric_validity(merged)
    pairwise = pairwise_metric_agreement(merged)
    validity.to_csv(args.output_dir / "metric_validity.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_metric_agreement.csv", index=False)
    method_agreement_summary(validity, pairwise).to_csv(
        args.output_dir / "method_agreement_summary.csv", index=False
    )

    # These tables are deliberately retained before span/keypoint matching.
    matched_statement_rows = attach_to_humans(statements.assign(dimension="factuality", automatic_score=np.nan), human)
    matched_statement_rows.to_csv(args.output_dir / "automatic_factuality_fp_statements.csv", index=False)
    matched_keypoint_rows = attach_to_humans(keypoints.assign(dimension="relevance", automatic_score=np.nan), human)
    matched_keypoint_rows.to_csv(args.output_dir / "automatic_coverage_keypoints.csv", index=False)
    unmatched = merged[merged["item_key"].isna()].groupby("dimension").size().rename("unmatched_rows")
    print(f"Wrote {len(merged)} automatic result rows; joined {merged['item_key'].notna().sum()} to human summaries.")
    print("Unmatched automatic rows by dimension:")
    print(unmatched.to_string() if len(unmatched) else "  none")


if __name__ == "__main__":
    main()
