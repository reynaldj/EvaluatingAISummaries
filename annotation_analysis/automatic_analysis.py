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
    "coherence": Path("coherence"),
    "fluency": Path("fluency"),
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


def infer_evaluator_model(path: Path):
    text = path.as_posix().lower()
    if "coherence_geval_ablation_gemma12b" in text or "fluency_evaluation_results_gemma12b" in text:
        return "gemma-12b"
    if "coherence_geval_ablation_gemma31b" in text or "fluency_evaluation_results_gemma_31b" in text:
        return "gemma-31b"
    if "coherence_geval_ablation_llama70b" in text or "fluency_evaluation_results_llama70b" in text:
        return "llama-70b"
    if "coherence_geval_ablation_llama3.1-8b-instruct" in text or "fluency_evaluation_results_llama-3.1-8b" in text:
        return "llama-3.1-8b"
    for candidate in (
        "llama-3.1-8b",
        "gpt-oss-120b",
        "mistral-large-3-675b-instruct-2512",
        "qwen3.5-122b-a10b",
    ):
        if candidate in text:
            return candidate
    return path.parent.name if path.parent.name else "unknown"


def factuality_evaluator_name(directory: Path):
    name = directory.name.lower()
    if "llama8b" in name:
        return "llama-3.1-8b"
    if "llama70b" in name:
        return "llama-70b"
    if "gemma-12b" in name:
        return "gemma-12b"
    if "gemma-31b" in name:
        return "gemma-31b"
    return infer_evaluator_model(directory)


def relevance_evaluator_name(directory: Path):
    """Read the evaluator identity from relevance's evaluator directory."""
    candidates = " ".join(part.lower() for part in directory.parts[-2:])
    if "gemma12b" in candidates:
        return "gemma-12b"
    if "gemma31b" in candidates:
        return "gemma-31b"
    if "llama-3.1-8b" in candidates or "llama3.1-8b" in candidates:
        return "llama-3.1-8b"
    if "llama_log" in candidates:
        return "llama-3.1-8b"
    return infer_evaluator_model(directory)


def discover_rating_files(path: Path):
    """Discover all detailed rating files below an evaluator-output directory."""
    if path.is_file():
        jsonl_path = path.with_suffix(".jsonl")
        return [(infer_evaluator_model(path), jsonl_path if jsonl_path.exists() else path)]
    candidates = []
    for evaluator_directory in sorted(path.glob("*")):
        if not evaluator_directory.is_dir():
            continue
        jsonl_files = sorted(evaluator_directory.glob("*_detailed.jsonl"))
        csv_files = sorted(evaluator_directory.glob("*_detailed.csv"))
        for candidate in jsonl_files or csv_files:
            candidates.append((infer_evaluator_model(candidate), candidate))
    return candidates


def model_hint_from_filename(path: Path):
    """Extract the generating-summary model embedded in factuality filenames."""
    name = path.stem
    match = re.search(r"degraded_(.+?)(?:__|_meta-|_gemma-|$)", name, flags=re.I)
    return match.group(1) if match else ""


def keypoint_type_from_path(path: Path):
    match = re.search(r"(?:narrow|wide|statistical)_keypoints", path.as_posix().lower())
    return match.group(0).removesuffix("_keypoints") if match else "not_applicable"


def read_jsonl_records(path):
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSONL record in {path} at line {line_number}: {error}") from error
        if isinstance(record, dict):
            records.append(record)
    return records


def load_factuality(directory):
    rows, statements = [], []
    directories = [directory]
    if directory.parent.exists():
        directories.extend(sorted(directory.parent.glob("results-factuality-evaluation-*")))
    directories = [path for path in dict.fromkeys(directories) if path.is_dir()]
    for result_directory in directories:
      evaluator_model = factuality_evaluator_name(result_directory)
      for path in sorted(result_directory.glob("*.json")):
        source = source_from_name(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data:
            metrics = entry.get("metrics", {})
            generation_model = model_hint_from_filename(path)
            rows.append({
                "dimension": "factuality", "filename": entry.get("filename", ""),
                "source_hint": source, "profile": entry.get("profile", ""),
                "automatic_model": generation_model,
                "generation_model": generation_model,
                "evaluator_model": evaluator_model,
                "automatic_score": metrics.get("precision"),
                "summary_model_hint": model_hint_from_filename(path),
                "degraded_hint": "",
                "keypoint_type": "not_applicable",
                "method_variant": "default",
                "tp_count": metrics.get("tp_count"), "fp_count": metrics.get("fp_count"),
                "metric_file": str(path),
            })
            for raw in entry.get("verdict_reasoning", {}).get("FP", []):
                statement = re.sub(r"\s*VERDICT:\s*FP.*$", "", str(raw), flags=re.I).strip()
                if statement:
                    statements.append({"filename": entry.get("filename", ""), "source_hint": source,
                                       "profile": entry.get("profile", ""), "statement": statement,
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
            generation_model = entry.get("summary_model", "") or entry.get("model", "")
            rows.append({
                "dimension": "relevance", "filename": entry.get("filename", ""),
                "source_hint": source, "profile": entry.get("profile", ""),
                "automatic_model": generation_model,
                "generation_model": generation_model,
                "evaluator_model": relevance_evaluator_name(result_directory),
                "summary_model_hint": generation_model,
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


def load_rating_csv(path, dimension, evaluator_model=None):
    if path.suffix.lower() == ".jsonl":
        frame = pd.DataFrame(read_jsonl_records(path))
        for column in frame.columns:
            if column != "expected_rating":
                frame[column] = frame[column].astype(str)
    else:
        frame = pd.read_csv(path, dtype=str)
    prompt_name = frame.get("prompt_name", "")
    prompt_text = (prompt_name.fillna("")).astype(str)
    if dimension in {"coherence", "fluency"}:
        is_auto_cot = prompt_text.str.contains("auto_cot", case=False, na=False)
        is_non_cot = prompt_text.str.contains("explain.txt|rate_explain.txt", case=False, na=False, regex=True)
        method_variant = np.where(is_auto_cot, "auto_cot", np.where(is_non_cot, "no_auto_cot", "unknown"))
    else:
        method_variant = "default"
    generation_model = frame.get("summary_model", "")
    return pd.DataFrame({
        "dimension": dimension,
        "filename": frame["filename"],
        "source_hint": frame["filename"].map(lambda x: ""),
        "profile": frame["profile"],
        "automatic_model": generation_model,
        "generation_model": generation_model,
        "evaluator_model": evaluator_model or infer_evaluator_model(path),
        "summary_model_hint": generation_model,
        "degraded_hint": frame.get("degraded", ""),
        "keypoint_type": "not_applicable",
        "method_variant": method_variant,
        "automatic_score": pd.to_numeric(frame["expected_rating"], errors="coerce"),
        "parsed_rating": pd.to_numeric(frame.get("parsed_rating"), errors="coerce"),
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


def bootstrap_cluster(frame, statistic, n_resamples=200, seed=42):
    keys = frame["item_key"].dropna().unique()
    if len(keys) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = []
    by_key = {key: group for key, group in frame.groupby("item_key")}
    for _ in range(min(n_resamples, 200)):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([by_key[key] for key in selected], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return np.nan, np.nan
    return np.quantile(samples, [0.025, 0.975])


def mapped_rating_accuracy(dimension, parsed, human):
    """Return exact integer-rating agreement for 1-5 rating dimensions."""
    if dimension not in ("coherence", "fluency") or parsed is None:
        return np.nan
    mapped = pd.to_numeric(parsed, errors="coerce")
    valid = mapped.notna() & human.notna()
    if not valid.any():
        return np.nan
    mapped = mapped[valid].astype(float)
    human = human[valid].astype(float)
    return float((mapped == human).mean())


def parsed_rating_errors(dimension, parsed, human):
    """Return MAE and bias for exact parsed 1-5 ratings."""
    if dimension not in ("coherence", "fluency") or parsed is None:
        return np.nan, np.nan
    parsed = pd.to_numeric(parsed, errors="coerce")
    valid = parsed.notna() & human.notna()
    if not valid.any():
        return np.nan, np.nan
    difference = parsed[valid].astype(float) - human[valid].astype(float)
    return float(np.abs(difference).mean()), float(difference.mean())


def automatic_pair_tie(first, second, dimension):
    """Use parsed integer ties for coherence/fluency and raw ties otherwise."""
    if dimension in ("coherence", "fluency"):
        return first.get("parsed_rating") == second.get("parsed_rating")
    return first["automatic_score"] == second["automatic_score"]


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
        mapped_accuracy = mapped_rating_accuracy(dimension, data.get("parsed_rating"), human)
        parsed_mae, parsed_bias = parsed_rating_errors(dimension, data.get("parsed_rating"), human)
        mae = np.mean(np.abs(comparable_auto - human))
        bias = np.mean(comparable_auto - human)
        rho_ci = bootstrap_cluster(data, lambda d: spearmanr(d["automatic_score"], d[dimension]).statistic)
        tau_ci = bootstrap_cluster(data, lambda d: kendalltau(d["automatic_score"], d[dimension]).statistic)
        rows.append({"dimension": dimension, "method_variant": method_variant,
                     "n_summaries": len(data), "spearman_rho": rho,
                     "spearman_p": rho_p, "rho_ci_95_low": rho_ci[0], "rho_ci_95_high": rho_ci[1],
                     "kendall_tau_b": tau, "kendall_p": tau_p, "tau_ci_95_low": tau_ci[0],
                     "tau_ci_95_high": tau_ci[1], "mae_1_to_5": mae, "bias_1_to_5": bias,
                     "mapped_rating_accuracy": mapped_accuracy,
                     "parsed_rating_mae_1_to_5": parsed_mae,
                     "parsed_rating_bias_1_to_5": parsed_bias})
    return pd.DataFrame(rows)


def metric_validity_by_generation_model(merged):
    """Report automatic-vs-human validity separately for each generation model."""
    rows = []
    model_frame = merged.copy()
    model_frame["generation_model"] = model_frame["generation_model"].fillna("unknown").astype(str)
    for (dimension, method_variant, generation_model), group in model_frame.groupby(
        ["dimension", "method_variant", "generation_model"], dropna=False
    ):
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
        mapped_accuracy = mapped_rating_accuracy(dimension, data.get("parsed_rating"), human)
        parsed_mae, parsed_bias = parsed_rating_errors(dimension, data.get("parsed_rating"), human)
        mae = np.mean(np.abs(comparable_auto - human))
        bias = np.mean(comparable_auto - human)
        rho_ci = bootstrap_cluster(data, lambda d: spearmanr(d["automatic_score"], d[dimension]).statistic)
        tau_ci = bootstrap_cluster(data, lambda d: kendalltau(d["automatic_score"], d[dimension]).statistic)
        rows.append({
            "dimension": dimension,
            "method_variant": method_variant,
            "generation_model": generation_model,
            "n_summaries": len(data),
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "rho_ci_95_low": rho_ci[0],
            "rho_ci_95_high": rho_ci[1],
            "kendall_tau_b": tau,
            "kendall_p": tau_p,
            "tau_ci_95_low": tau_ci[0],
            "tau_ci_95_high": tau_ci[1],
            "mae_1_to_5": mae,
            "bias_1_to_5": bias,
            "mapped_rating_accuracy": mapped_accuracy,
            "parsed_rating_mae_1_to_5": parsed_mae,
            "parsed_rating_bias_1_to_5": parsed_bias,
        })
    return pd.DataFrame(rows)


def metric_validity_by_evaluator_model(merged):
    """Report automatic-vs-human validity separately for each evaluator model."""
    rows = []
    model_frame = merged.copy()
    model_frame["evaluator_model"] = model_frame["evaluator_model"].fillna("unknown").astype(str)
    for (dimension, method_variant, evaluator_model), group in model_frame.groupby(
        ["dimension", "method_variant", "evaluator_model"], dropna=False
    ):
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
        mapped_accuracy = mapped_rating_accuracy(dimension, data.get("parsed_rating"), human)
        parsed_mae, parsed_bias = parsed_rating_errors(dimension, data.get("parsed_rating"), human)
        mae = np.mean(np.abs(comparable_auto - human))
        bias = np.mean(comparable_auto - human)
        rho_ci = bootstrap_cluster(data, lambda d: spearmanr(d["automatic_score"], d[dimension]).statistic)
        tau_ci = bootstrap_cluster(data, lambda d: kendalltau(d["automatic_score"], d[dimension]).statistic)
        rows.append({
            "dimension": dimension,
            "method_variant": method_variant,
            "evaluator_model": evaluator_model,
            "n_summaries": len(data),
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "rho_ci_95_low": rho_ci[0],
            "rho_ci_95_high": rho_ci[1],
            "kendall_tau_b": tau,
            "kendall_p": tau_p,
            "tau_ci_95_low": tau_ci[0],
            "tau_ci_95_high": tau_ci[1],
            "mae_1_to_5": mae,
            "bias_1_to_5": bias,
            "mapped_rating_accuracy": mapped_accuracy,
            "parsed_rating_mae_1_to_5": parsed_mae,
            "parsed_rating_bias_1_to_5": parsed_bias,
        })
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
                                    "auto_tie": automatic_pair_tie(a, b, dimension)})
        pairs = pd.DataFrame(comparisons)
        if pairs.empty:
            continue
        ci = bootstrap_cluster(pairs, lambda d: d["correct"].mean())
        rows.append({"dimension": dimension, "method_variant": method_variant, "n_pairs": len(pairs), "accuracy": pairs["correct"].mean(),
                     "metric_tie_rate": pairs["auto_tie"].mean(), "ci_95_low": ci[0], "ci_95_high": ci[1]})
    return pd.DataFrame(rows)


def pairwise_metric_agreement_by_generation_model(merged):
    """Report pairwise agreement separately for each generation model."""
    rows = []
    model_frame = merged.copy()
    model_frame["generation_model"] = model_frame["generation_model"].fillna("unknown").astype(str)
    for (dimension, method_variant, generation_model), group in model_frame.groupby(
        ["dimension", "method_variant", "generation_model"], dropna=False
    ):
        comparisons = []
        for item_key, item in group.dropna(subset=["consensus_rank", "automatic_score"]).groupby("item_key"):
            for _, left, right in ((None, *pair) for pair in combinations(item.index, 2)):
                a, b = item.loc[left], item.loc[right]
                human_diff = a["consensus_rank"] - b["consensus_rank"]
                auto_diff = a["automatic_score"] - b["automatic_score"]
                if human_diff == 0:
                    continue
                comparisons.append({
                    "item_key": item_key,
                    "correct": (auto_diff * human_diff) < 0,
                    "auto_tie": automatic_pair_tie(a, b, dimension),
                })
        pairs = pd.DataFrame(comparisons)
        if pairs.empty:
            continue
        ci = bootstrap_cluster(pairs, lambda d: d["correct"].mean())
        rows.append({
            "dimension": dimension,
            "method_variant": method_variant,
            "generation_model": generation_model,
            "n_pairs": len(pairs),
            "accuracy": pairs["correct"].mean(),
            "metric_tie_rate": pairs["auto_tie"].mean(),
            "ci_95_low": ci[0],
            "ci_95_high": ci[1],
        })
    return pd.DataFrame(rows)


def pairwise_metric_agreement_by_evaluator_model(merged):
    """Report pairwise agreement separately for each evaluator model."""
    rows = []
    model_frame = merged.copy()
    model_frame["evaluator_model"] = model_frame["evaluator_model"].fillna("unknown").astype(str)
    for (dimension, method_variant, evaluator_model), group in model_frame.groupby(
        ["dimension", "method_variant", "evaluator_model"], dropna=False
    ):
        comparisons = []
        for item_key, item in group.dropna(subset=["consensus_rank", "automatic_score"]).groupby("item_key"):
            for _, left, right in ((None, *pair) for pair in combinations(item.index, 2)):
                a, b = item.loc[left], item.loc[right]
                human_diff = a["consensus_rank"] - b["consensus_rank"]
                auto_diff = a["automatic_score"] - b["automatic_score"]
                if human_diff == 0:
                    continue
                comparisons.append({
                    "item_key": item_key,
                    "correct": (auto_diff * human_diff) < 0,
                    "auto_tie": automatic_pair_tie(a, b, dimension),
                })
        pairs = pd.DataFrame(comparisons)
        if pairs.empty:
            continue
        ci = bootstrap_cluster(pairs, lambda d: d["correct"].mean())
        rows.append({
            "dimension": dimension,
            "method_variant": method_variant,
            "evaluator_model": evaluator_model,
            "n_pairs": len(pairs),
            "accuracy": pairs["correct"].mean(),
            "metric_tie_rate": pairs["auto_tie"].mean(),
            "ci_95_low": ci[0],
            "ci_95_high": ci[1],
        })
    return pd.DataFrame(rows)


def method_agreement_summary(validity, pairwise):
    key_columns = ["dimension", "method_variant"]
    model_column = next(
        (column for column in ("automatic_model", "generation_model", "evaluator_model")
         if column in validity.columns or column in pairwise.columns),
        None,
    )
    if model_column:
        key_columns.append(model_column)
    if validity.empty and not set(key_columns).issubset(validity.columns):
        validity = pd.DataFrame(columns=key_columns)
    if pairwise.empty and not set(key_columns).issubset(pairwise.columns):
        pairwise = pd.DataFrame(columns=key_columns)
    for column in (
        "n_summaries", "spearman_rho", "kendall_tau_b", "mae_1_to_5", "bias_1_to_5",
        "parsed_rating_mae_1_to_5", "parsed_rating_bias_1_to_5",
    ):
        if column not in validity.columns:
            validity[column] = np.nan
    if "mapped_rating_accuracy" not in validity.columns:
        validity["mapped_rating_accuracy"] = np.nan
    for column in ("n_pairs", "accuracy", "metric_tie_rate"):
        if column not in pairwise.columns:
            pairwise[column] = np.nan
    if validity.empty and pairwise.empty:
        columns = ["dimension", "method_variant", "n_summaries", "n_pairs",
                   "spearman_rho", "kendall_tau_b", "mae_1_to_5", "bias_1_to_5",
                   "mapped_rating_accuracy", "parsed_rating_mae_1_to_5",
                   "parsed_rating_bias_1_to_5", "accuracy", "metric_tie_rate", "agreement_rank"]
        if model_column:
            columns.insert(2, model_column)
        return pd.DataFrame(columns=columns)
    combined = validity.merge(pairwise, on=key_columns, how="outer")
    sort_columns = ["dimension"]
    if model_column:
        sort_columns.append(model_column)
    sort_columns.extend(["spearman_rho", "accuracy", "mae_1_to_5"])
    combined = combined.sort_values(
        by=sort_columns,
        ascending=[True] + [False] * (len(sort_columns) - 1),
        kind="mergesort",
    ).reset_index(drop=True)
    if model_column:
        combined["agreement_rank"] = combined.groupby(["dimension", model_column]).cumcount() + 1
    else:
        combined["agreement_rank"] = combined.groupby("dimension").cumcount() + 1
    output_columns = ["dimension", "method_variant", "n_summaries", "n_pairs",
                      "spearman_rho", "kendall_tau_b", "mae_1_to_5", "bias_1_to_5",
                      "mapped_rating_accuracy", "parsed_rating_mae_1_to_5",
                      "parsed_rating_bias_1_to_5", "accuracy", "metric_tie_rate", "agreement_rank"]
    if model_column:
        output_columns.insert(2, model_column)
    return combined[output_columns]


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
    coherence = pd.concat(
        [load_rating_csv(path, "coherence", evaluator_model) for evaluator_model, path in discover_rating_files(args.coherence_file)],
        ignore_index=True,
    )
    fluency = pd.concat(
        [load_rating_csv(path, "fluency", evaluator_model) for evaluator_model, path in discover_rating_files(args.fluency_file)],
        ignore_index=True,
    )
    metrics = pd.concat([factuality, relevance, coherence, fluency], ignore_index=True)
    merged = attach_to_humans(metrics, human)
    metrics.to_csv(args.output_dir / "automatic_metrics_raw.csv", index=False)
    merged.to_csv(args.output_dir / "human_automatic_merged.csv", index=False)
    validity = metric_validity(merged)
    pairwise = pairwise_metric_agreement(merged)
    validity_by_generation_model = metric_validity_by_generation_model(merged)
    validity_by_evaluator_model = metric_validity_by_evaluator_model(merged)
    pairwise_by_generation_model = pairwise_metric_agreement_by_generation_model(merged)
    pairwise_by_evaluator_model = pairwise_metric_agreement_by_evaluator_model(merged)

    validity.to_csv(args.output_dir / "metric_validity.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_metric_agreement.csv", index=False)
    validity_by_generation_model.to_csv(args.output_dir / "metric_validity_by_generation_model.csv", index=False)
    pairwise_by_generation_model.to_csv(args.output_dir / "pairwise_metric_agreement_by_generation_model.csv", index=False)
    validity_by_evaluator_model.to_csv(args.output_dir / "metric_validity_by_evaluator_model.csv", index=False)
    pairwise_by_evaluator_model.to_csv(args.output_dir / "pairwise_metric_agreement_by_evaluator_model.csv", index=False)

    method_agreement_summary(validity, pairwise).to_csv(
        args.output_dir / "method_agreement_summary.csv", index=False
    )
    method_agreement_summary(validity_by_generation_model, pairwise_by_generation_model).to_csv(
        args.output_dir / "method_agreement_summary_by_generation_model.csv", index=False
    )
    method_agreement_summary(validity_by_evaluator_model, pairwise_by_evaluator_model).to_csv(
        args.output_dir / "method_agreement_summary_by_evaluator_model.csv", index=False
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

#