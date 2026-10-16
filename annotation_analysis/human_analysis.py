"""Build reproducible human-evaluation analysis tables.

Run from the repository root:
    python annotation_analysis/human_analysis.py

The script never changes source annotation exports.  It writes normalized raw
ratings, span labels, agreement summaries, and consensus scores below
``annotation_analysis/analysis_output`` by default.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

try:
    import krippendorff
except ImportError:  # Keep ingestion usable if the optional package is absent.
    krippendorff = None


DIMENSIONS = ("coherence", "factuality", "fluency", "relevance")
SLOTS = ("a", "b", "c")


def parse_json(value, default):
    if pd.isna(value) or not str(value).strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def parse_rating(value):
    data = parse_json(value, [])
    try:
        return float(data[0]["rating"])
    except (IndexError, KeyError, TypeError):
        return np.nan


def parse_ranking(value):
    data = parse_json(value, [])
    try:
        return data[0]["ranker"]["summary_ranking"]
    except (IndexError, KeyError, TypeError):
        return []


def word_count(text):
    return len(str(text).split()) if pd.notna(text) else 0


def normalized_filename(value):
    """Normalize common UTF-8-as-Latin-1 export damage for joins only."""
    text = str(value).strip()
    if "Ã" in text or "â" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    return text


def read_assignment_metadata(assignment_dir: Path, pilot_dir: Path | None = None):
    """Return slot metadata keyed by source, filename, and annotator assignment."""
    paths = list(assignment_dir.glob("*_annotations.json"))
    if pilot_dir is not None and pilot_dir.exists():
        paths.extend(sorted(pilot_dir.glob("*.json")))
    metadata = {}
    for path in paths:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        is_pilot_assignment = pilot_dir is not None and path.parent == pilot_dir
        dataset = "__pilot__" if is_pilot_assignment else path.stem.removesuffix("_annotations")
        for record in records:
            key = (str(record.get("source", "")).lower(),
                   normalized_filename(record.get("filename", "")), dataset)
            entry = metadata.setdefault(key, {"datasets": set()})
            entry["datasets"].add(path.stem.removesuffix("_annotations"))
            for slot in SLOTS:
                for field in ("model", "profile", "degraded"):
                    value = record.get(f"{field}_{slot}")
                    if value is not None and str(value).lower() != "nan":
                        entry[f"{field}_{slot}"] = value
    return metadata


def read_exports(results_dir: Path):
    frames = []
    for path in sorted(results_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        frame["input_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV exports found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def build_raw_tables(exports: pd.DataFrame, assignment_metadata):
    ratings, error_spans, transcript_spans = [], [], []
    for export_row, row in exports.iterrows():
        filename = str(row.get("filename", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        item_key = f"{source}::{filename}"
        input_stem = Path(row["input_file"]).stem
        assignment_dataset = "__pilot__" if input_stem.lower().startswith("pilot_") else input_stem
        assignment = assignment_metadata.get(
            (source, normalized_filename(filename), assignment_dataset), {}
        )
        annotation_id = str(row.get("annotation_id") or row.get("id") or export_row)
        annotator = str(row.get("annotator", "unknown")).strip()
        ranking = parse_ranking(row.get("summary_ranking"))
        rank_map = {summary_id: position for position, summary_id in enumerate(ranking, 1)}

        for slot in SLOTS:
            summary_id = f"summary_{slot}"
            summary = row.get(summary_id, "")
            record = {
                "item_key": item_key,
                "annotation_id": annotation_id,
                "filename": filename,
                "source": source,
                "annotator": annotator,
                "input_file": row["input_file"],
                "summary_id": summary_id,
                "summary_text": summary,
                "summary_model": row.get(f"model_{slot}") or assignment.get(f"model_{slot}", ""),
                "degradation_profile": row.get(f"profile_{slot}") or assignment.get(f"profile_{slot}", ""),
                "degraded_dimension": row.get(f"degraded_{slot}") or assignment.get(f"degraded_{slot}", ""),
                "assignment_datasets": ";".join(sorted(assignment.get("datasets", []))),
                "rank": rank_map.get(summary_id, np.nan),
                "summary_words": word_count(summary),
            }
            for dimension in DIMENSIONS:
                record[dimension] = parse_rating(row.get(f"{slot}_{dimension}"))
            ratings.append(record)

            for span in parse_json(row.get(f"{slot}_error_labels"), []):
                error_spans.append({
                    **{key: record[key] for key in (
                        "item_key", "annotation_id", "filename", "source", "annotator",
                        "input_file", "summary_id", "summary_model", "degradation_profile",
                    )},
                    "span_start": span.get("start"),
                    "span_end": span.get("end"),
                    "span_text": span.get("text", ""),
                    "label": "; ".join(span.get("labels", [])),
                })

        for span in parse_json(row.get("transcript_labels"), []):
            transcript_spans.append({
                "item_key": item_key,
                "annotation_id": annotation_id,
                "filename": filename,
                "source": source,
                "annotator": annotator,
                "input_file": row["input_file"],
                "span_start": span.get("start"),
                "span_end": span.get("end"),
                "span_text": span.get("text", ""),
                "label": "; ".join(span.get("labels", [])),
            })
    return pd.DataFrame(ratings), pd.DataFrame(error_spans), pd.DataFrame(transcript_spans)


def paired_values(frame, value_column, unit_columns):
    """Return all annotator-pair values on units rated by both annotators."""
    pairs = []
    for _, group in frame.dropna(subset=[value_column]).groupby(unit_columns):
        values = group.groupby("annotator")[value_column].mean().to_dict()
        for left, right in combinations(sorted(values), 2):
            pairs.append((values[left], values[right]))
    return pairs


def bootstrap_alpha(raw, dimension, unit_columns, n_resamples=2_000, seed=42):
    """Estimate a transcript-cluster bootstrap CI for ordinal Krippendorff alpha."""
    if krippendorff is None:
        return np.nan, np.nan, 0

    overlap_units = raw.groupby(unit_columns)["annotator"].nunique()
    overlap_keys = overlap_units[overlap_units >= 2].index
    if len(overlap_keys) == 0:
        return np.nan, np.nan, 0

    overlap_frame = raw.set_index(unit_columns).loc[overlap_keys].reset_index()
    transcript_keys = overlap_frame["item_key"].drop_duplicates().tolist()
    if len(transcript_keys) < 2:
        return np.nan, np.nan, 0

    annotators = sorted(overlap_frame["annotator"].dropna().unique())
    transcript_matrices = {}
    for key, group in overlap_frame.groupby("item_key", sort=False):
        unit_values = []
        for _, unit in group.groupby("summary_id", sort=False):
            values = unit.groupby("annotator")[dimension].mean()
            unit_values.append([
                values.get(annotator, np.nan) for annotator in annotators
            ])
        transcript_matrices[key] = np.asarray(unit_values, dtype=float).T

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        sampled_keys = rng.choice(transcript_keys, size=len(transcript_keys), replace=True)
        matrix = np.concatenate(
            [transcript_matrices[key] for key in sampled_keys], axis=1
        )
        if len(annotators) < 2 or matrix.shape[1] < 2:
            continue
        try:
            estimate = krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal",
                value_domain=list(range(1, 6)),
            )
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(estimate):
            estimates.append(float(estimate))

    if not estimates:
        return np.nan, np.nan, 0
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)), len(estimates)


def rating_agreement(raw):
    rows = []
    unit_columns = ["item_key", "summary_id"]
    for dimension in DIMENSIONS:
        matrix = raw.pivot_table(
            index="annotator", columns=unit_columns, values=dimension, aggfunc="mean" # aggfunc actually not needed since no annotator annoates same summary twice
        )
        pair_values = paired_values(raw, dimension, unit_columns)
        exact = np.mean([a == b for a, b in pair_values]) if pair_values else np.nan
        adjacent = np.mean([abs(a - b) <= 1 for a, b in pair_values]) if pair_values else np.nan
        alpha = np.nan
        if krippendorff is not None and matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
            try:
                alpha = krippendorff.alpha(
                    reliability_data=matrix.to_numpy(dtype=float),
                    level_of_measurement="ordinal",
                    value_domain=list(range(1, 6)),
                )
            except (ValueError, ZeroDivisionError):
                pass
        alpha_ci_low, alpha_ci_high, alpha_bootstrap_n = bootstrap_alpha(
            raw, dimension, unit_columns
        )
        overlap_units = sum(group["annotator"].nunique() >= 2
                            for _, group in raw.groupby(unit_columns))
        rows.append({
            "dimension": dimension,
            "krippendorff_alpha_ordinal": alpha,
            "alpha_ci_low": alpha_ci_low,
            "alpha_ci_high": alpha_ci_high,
            "alpha_bootstrap_replicates": alpha_bootstrap_n,
            "paired_ratings": len(pair_values),
            "exact_agreement": exact,
            "adjacent_agreement": adjacent,
            "overlap_units": overlap_units,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    result = bootstrap((values,), np.mean, n_resamples=2_000, random_state=seed,
                       confidence_level=0.95, method="basic")
    return result.confidence_interval.low, result.confidence_interval.high


def aggregate_scores(raw):
    identity_columns = ["item_key", "filename", "source", "summary_id"]
    metadata_columns = ["summary_text", "summary_model", "degradation_profile", "degraded_dimension"]
    rows = []
    for keys, group in raw.groupby(identity_columns, dropna=False):
        record = dict(zip(identity_columns, keys))
        for column in metadata_columns:
            values = group[column].dropna()
            record[column] = values.iloc[0] if not values.empty else ""
        record["n_annotators"] = group["annotator"].nunique()
        record["mean_rank"] = group["rank"].mean()
        for dimension in DIMENSIONS:
            record[dimension] = group[dimension].mean()
            record[f"{dimension}_sd"] = group[dimension].std()
        rows.append(record)
    aggregated = pd.DataFrame(rows)
    aggregated["consensus_rank"] = aggregated.groupby("item_key")["mean_rank"].rank(
        method="average", ascending=True
    )
    return aggregated


def descriptive_rows(frame, group_columns, value_level):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for dimension in DIMENSIONS:
            values = group[dimension].dropna()
            low, high = bootstrap_ci(values)
            record = dict(zip(group_columns, keys))
            record.update({
                "dimension": dimension,
                "value_level": value_level,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "ci_95_low": low,
                "ci_95_high": high,
            })
            rows.append(record)
    return pd.DataFrame(rows)


def annotator_label(value):
    """Normalize export filenames to the study's annotator labels."""
    name = Path(str(value)).stem.upper()
    for label in ("H1", "H2", "L3", "L4"):
        if name.startswith(label):
            return label
    return str(value)


def profile_categories(value):
    """Collapse detailed profiles into the requested quality/degradation groups."""
    text = str(value).lower()
    categories = []
    if "high_quality" in text:
        categories.append("high_quality")
    if "balanced_mid_quality" in text:
        categories.append("balanced_mid_quality")
    if "low_factuality" in text:
        categories.append("low_factuality")
    if "low_relevance" in text:
        categories.append("low_coverage")
    if "low_coherence" in text:
        categories.append("low_coherence")
    if "low_fluency" in text:
        categories.append("low_fluency")
    return categories


def human_descriptives(aggregated):
    """Summary-level descriptives grouped by source and generation model."""
    rows = []
    for keys, group in aggregated.groupby(["source", "summary_model"], dropna=False):
        for dimension in DIMENSIONS:
            values = group[dimension].dropna()
            low, high = bootstrap_ci(values)
            rows.append({
                "source": keys[0], "summary_model": keys[1], "dimension": dimension,
                "n_summaries": len(values), "mean": values.mean(), "sd": values.std(),
                "median": values.median(), "ci_95_low": low, "ci_95_high": high,
            })
    return pd.DataFrame(rows)


def human_descriptives_by_annotator(raw):
    """Raw-rating descriptives showing annotator-level rating behavior."""
    rows = raw.copy()
    rows["annotator"] = rows["input_file"].map(annotator_label)
    return descriptive_rows(rows, ["annotator"], "raw_annotator_ratings")


def human_descriptives_by_dimension(aggregated):
    """Overall summary-level descriptives for each evaluation dimension."""
    rows = []
    for dimension in DIMENSIONS:
        values = aggregated[dimension].dropna()
        low, high = bootstrap_ci(values)
        rows.append({
            "dimension": dimension,
            "value_level": "summary_consensus_means",
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(),
            "median": values.median(),
            "ci_95_low": low,
            "ci_95_high": high,
        })
    return pd.DataFrame(rows)


def human_descriptives_by_profile(aggregated):
    """Summary-level descriptives for simplified, possibly overlapping profiles."""
    rows = aggregated.copy()
    rows["profile_category"] = rows["degradation_profile"].map(profile_categories)
    rows = rows.explode("profile_category")
    rows = rows.dropna(subset=["profile_category"])
    return descriptive_rows(rows, ["profile_category"], "summary_consensus_means")


def human_descriptives_by_model_profile(aggregated):
    """Descriptives for the generation-model/profile plot, by dimension."""
    rows = aggregated.copy()
    rows["profile_category"] = rows["degradation_profile"].map(profile_categories)
    rows = rows.explode("profile_category")
    rows = rows.dropna(subset=["profile_category", "summary_model"])
    return descriptive_rows(
        rows,
        ["summary_model", "profile_category"],
        "summary_consensus_means",
    )


def dataset_summary(raw):
    rows = []
    for source, group in raw.groupby("source", dropna=False):
        item_annotator = group[["item_key", "annotator"]].drop_duplicates()
        counts = item_annotator.groupby("item_key")["annotator"].nunique()
        rows.append({
            "source": source,
            "transcripts": group["item_key"].nunique(),
            "ai_summaries": len(group),
            "human_rating_rows": len(group) * len(DIMENSIONS),
            "annotators": group["annotator"].nunique(),
            "overlap_transcripts": (counts >= 2).sum(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("main_annotation/Results"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/human_analysis"))
    parser.add_argument("--assignment-dir", type=Path,
                        default=Path("main_annotation/annotator_datasets"))
    parser.add_argument("--pilot-assignment-dir", type=Path,
                        default=Path("pilot/pilot_labelstudio_import"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exports = read_exports(args.results_dir)
    assignments = read_assignment_metadata(args.assignment_dir, args.pilot_assignment_dir)
    raw, error_spans, transcript_spans = build_raw_tables(exports, assignments)
    aggregated = aggregate_scores(raw)

    raw.to_csv(args.output_dir / "human_ratings_raw.csv", index=False)
    error_spans.to_csv(args.output_dir / "human_error_spans_raw.csv", index=False)
    transcript_spans.to_csv(args.output_dir / "human_transcript_labels_raw.csv", index=False)
    aggregated.to_csv(args.output_dir / "human_scores_aggregated.csv", index=False)
    dataset_summary(raw).to_csv(args.output_dir / "dataset_composition.csv", index=False)
    rating_agreement(raw).to_csv(args.output_dir / "rating_iaa.csv", index=False)
    human_descriptives(aggregated).to_csv(args.output_dir / "human_descriptives.csv", index=False)
    human_descriptives_by_dimension(aggregated).to_csv(
        args.output_dir / "human_descriptives_by_dimension.csv", index=False
    )
    descriptive_rows(aggregated, ["summary_model"], "summary_consensus_means").to_csv(
        args.output_dir / "human_descriptives_by_generation_model.csv", index=False
    )
    human_descriptives_by_profile(aggregated).to_csv(
        args.output_dir / "human_descriptives_by_profile.csv", index=False
    )
    human_descriptives_by_model_profile(aggregated).to_csv(
        args.output_dir / "human_descriptives_by_generation_model_profile.csv", index=False
    )
    human_descriptives_by_annotator(raw).to_csv(
        args.output_dir / "human_descriptives_by_annotator.csv", index=False
    )

    print(f"Read {len(exports)} annotation exports.")
    print(f"Wrote {len(raw)} summary ratings and {len(aggregated)} aggregated summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
