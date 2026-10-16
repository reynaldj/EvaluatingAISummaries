"""Learn integer 1-5 mappings from a small, auditable summary panel.

The script selects up to five shared summary instances for each target rating.
Summaries whose factuality and relevance ratings both equal the target are
prioritized, followed by summaries matching the target in one dimension and
then other shared summaries. The same panel is used for both mappings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSIONS = ("factuality", "relevance")
OUTPUT_COLUMNS = [
    "dimension", "bin", "lower_score", "upper_score", "n_calibration_rows",
    "mean_human_rating", "min_human_rating", "max_human_rating",
]


def add_summary_key(rows):
    rows = rows.copy()
    summary_id = rows.get("summary_id", pd.Series("", index=rows.index)).fillna("").astype(str)
    rows["summary_key"] = rows["item_key"].astype(str) + "::" + summary_id
    return rows


def choose_joint_calibration_summaries(prepared, summaries_per_rating, seed):
    """Prioritize equal joint ratings while filling each target-rating panel."""
    dimension_tables = []
    for dimension in DIMENSIONS:
        rows = prepared[dimension]
        if rows.empty:
            return set(), {}
        dimension_tables.append(
            rows[["summary_key", "human_rating"]]
            .drop_duplicates("summary_key")
            .rename(columns={"human_rating": f"{dimension}_rating"})
        )
    candidates = dimension_tables[0].merge(dimension_tables[1], on="summary_key", how="inner")
    rng = np.random.default_rng(seed)
    selected = set()
    selected_by_rating = {}
    for rating in range(1, 6):
        group = candidates.copy()
        group["matches_both"] = (
            (group["factuality_rating"] == rating)
            & (group["relevance_rating"] == rating)
        )
        group["matches_one"] = (
            (group["factuality_rating"] == rating)
            | (group["relevance_rating"] == rating)
        )
        group["distance"] = (
            (group["factuality_rating"] - rating).abs()
            + (group["relevance_rating"] - rating).abs()
        )
        group = group.sample(frac=1, random_state=int(rng.integers(0, 2**32 - 1)))
        group = group[~group["summary_key"].isin(selected)]
        group = group.sort_values(
            ["matches_both", "matches_one", "distance"],
            ascending=[False, False, True],
            kind="stable",
        )
        chosen = group.head(summaries_per_rating)
        keys = chosen["summary_key"].tolist()
        selected.update(keys)
        selected_by_rating[rating] = keys
    return selected, selected_by_rating


def segment_cost(scores, labels, start, end, bin_rating):
    """Absolute ordinal loss for one contiguous score segment."""
    return float(np.abs(labels[start:end] - bin_rating).sum())


def fit_integer_mapping(scores, labels):
    """Fit five ordered score bins by dynamic programming over score values."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    grouped_labels = [labels[inverse == index] for index in range(len(unique_scores))]
    group_count = len(unique_scores)
    if group_count < 5:
        raise ValueError(
            f"Need at least five distinct automatic scores; found {group_count}."
        )

    costs = np.full((6, group_count + 1), np.inf)
    choices = np.full((6, group_count + 1), -1, dtype=int)
    costs[0, 0] = 0.0
    for bin_rating in range(1, 6):
        for end in range(bin_rating, group_count + 1):
            for start in range(bin_rating - 1, end):
                if not np.isfinite(costs[bin_rating - 1, start]):
                    continue
                segment_labels = np.concatenate(grouped_labels[start:end])
                candidate = costs[bin_rating - 1, start] + np.abs(
                    segment_labels - bin_rating
                ).sum()
                if candidate < costs[bin_rating, end]:
                    costs[bin_rating, end] = candidate
                    choices[bin_rating, end] = start

    boundaries = []
    end = group_count
    for bin_rating in range(5, 0, -1):
        start = choices[bin_rating, end]
        if start < 0:
            raise ValueError("Could not fit five ordered score bins.")
        boundaries.append((start, end))
        end = start
    boundaries.reverse()

    thresholds = []
    for (_, left_end), (right_start, _) in zip(boundaries, boundaries[1:]):
        left_score = unique_scores[left_end - 1]
        right_score = unique_scores[right_start]
        thresholds.append((left_score + right_score) / 2.0)

    return np.asarray(thresholds), boundaries, unique_scores, grouped_labels


def predict_rating(scores, thresholds):
    return np.digitize(np.asarray(scores, dtype=float), thresholds, right=False) + 1


def quadratic_weighted_kappa(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    observed = np.zeros((5, 5), dtype=float)
    for left, right in zip(actual, predicted):
        if 1 <= left <= 5 and 1 <= right <= 5:
            observed[left - 1, right - 1] += 1
    actual_hist = observed.sum(axis=1)
    predicted_hist = observed.sum(axis=0)
    expected = np.outer(actual_hist, predicted_hist)
    if observed.sum() == 0 or expected.sum() == 0:
        return np.nan
    expected /= observed.sum()
    weights = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16.0
    observed_loss = (weights * observed).sum()
    expected_loss = (weights * expected).sum()
    return 1.0 - observed_loss / expected_loss if expected_loss else np.nan


def prepare_rows(merged, dimension):
    rows = merged.loc[
        (merged["dimension"] == dimension)
        & (merged["join_status"] == "matched")
    ].copy()
    rows["automatic_score"] = pd.to_numeric(rows["automatic_score"], errors="coerce")
    rows["human_rating"] = pd.to_numeric(rows[dimension], errors="coerce")
    rows = rows.dropna(subset=["item_key", "automatic_score", "human_rating"])
    rows = rows[rows["human_rating"].between(1, 5)]
    return add_summary_key(rows)


def mapping_rows(dimension, thresholds, calibration_rows, boundaries, unique_scores, grouped_labels):
    output = []
    for index in range(5):
        lower = -np.inf if index == 0 else thresholds[index - 1]
        upper = np.inf if index == 4 else thresholds[index]
        start, end = boundaries[index]
        labels = np.concatenate(grouped_labels[start:end])
        output.append({
            "dimension": dimension,
            "bin": index + 1,
            "lower_score": lower,
            "upper_score": upper,
            "n_calibration_rows": int(len(labels)),
            "mean_human_rating": float(labels.mean()) if len(labels) else np.nan,
            "min_human_rating": float(labels.min()) if len(labels) else np.nan,
            "max_human_rating": float(labels.max()) if len(labels) else np.nan,
        })
    return output


def evaluate(rows, predictions, split):
    """Evaluate integer predictions against human ratings for one data split.

    MAE is mean absolute error in rating points:
        MAE = (1 / N) * sum_i |human_rating_i - predicted_rating_i|
    where N is the number of rows in this split.
    """
    if not len(rows):
        return {
            "split": split, "n_rows": 0, "accuracy": np.nan,
            "mae": np.nan, "quadratic_weighted_kappa": np.nan,
        }
    actual = rows["human_rating"].to_numpy(dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    return {
        "split": split,
        "n_rows": len(rows),
        "accuracy": float((actual == predicted).mean()),
        "mae": float(np.abs(actual - predicted).mean()),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(actual, predicted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-file", type=Path,
        default=Path("annotation_analysis/analysis_output/automatic_analysis/human_automatic_merged.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("annotation_analysis/analysis_output/calibration_mapping"),
    )
    parser.add_argument("--summaries-per-rating", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.summaries_per_rating < 1:
        raise ValueError("--summaries-per-rating must be at least 1.")

    merged = pd.read_csv(args.merged_file, low_memory=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_mappings = []
    all_metrics = []
    all_predictions = []
    prepared = {dimension: prepare_rows(merged, dimension) for dimension in DIMENSIONS}
    calibration_summaries, selected_by_rating = choose_joint_calibration_summaries(
        prepared, args.summaries_per_rating, args.seed
    )
    print(
        "Joint calibration summaries by rating: "
        + ", ".join(f"{rating}={len(selected_by_rating.get(rating, []))}" for rating in range(1, 6))
    )
    selected_records = []

    for dimension in DIMENSIONS:
        rows = prepared[dimension]
        if rows.empty:
            print(f"Skipping {dimension}: no matched rows.")
            continue
        selected = rows[rows["summary_key"].isin(calibration_summaries)].copy()
        selected_records.append(
            selected[["dimension", "summary_key", "item_key", "summary_id", "summary_model",
                      "human_rating"]].drop_duplicates()
        )
        rows["split"] = np.where(rows["summary_key"].isin(calibration_summaries), "calibration", "test")
        calibration = rows[rows["split"] == "calibration"]
        test = rows[rows["split"] == "test"]
        thresholds, boundaries, unique_scores, grouped_labels = fit_integer_mapping(
            calibration["automatic_score"], calibration["human_rating"]
        )
        calibration_prediction = predict_rating(calibration["automatic_score"], thresholds)
        test_prediction = predict_rating(test["automatic_score"], thresholds)
        rows["calibrated_rating"] = predict_rating(rows["automatic_score"], thresholds)
        rows["summaries_per_rating"] = args.summaries_per_rating
        rows["mapping_seed"] = args.seed
        all_predictions.append(rows)
        all_mappings.extend(
            mapping_rows(dimension, thresholds, calibration, boundaries, unique_scores, grouped_labels)
        )
        all_metrics.extend([
            {"dimension": dimension, **evaluate(calibration, calibration_prediction, "calibration")},
            {"dimension": dimension, **evaluate(test, test_prediction, "test")},
        ])
        print(
            f"{dimension}: {len(rows)} matched rows, "
            f"{selected['summary_key'].nunique()} calibration summaries, "
            f"{len(test)} held-out rows"
        )
        print("  thresholds:", ", ".join(f"{value:.6g}" for value in thresholds))

    pd.DataFrame(all_mappings, columns=OUTPUT_COLUMNS).to_csv(
        args.output_dir / "learned_integer_mappings.csv", index=False
    )
    pd.DataFrame(all_metrics).to_csv(
        args.output_dir / "mapping_evaluation.csv", index=False
    )
    if selected_records:
        pd.concat(selected_records, ignore_index=True).to_csv(
            args.output_dir / "selected_calibration_summaries.csv", index=False
        )
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            args.output_dir / "calibrated_predictions.csv", index=False
        )
    print(f"Wrote calibration outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
