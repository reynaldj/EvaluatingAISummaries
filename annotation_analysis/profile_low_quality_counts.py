"""Count low human ratings by the study's profile categories and dimension."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from human_analysis import profile_categories


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "annotation_analysis/analysis_output/human_analysis/human_scores_aggregated.csv"
DEFAULT_OUTPUT = ROOT / "annotation_analysis/analysis_output/human_analysis/low_quality_counts_by_profile.csv"
SPECIAL_PROFILES = (
    "all_degraded",
)
OUTPUT_DIMENSIONS = ("coherence", "factuality", "fluency", "coverage")


def expanded_profile_categories(value):
    """Keep established categories and fold severe profiles into their dimensions."""
    text = str(value).lower()
    categories = profile_categories(value)
    categories.extend(category for category in SPECIAL_PROFILES if category in text)
    if "severe_coherence_fluency_degradation" in text:
        categories.extend(("low_coherence", "low_fluency"))
    if "severe_factuality_relevance_degradation" in text:
        categories.extend(("low_factuality", "low_coverage"))
    categories = ["low_coverage" if category == "low_coverage" else category
                  for category in categories]
    return list(dict.fromkeys(categories))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-low-rating", type=float, default=2.0,
                        help="Maximum 1-5 consensus rating counted as low quality (default: 2).")
    args = parser.parse_args()

    summaries = pd.read_csv(args.input_file, low_memory=False)
    summaries["profile_category"] = summaries.degradation_profile.map(expanded_profile_categories)
    annotated_summary_count = summaries[["item_key", "summary_id"]].drop_duplicates().shape[0]
    summaries = summaries.explode("profile_category").dropna(subset=["profile_category"])

    rows = []
    for profile, profile_rows in summaries.groupby("profile_category", sort=True):
        record = {
            "profile_category": profile,
            "n_profile_summaries": int(
                profile_rows[["item_key", "summary_id"]].drop_duplicates().shape[0]
            ),
            "low_rating_threshold": args.max_low_rating,
        }
        for output_dimension in OUTPUT_DIMENSIONS:
            dimension = "relevance" if output_dimension == "coverage" else output_dimension
            ratings = pd.to_numeric(profile_rows[dimension], errors="coerce").dropna()
            if ratings.empty:
                record[f"n_rated_{output_dimension}"] = 0
                record[f"n_low_{output_dimension}"] = 0
                record[f"pct_low_{output_dimension}"] = float("nan")
                continue
            n_low = int(ratings.le(args.max_low_rating).sum())
            record[f"n_rated_{output_dimension}"] = int(ratings.size)
            record[f"n_low_{output_dimension}"] = n_low
            record[f"pct_low_{output_dimension}"] = 100.0 * n_low / len(ratings)
        rows.append(record)

    output = pd.DataFrame(rows)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False, float_format="%.4f")
    print(f"Annotated summaries: {annotated_summary_count}")
    print(f"Low-quality rule: human consensus score <= {args.max_low_rating}/5")
    print(f"Wrote {len(output)} profile/dimension rows to {args.output_file}")


if __name__ == "__main__":
    main()
