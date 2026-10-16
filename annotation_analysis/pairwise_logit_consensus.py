"""Run the thesis pairwise-preference logit on consensus human judgments.

This is the consensus-data companion to pairwise_logit_analysis.py.  Unlike
the legacy script, it never infers a ranking from the dimension scores: the
outcome is the consensus rank produced from explicit human rankings.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DIMENSIONS = ("coherence", "factuality", "fluency", "relevance")


def make_pairs(scores):
    rows, skipped_ties = [], 0
    for item_key, group in scores.groupby("item_key"):
        group = group.dropna(subset=["consensus_rank", *DIMENSIONS]).reset_index(drop=True)
        if len(group) != 3:
            continue
        for left_index, right_index in combinations(range(3), 2):
            left, right = group.loc[left_index], group.loc[right_index]
            rank_difference = left["consensus_rank"] - right["consensus_rank"]
            if rank_difference == 0:
                skipped_ties += 1
                continue
            # ``preference = 1`` means the left summary is preferred to the
            # right summary (lower consensus rank is better).  Retaining both
            # IDs and ranks makes every model row inspectable in the CSV.
            row = {
                "item_key": item_key,
                "left_summary_id": left["summary_id"],
                "left_summary_model": left["summary_model"],
                "right_summary_id": right["summary_id"],
                "right_summary_model": right["summary_model"],
                "left_consensus_rank": left["consensus_rank"],
                "right_consensus_rank": right["consensus_rank"],
                "preference": int(rank_difference < 0),
            }
            for dimension in DIMENSIONS:
                row[f"diff_{dimension}"] = left[dimension] - right[dimension]
            rows.append(row)
    return pd.DataFrame(rows), skipped_ties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("annotation_analysis/analysis_output/human_analysis/human_scores_aggregated.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/pairwise_logit_consensus"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.input)
    for column in ["consensus_rank", *DIMENSIONS]:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    pairs, skipped_ties = make_pairs(scores)
    predictors = [f"diff_{dimension}" for dimension in DIMENSIONS]
    pairs = pairs.dropna(subset=predictors)
    if pairs.empty or pairs["preference"].nunique() < 2:
        raise ValueError("No usable preference pairs with both outcomes.")

    x_raw = pairs[predictors]
    x_scaled = (x_raw - x_raw.mean()) / x_raw.std(ddof=1).replace(0, 1)
    x = sm.add_constant(x_scaled)
    model = sm.Logit(pairs["preference"], x)
    result = model.fit(method="bfgs", maxiter=500, disp=False,
                       cov_type="cluster", cov_kwds={"groups": pairs["item_key"]})
    table = pd.DataFrame({
        "predictor": result.params.index,
        "odds_ratio_per_sd": np.exp(result.params),
        "ci_95_low": np.exp(result.params - 1.96 * result.bse),
        "ci_95_high": np.exp(result.params + 1.96 * result.bse),
        "p_value": result.pvalues,
    })
    table["predictor"] = table["predictor"].str.replace("diff_", "", regex=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "pairwise_consensus_data.csv", index=False)
    table.to_csv(args.output_dir / "pairwise_logit_consensus_results.csv", index=False)
    (args.output_dir / "pairwise_logit_consensus_summary.txt").write_text(result.summary().as_text(), encoding="utf-8")
    print(f"Fitted {len(pairs)} pairs from {pairs['item_key'].nunique()} transcripts; skipped {skipped_ties} tied pairs.")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
