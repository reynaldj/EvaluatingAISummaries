"""Compute evaluator-model correlations on the short, reference-matched cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t as student_t

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "annotation_analysis/analysis_output/reference_baselines"
AUTOMATIC_DIR = ROOT / "annotation_analysis/analysis_output/automatic_analysis"


def correlation_intervals(frame: pd.DataFrame, n_resamples: int = 1000, seed: int = 42):
    """Cluster bootstrap correlations, returning percentile CIs and bootstrap SEs."""
    keys = frame.item_key.dropna().unique()
    if len(keys) < 2:
        return (np.nan, np.nan), (np.nan, np.nan), np.nan, np.nan
    clusters = {key: np.flatnonzero(frame.item_key.to_numpy() == key) for key in keys}
    automatic = frame.automatic_score.to_numpy(dtype=float)
    human = frame[frame.name_dimension.iloc[0]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rho_draws, tau_draws = [], []
    for _ in range(n_resamples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([clusters[key] for key in selected])
        rho = spearmanr(automatic[indices], human[indices]).statistic
        tau = kendalltau(automatic[indices], human[indices], variant="b").statistic
        if np.isfinite(rho):
            rho_draws.append(rho)
        if np.isfinite(tau):
            tau_draws.append(tau)
    rho_ci = tuple(np.quantile(rho_draws, [0.025, 0.975])) if rho_draws else (np.nan, np.nan)
    tau_ci = tuple(np.quantile(tau_draws, [0.025, 0.975])) if tau_draws else (np.nan, np.nan)
    rho_se = float(np.std(rho_draws, ddof=1)) if len(rho_draws) > 1 else np.nan
    tau_se = float(np.std(tau_draws, ddof=1)) if len(tau_draws) > 1 else np.nan
    return rho_ci, tau_ci, rho_se, tau_se


def cluster_wald_pvalue(estimate: float, standard_error: float, n_clusters: int) -> float:
    """Two-sided Wald p-value using a transcript-cluster bootstrap SE."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error) or standard_error <= 0 or n_clusters < 2:
        return np.nan
    return float(2 * student_t.sf(abs(estimate / standard_error), df=n_clusters - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-file", type=Path, default=BASELINE_DIR / "baseline_dataset.csv")
    parser.add_argument("--merged-file", type=Path, default=AUTOMATIC_DIR / "human_automatic_merged.csv")
    parser.add_argument("--output-file", type=Path,
                        default=AUTOMATIC_DIR / "metric_validity_by_evaluator_model_short.csv")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort_file, usecols=["item_key", "source"])
    if not cohort.source.eq("short").all():
        raise ValueError("The baseline reference-matched cohort must contain only short summaries.")
    short_keys = set(cohort.item_key.dropna())

    merged = pd.read_csv(args.merged_file, low_memory=False)
    matched = merged.loc[
        merged.join_status.eq("matched") & merged.item_key.isin(short_keys)
    ].copy()
    for column in ("automatic_score", "coherence", "factuality", "fluency", "relevance"):
        matched[column] = pd.to_numeric(matched[column], errors="coerce")

    results = []
    groups = ["dimension", "method_variant", "evaluator_model"]
    for keys, group in matched.groupby(groups, dropna=False, sort=True):
        dimension = keys[0]
        data = group.dropna(subset=["automatic_score", dimension, "item_key"])
        if len(data) < 3:
            continue
        automatic = data.automatic_score.astype(float)
        human = data[dimension].astype(float)
        rho, rho_p_iid = spearmanr(automatic, human)
        tau, tau_p_iid = kendalltau(automatic, human, variant="b")
        data = data.copy()
        data["name_dimension"] = dimension
        rho_ci, tau_ci, rho_se, tau_se = correlation_intervals(data)
        n_clusters = int(data.item_key.nunique())
        rho_p_cluster = cluster_wald_pvalue(rho, rho_se, n_clusters)
        tau_p_cluster = cluster_wald_pvalue(tau, tau_se, n_clusters)

        comparable = 1 + 4 * automatic if dimension in ("factuality", "relevance") else automatic
        results.append(dict(zip(groups, keys)) | {
            "n_summaries": len(data),
            "transcript_clusters": n_clusters,
            "spearman_rho": rho,
            "spearman_p": rho_p_cluster,
            "spearman_p_cluster_adjusted": rho_p_cluster,
            "spearman_p_iid": rho_p_iid,
            "spearman_cluster_bootstrap_se": rho_se,
            "rho_ci_95_low": rho_ci[0],
            "rho_ci_95_high": rho_ci[1],
            "kendall_tau_b": tau,
            "kendall_p": tau_p_cluster,
            "kendall_p_cluster_adjusted": tau_p_cluster,
            "kendall_p_iid": tau_p_iid,
            "kendall_cluster_bootstrap_se": tau_se,
            "tau_ci_95_low": tau_ci[0],
            "tau_ci_95_high": tau_ci[1],
            "mae_1_to_5": float(np.mean(np.abs(comparable - human))),
            "bias_1_to_5": float(np.mean(comparable - human)),
        })

    result = pd.DataFrame(results)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_file, index=False)
    present_keys = set(matched.item_key.dropna())
    print(f"Short reference-matched transcript keys: {len(short_keys)}")
    print(f"Keys represented in matched evaluator results: {len(present_keys)}")
    print(f"Matched evaluator rows analyzed: {len(matched)}")
    print(f"Wrote {len(result)} grouped validity rows to {args.output_file}")


if __name__ == "__main__":
    main()
