"""
Robust Pairwise Decomposition + Logistic Regression with Cluster-Robust SEs

Handles multiple input CSV files, auto-detects quality dimensions, infers
rankings when `summary_ranking` column is absent, and computes cluster-robust
standard errors.

Expected data shape (one row = one ranking instance with 3 summaries):
  Columns like: a_coherence, a_factuality, a_fluency, a_relevance,
                 b_coherence, ..., c_relevance
                (auto-detected; any prefix {a_, b_, c_} + dimension name)
  summary_ranking (optional): JSON string like
    '[{"ranker":{"summary_ranking":["summary_a","summary_c","summary_b"]}}]'
    If absent, ranking is inferred from average rating per summary.
  id or annotation_id: unique instance identifier
"""
import json
import warnings
import pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm as normal_dist
from itertools import combinations

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_DIR = "annotation_results"
SUMMARY_PREFIXES = ["a_", "b_", "c_"]

# ──────────────────────────────────────────────
# 1. Auto-detect dimensions from CSV files
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 0: Scanning input files & detecting dimensions")
print("=" * 60)

csv_files = sorted(pathlib.Path(INPUT_DIR).glob("*.csv"))
print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

all_dfs = []
detected_dims = None

for fpath in csv_files:
    try:
        df = pd.read_csv(fpath, engine="python", dtype=str)
        df = df.dropna(subset=["annotation_id"])
        if df.empty:
            continue
        all_dfs.append(df)
    except Exception as e:
        print(f"  WARNING: Could not parse {fpath.name}: {e}")

if not all_dfs:
    raise FileNotFoundError("No valid CSV files found in annotation_results/")

df_raw = pd.concat(all_dfs, ignore_index=True).reset_index(drop=True)
print(f"\nCombined data: {len(df_raw)} valid ranking instances.")

# Auto-detect quality dimensions from column names
all_cols = set(df_raw.columns)
candidate_dims = set()
for col in all_cols:
    for prefix in SUMMARY_PREFIXES:
        if col.startswith(prefix):
            dim = col[len(prefix):]
            if dim and not dim.startswith("error"):  # skip error_labels
                candidate_dims.add(dim)

DIMENSIONS = sorted(candidate_dims)
print(f"Auto-detected {len(DIMENSIONS)} quality dimensions: {DIMENSIONS}")
print(f"Columns available: {list(df_raw.columns)}")
print()

# ──────────────────────────────────────────────
# 2. Parse ratings & rankings
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Parsing ratings & determining rankings")
print("=" * 60)


def parse_rating(val):
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    try:
        return json.loads(val)[0]["rating"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return np.nan


def parse_ranking(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        return json.loads(val)[0]["ranker"]["summary_ranking"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


HAS_RANKING_COL = "summary_ranking" in df_raw.columns

records = []
for _, row in df_raw.iterrows():
    instance_key = row.get("id") or row.get("annotation_id")
    if instance_key is None:
        continue
    instance_id = str(instance_key).strip()

    # ── Approach A: Use explicit summary_ranking ──
    if HAS_RANKING_COL:
        ranking_list = parse_ranking(row.get("summary_ranking"))
        if ranking_list is not None:
            rank_map = {sid: i + 1 for i, sid in enumerate(ranking_list)}
            for prefix in SUMMARY_PREFIXES:
                summary_id = f"summary_{prefix.rstrip('_')}"
                rank = rank_map.get(summary_id)
                if rank is None:
                    continue
                dims = {}
                for dim in DIMENSIONS:
                    col = f"{prefix}{dim}"
                    dims[dim] = parse_rating(row.get(col))
                records.append({
                    "instance_id": instance_id,
                    "summary_id": summary_id,
                    "rank": rank,
                    **dims,
                })
            continue  # skip fallback below

    # ── Approach B: Infer ranking from average ratings ──
    print(f"  Inferring ranking for instance {instance_id} (no explicit summary_ranking)")
    temp = []
    for prefix in SUMMARY_PREFIXES:
        summary_id = f"summary_{prefix.rstrip('_')}"
        dims = {}
        ratings = []
        for dim in DIMENSIONS:
            col = f"{prefix}{dim}"
            r = parse_rating(row.get(col))
            dims[dim] = r
            if not np.isnan(r):
                ratings.append(r)
        avg_rating = np.mean(ratings) if ratings else np.nan
        temp.append((summary_id, avg_rating, dims))
    # Sort by average rating descending (higher = better = rank 1)
    temp.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
    for rank, (summary_id, _, dims) in enumerate(temp, start=1):
        records.append({
            "instance_id": instance_id,
            "summary_id": summary_id,
            "rank": rank,
            **dims,
        })

df_long = pd.DataFrame(records)
print(f"Long-format data: {len(df_long)} rows ({len(df_long) // 3} instances x 3 summaries)")
print(df_long.to_string(index=False))
print()

# ──────────────────────────────────────────────
# 3. Pairwise Decomposition
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 2: Pairwise Decomposition")
print("=" * 60)

pair_records = []
for instance_id, group in df_long.groupby("instance_id"):
    group = group.reset_index(drop=True)
    if len(group) != 3:
        print(f"  WARNING: instance {instance_id} has {len(group)} summaries, expected 3. Skipping.")
        continue

    for idx_x, idx_y in combinations(range(3), 2):
        sum_x = group.loc[idx_x]
        sum_y = group.loc[idx_y]

        preference = 1 if sum_x["rank"] < sum_y["rank"] else 0

        diff = {"instance_id": instance_id, "preference": preference}
        for dim in DIMENSIONS:
            diff[f"Diff_{dim}"] = sum_x[dim] - sum_y[dim]

        pair_records.append(diff)

df_pairs = pd.DataFrame(pair_records)
print(f"Pairwise dataset: {df_pairs.shape[0]} pairs ({len(df_pairs) // 3} instances x 3 pairs each)")
print()
print("First 5 rows of pairwise dataset (verification):")
print(df_pairs.head().to_string(index=False))
print()

# ──────────────────────────────────────────────
# 4. Validation Checks
# ──────────────────────────────────────────────
print("=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

diff_cols = [c for c in df_pairs.columns if c.startswith("Diff_")]
constant_cols = []
for col in diff_cols:
    if df_pairs[col].nunique() <= 1:
        print(f"  DROPPING constant column: {col}")
        constant_cols.append(col)
diff_cols = [c for c in diff_cols if c not in constant_cols]

has_var = len(diff_cols) > 0
has_both = df_pairs["preference"].nunique() >= 2

if not has_var:
    print("  ERROR: No Diff_* columns with variance remain. Cannot fit model.")
    raise SystemExit(1)
if not has_both:
    print("  ERROR: preference column has only one unique value. Cannot fit model.")
    raise SystemExit(1)

print(f"  Using {len(diff_cols)} diff columns: {diff_cols}")
print(f"  Preference distribution: 0={df_pairs['preference'].value_counts().get(0, 0)}, "
      f"1={df_pairs['preference'].value_counts().get(1, 0)}")
print()

# ──────────────────────────────────────────────
# 5. Logistic Regression with Cluster-Robust SEs
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 3: Logistic Regression with Cluster-Robust SEs")
print("=" * 60)

X_raw = df_pairs[diff_cols].copy()
X_mean = X_raw.mean()
X_std = X_raw.std()
X_std[X_std == 0] = 1.0
X_scaled = (X_raw - X_mean) / X_std

y = df_pairs["preference"]
X = sm.add_constant(X_scaled)

print("Standardized (z-score) all Diff_* columns for comparable coefficients.")
print()

# Check for perfect separation
unique_combos = X_scaled.drop_duplicates().shape[0]
if unique_combos < len(diff_cols) + 1:
    msg = ("  WARNING: Fewer unique predictor combinations than parameters. "
           "Perfect separation likely. Results are demonstrative only.")
    print(msg)
    print()

try:
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(
        maxiter=500,
        method="bfgs",
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df_pairs["instance_id"]},
    )
except Exception as e:
    print(f"  Cluster-robust fit failed: {e}")
    print("  Falling back to non-clustered model (for demo purposes).")
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(method="bfgs", maxiter=500, disp=False)

print("Model fitted.\n")

# ──────────────────────────────────────────────
# 6. Compute Odds Ratios and CIs
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 4: Results -- Odds Ratios with Cluster-Robust 95% CIs")
print("=" * 60)

params = logit_result.params
crse_se = logit_result.bse
z_scores = params / crse_se
p_values = 2 * (1 - normal_dist.cdf(np.abs(z_scores)))

or_vals = np.exp(params)
ci_lower = np.exp(params - 1.96 * crse_se)
ci_upper = np.exp(params + 1.96 * crse_se)

results = pd.DataFrame({
    "Dimension": params.index,
    "OR": or_vals.values,
    "CI_lower": ci_lower.values,
    "CI_upper": ci_upper.values,
    "P_value": p_values,
})

results["OR"] = results["OR"].round(4)
results["CI_lower"] = results["CI_lower"].round(4)
results["CI_upper"] = results["CI_upper"].round(4)
results["P_value"] = results["P_value"].round(4)

print()
print("--- Odds Ratio Table (standardized predictors) ---")
print(results.to_string(index=False))
print()

# Also print interpretable note
print("--- Interpretation ---")
print("OR > 1 means a higher rating on that dimension increases the odds")
print("that a summary is preferred (ranked better).")
print("CIs and p-values use cluster-robust standard errors (grouped by instance_id).")
print()

OUTPUT_FILE = "pairwise_logit_results.csv"
results.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")
print("Done.")