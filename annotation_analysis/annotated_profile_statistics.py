"""Trace human-rated summary slots to their annotator-specific profile assignments."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "main_annotation/Results"
ASSIGNMENTS_DIR = ROOT / "main_annotation/annotator_datasets"
PILOT_ASSIGNMENTS = ROOT / "pilot/pilot_labelstudio_import/labelstudio_filtered.json"
OUTPUT_DIR = ROOT / "annotation_analysis/analysis_output/human_analysis"
DIMENSIONS = ("coherence", "factuality", "fluency", "relevance")

# Collapse the two severe profiles into their corresponding two-dimension
# profiles. Normalize the existing reversed label as well so counts combine.
PROFILE_ALIASES = {
    "severe_coherence_fluency_degradation": "low_coherence_low_fluency",
    "severe_factuality_relevance_degradation": "low_factuality_low_relevance",
    "low_relevance_low_factuality": "low_factuality_low_relevance",
}


def normalized_profile(profile: object) -> str:
    label = assignment_profile(profile)
    return PROFILE_ALIASES.get(label, label)


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_key(source: object, filename: object) -> tuple[str, str]:
    return str(source or "").strip().lower(), normalized_text(filename).replace("/", "\\").casefold()


def assignment_profile(value: object) -> str:
    """Decode normal labels and the occasional one-entry dict encoded as text."""
    if value is None:
        return "unassigned"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return "unassigned"
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict) and len(parsed) == 1:
            text = str(next(iter(parsed.values()))).strip()
    except (SyntaxError, ValueError):
        pass
    return "unassigned" if text.lower() in {"", "nan", "none"} else text


def parse_rating(value: object) -> float:
    if value is None or not str(value).strip():
        return np.nan
    try:
        parsed = json.loads(str(value))
        return float(parsed[0]["rating"])
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
        return np.nan


def load_assignments(result_path: Path) -> tuple[dict, Path]:
    if result_path.stem.startswith("Pilot_Annotation"):
        path = PILOT_ASSIGNMENTS
    else:
        path = ASSIGNMENTS_DIR / f"{result_path.stem}_annotations.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for record in records:
        key = normalized_key(record.get("source"), record.get("filename"))
        if key in mapping:
            raise ValueError(f"Duplicate assignment key {key} in {path}")
        mapping[key] = record
    return mapping, path


def read_annotated_summary_slots(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_rows, audit_rows = [], []
    for result_path in sorted(results_dir.glob("*.csv")):
        assignments, assignment_path = load_assignments(result_path)
        exports = pd.read_csv(result_path, dtype=str, encoding="utf-8")
        for export_index, export in exports.iterrows():
            lookup_key = normalized_key(export.get("source"), export.get("filename"))
            assignment = assignments.get(lookup_key)
            if assignment is None:
                audit_rows.append({
                    "results_file": result_path.name,
                    "assignment_file": str(assignment_path.relative_to(ROOT)),
                    "filename": export.get("filename", ""),
                    "source": export.get("source", ""),
                    "status": "no_assignment_match",
                })
                continue

            for slot in "abc":
                ratings = {dimension: parse_rating(export.get(f"{slot}_{dimension}"))
                           for dimension in DIMENSIONS}
                if not any(np.isfinite(value) for value in ratings.values()):
                    continue
                summary_text = export.get(f"summary_{slot}") or assignment.get(f"summary_{slot}") or ""
                assigned_text = assignment.get(f"summary_{slot}") or ""
                model = str(assignment.get(f"model_{slot}") or "").strip()
                profile = normalized_profile(assignment.get(f"profile_{slot}"))
                content_identity = "\x1f".join([
                    str(export.get("source", "")).strip().lower(),
                    normalized_text(export.get("filename")),
                    model,
                    normalized_text(summary_text),
                ])
                summary_key = hashlib.sha256(content_identity.encode("utf-8")).hexdigest()
                assignment_rows.append({
                    "summary_key": summary_key,
                    "source": str(export.get("source", "")).strip().lower(),
                    "filename": export.get("filename", ""),
                    "summary_model": model or "unknown",
                    "profile": profile,
                    "summary_id_in_export": f"summary_{slot}",
                    "summary_text": summary_text,
                    "results_file": result_path.name,
                    "annotation_id": export.get("annotation_id", export.get("id", export_index)),
                    "assignment_summary_matches_export": (
                        normalized_text(summary_text) == normalized_text(assigned_text)
                    ) if assigned_text else np.nan,
                    **ratings,
                })
                audit_rows.append({
                    "results_file": result_path.name,
                    "assignment_file": str(assignment_path.relative_to(ROOT)),
                    "filename": export.get("filename", ""),
                    "source": export.get("source", ""),
                    "summary_id_in_export": f"summary_{slot}",
                    "summary_model": model,
                    "profile": profile,
                    "status": "matched",
                    "assignment_summary_matches_export": (
                        normalized_text(summary_text) == normalized_text(assigned_text)
                    ) if assigned_text else np.nan,
                })

    return pd.DataFrame(assignment_rows), pd.DataFrame(audit_rows)


def aggregate_unique_summaries(annotations: pd.DataFrame) -> pd.DataFrame:
    identity = ["summary_key"]
    rows = []
    for summary_key, group in annotations.groupby(identity, sort=False, dropna=False):
        profiles = group.profile.dropna().unique().tolist()
        if len(profiles) != 1:
            raise ValueError(f"Profile mismatch across annotations for summary {summary_key}: {profiles}")
        row = {
            "summary_key": summary_key,
            "source": group.source.iloc[0],
            "filename": group.filename.iloc[0],
            "summary_model": group.summary_model.iloc[0],
            "profile": profiles[0],
            "summary_text": group.summary_text.iloc[0],
            "n_human_annotations": group.results_file.nunique(),
            "annotators": ";".join(sorted(group.results_file.dropna().unique())),
            "summary_ids_seen": ";".join(sorted(group.summary_id_in_export.dropna().unique())),
        }
        for dimension in DIMENSIONS:
            row[dimension] = pd.to_numeric(group[dimension], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def profile_statistics(summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(summaries)
    for profile, group in summaries.groupby("profile", sort=True, dropna=False):
        row = {
            "profile": profile,
            "n_annotated_summaries": len(group),
            "pct_of_annotated_summaries": 100.0 * len(group) / total if total else np.nan,
            "n_short": int(group.source.eq("short").sum()),
            "n_long": int(group.source.eq("long").sum()),
            "n_human_annotation_records": int(group.n_human_annotations.sum()),
            "n_generation_models": int(group.summary_model.nunique()),
        }
        for dimension in DIMENSIONS:
            output_dimension = "coverage" if dimension == "relevance" else dimension
            values = pd.to_numeric(group[dimension], errors="coerce").dropna()
            row[f"mean_{output_dimension}"] = values.mean()
            row[f"median_{output_dimension}"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    annotations, audit = read_annotated_summary_slots(args.results_dir)
    if annotations.empty:
        raise ValueError(f"No rated summary slots found in {args.results_dir}")
    summaries = aggregate_unique_summaries(annotations)
    stats = profile_statistics(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries.to_csv(args.output_dir / "annotated_summaries_with_profiles.csv", index=False)
    stats.to_csv(args.output_dir / "annotated_summary_profile_statistics.csv", index=False,
                 float_format="%.4f")
    audit.to_csv(args.output_dir / "annotated_profile_assignment_audit.csv", index=False)
    by_model = (summaries.pivot_table(index="profile", columns="summary_model",
                                      values="summary_key", aggfunc="count", fill_value=0)
                .reset_index())
    by_model.to_csv(args.output_dir / "annotated_profile_counts_by_model.csv", index=False)

    mismatches = int((audit.get("assignment_summary_matches_export", pd.Series(dtype=bool)) == False).sum())
    unmatched = int(audit.status.ne("matched").sum()) if "status" in audit else 0
    print(f"Human-rated summary slots across exports: {len(annotations)}")
    print(f"Unique annotated summaries: {len(summaries)}")
    print(f"Exact profiles represented: {len(stats)}")
    print(f"Assignment text mismatches: {mismatches}; unmatched exports: {unmatched}")
    print(f"Wrote profile statistics and trace files to {args.output_dir}")


if __name__ == "__main__":
    main()
