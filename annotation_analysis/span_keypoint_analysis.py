"""Directly compare human span annotations with automatic statements/keypoints.

This is a transparent lexical alignment baseline.  Every proposed match and its
token overlap is written to CSV for manual audit; unmatched paraphrases should
be reviewed rather than silently counted as failures.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def tokens(text):
    return set(re.findall(r"[\wÀ-ÿ]+", str(text).lower()))


def similarity(left, right):
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0, 0.0, 0.0
    overlap = len(a & b)
    precision = overlap / len(a)
    recall = overlap / len(b)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compare(auto, human, auto_text_column, unit_columns, threshold, analysis_name):
    auto = auto[auto["join_status"] == "matched"].dropna(subset=["item_key", auto_text_column]).copy()
    human = human.dropna(subset=["span_text"]).copy()
    output_rows, summaries = [], []
    human_groups = {key: group for key, group in human.groupby(unit_columns)}
    auto_group_columns = unit_columns + (["keypoint_type"] if "keypoint_type" in auto.columns else [])
    for keys, auto_group in auto.groupby(auto_group_columns):
        unit_keys = keys[:len(unit_columns)] if isinstance(keys, tuple) else (keys,)
        human_group = human_groups.get(unit_keys[0] if len(unit_columns) == 1 else unit_keys)
        if human_group is None:
            continue
        keypoint_type = auto_group["keypoint_type"].iloc[0] if "keypoint_type" in auto_group else "not_applicable"
        # In overlap data, evaluate against each annotator independently.
        for annotator, gold in human_group.groupby("annotator"):
            candidate_edges = []
            for auto_index, automatic in auto_group.iterrows():
                for human_index, labelled in gold.iterrows():
                    precision, recall, f1 = similarity(automatic[auto_text_column], labelled["span_text"])
                    candidate_edges.append({
                        "auto_index": auto_index, "human_index": human_index,
                        "auto_text": automatic[auto_text_column], "human_span": labelled["span_text"],
                        "token_precision": precision, "token_recall": recall, "token_f1": f1,
                        "match": f1 >= threshold,
                    })
            edges = pd.DataFrame(candidate_edges)
            matched_auto = set(edges.loc[edges["match"], "auto_index"]) if not edges.empty else set()
            matched_human = set(edges.loc[edges["match"], "human_index"]) if not edges.empty else set()
            n_auto, n_human = len(auto_group), len(gold)
            precision = len(matched_auto) / n_auto if n_auto else np.nan
            recall = len(matched_human) / n_human if n_human else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            summaries.append({
                "analysis": analysis_name, "keypoint_type": keypoint_type,
                "item_key": unit_keys[0], "summary_id": unit_keys[1] if len(unit_columns) > 1 else "",
                "annotator": annotator, "automatic_units": n_auto, "human_units": n_human,
                "matched_automatic_units": len(matched_auto), "matched_human_units": len(matched_human),
                "precision": precision, "recall": recall, "f1": f1,
            })
            if not edges.empty:
                edges["analysis"] = analysis_name
                edges["keypoint_type"] = keypoint_type
                edges["item_key"] = unit_keys[0]
                edges["summary_id"] = unit_keys[1] if len(unit_columns) > 1 else ""
                edges["annotator"] = annotator
                output_rows.append(edges)
    details = pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()
    per_unit = pd.DataFrame(summaries)
    return details, per_unit


def aggregate(per_unit):
    if per_unit.empty:
        return pd.DataFrame()
    return per_unit.groupby(["analysis", "keypoint_type"]).agg(
        comparisons=("item_key", "size"),
        automatic_units=("automatic_units", "sum"),
        human_units=("human_units", "sum"),
        matched_automatic_units=("matched_automatic_units", "sum"),
        matched_human_units=("matched_human_units", "sum"),
    ).reset_index().assign(
        precision=lambda x: x.matched_automatic_units / x.automatic_units,
        recall=lambda x: x.matched_human_units / x.human_units,
    ).assign(
        f1=lambda x: 2 * x.precision * x.recall / (x.precision + x.recall),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/human_analysis"))
    parser.add_argument("--automatic-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/automatic_analysis"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("annotation_analysis/analysis_output/span_analysis"))
    parser.add_argument("--token-f1-threshold", type=float, default=0.50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(args.human_dir / "human_error_spans_raw.csv", dtype=str)
    transcript = pd.read_csv(args.human_dir / "human_transcript_labels_raw.csv", dtype=str)
    fp = pd.read_csv(args.automatic_dir / "automatic_factuality_fp_statements.csv", dtype=str)
    keypoints = pd.read_csv(args.automatic_dir / "automatic_coverage_keypoints.csv", dtype=str)

    error_details, error_units = compare(fp, errors, "statement", ["item_key", "summary_id"],
                                         args.token_f1_threshold, "factuality_error_spans")
    keypoint_details, keypoint_units = compare(keypoints, transcript, "keypoint", ["item_key"],
                                               args.token_f1_threshold, "coverage_keypoints")
    details = pd.concat([error_details, keypoint_details], ignore_index=True)
    units = pd.concat([error_units, keypoint_units], ignore_index=True)
    details.to_csv(args.output_dir / "span_keypoint_alignment_details.csv", index=False)
    units.to_csv(args.output_dir / "span_keypoint_alignment_by_unit.csv", index=False)
    aggregate(units).to_csv(args.output_dir / "span_keypoint_alignment_summary.csv", index=False)
    print(f"Wrote {len(units)} per-unit comparisons and {len(details)} proposed alignments.")


if __name__ == "__main__":
    main()
