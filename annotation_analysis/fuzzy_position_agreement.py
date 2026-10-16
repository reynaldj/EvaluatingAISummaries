"""Localize automatic items by fuzzy alignment, then compare source token positions.

Run from the repository root with Python. Inputs are read-only.
Human spans are never passed to the locator. Thresholds are exploratory defaults.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import rapidfuzz

from automatic_analysis import attach_to_humans, load_factuality, load_relevance
from human_analysis import (aggregate_scores, build_raw_tables, normalized_filename,
                            parse_json, read_exports)
from fuzzy_span_alignment import SourceText, localize, positional_scores, validate_human_span


FACTUALITY_EVALUATORS = {
    "results-factuality-evaluation-llama8b": "llama-3.1-8b",
    "results-factuality-evaluation-llama70b-correct": "llama-70b",
    "results-factuality-evaluation-gemma-12b": "gemma-12b",
    "results-factuality-evaluation-gemma-31b": "gemma-31b",
}
COVERAGE_EVALUATORS = {"gemma12b": "gemma-12b", "gemma31b": "gemma-31b",
                       "llama-3.1-8b": "llama-3.1-8b", "llama-70b": "llama-70b"}
GROUP = ["analysis", "evaluator_model", "keypoint_type"]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def save_csv(rows, path, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def source_documents(exports):
    documents = {}
    for _, row in exports.fillna("").iterrows():
        item = f"{str(row['source']).strip().lower()}::{str(row['filename']).strip()}"
        for field in ("transcript", "summary_a", "summary_b", "summary_c"):
            text = str(row[field])
            kind = "transcript" if field == "transcript" else "summary"
            document_id = digest(json_text([item, kind, text]))
            document = documents.setdefault(document_id, {
                "document_id": document_id, "item_key": item,
                "field": kind, "text_sha256": digest(text), "text": text, "export_locations": [],
            })
            location = {"input_file": row["input_file"], "annotator": str(row["annotator"]), "slot": field}
            if location not in document["export_locations"]:
                document["export_locations"].append(location)
    return documents


def human_references(exports, documents):
    """Include completed units with no highlighted spans; validate every actual span."""
    references = defaultdict(dict)
    audit = []
    for _, row in exports.fillna("").iterrows():
        item = f"{str(row['source']).strip().lower()}::{str(row['filename']).strip()}"
        for field, label_field in (("transcript", "transcript_labels"), ("summary_a", "a_error_labels"),
                                   ("summary_b", "b_error_labels"), ("summary_c", "c_error_labels")):
            kind = "transcript" if field == "transcript" else "summary"
            document_id = digest(json_text([item, kind, str(row[field])]))
            annotator = str(row["annotator"])
            source = documents[document_id]
            ref = references[document_id].setdefault(annotator, {"spans": [], "invalid": 0})
            for span in parse_json(row[label_field], []):
                try:
                    start, end, status = validate_human_span(source["text"], span.get("start"), span.get("end"), span.get("text", ""))
                except (TypeError, ValueError, OverflowError):
                    start, end, status = None, None, "invalid_offsets_or_text"
                audit.append({
                    "document_id": document_id, "item_key": item, "field": field,
                    "annotator": annotator, "label": "; ".join(span.get("labels", [])),
                    "original_start": span.get("start"), "original_end": span.get("end"),
                    "start": start, "end": end, "span_text": span.get("text", ""), "status": status,
                })
                if start is None:
                    ref["invalid"] += 1
                else:
                    ref["spans"].append((start, end))
    for annotators in references.values():
        for ref in annotators.values():
            ref["spans"] = sorted(set(ref["spans"]))
    return references, audit


def human_join_table(exports, args):
    """Resolve shuffled A/B/C slots by exact source-text identity, not slot metadata.

    Exact identity here establishes which *document* was evaluated; extraction
    localization still always uses fuzzy alignment.
    """
    metadata = defaultdict(set)
    paths = sorted(args.assignment_dir.glob("*_annotations.json")) + sorted(args.pilot_assignment_dir.glob("*.json"))
    for path in paths:
        for record in json.loads(path.read_text(encoding="utf-8")):
            for slot in ("a", "b", "c"):
                if not record.get(f"model_{slot}") or not record.get(f"profile_{slot}"):
                    continue  # Label Studio import copies may omit metadata entirely.
                key = (str(record.get("source", "")).lower(), normalized_filename(record.get("filename", "")), record.get(f"summary_{slot}", ""))
                metadata[key].add(tuple(str(record.get(f"{field}_{slot}", "")) for field in ("model", "profile", "degraded")))
    ratings, _, _ = build_raw_tables(exports, {})
    audit = []
    for index, row in ratings.iterrows():
        key = (row["source"], normalized_filename(row["filename"]), row["summary_text"])
        candidates = metadata.get(key, set())
        status = "matched" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
        document_id = digest(json_text([row["item_key"], "summary", row["summary_text"]]))
        audit.append({"document_id": document_id, "item_key": row["item_key"], "input_file": row["input_file"],
                      "original_slot": row["summary_id"], "annotator": row["annotator"], "metadata_status": status})
        ratings.at[index, "summary_id"] = document_id
        if status == "matched":
            for column, value in zip(("summary_model", "degradation_profile", "degraded_dimension"), next(iter(candidates))):
                ratings.at[index, column] = value
        else:
            ratings.at[index, "degradation_profile"] = "__unresolved_metadata__"
    return aggregate_scores(ratings), audit


def attach_with_source(frame, human):
    """Reuse the existing model/profile joins, additionally isolating long/short sources."""
    if frame.empty:
        return frame
    parts = []
    for source, group in frame.groupby("source_hint", dropna=False):
        subset = human if not source or pd.isna(source) else human[human["source"].eq(source)]
        parts.append(attach_to_humans(group, subset))
    return pd.concat(parts, ignore_index=True)


def automatic_frames(args, human):
    frames = []
    for folder, evaluator in FACTUALITY_EVALUATORS.items():
        if args.evaluators and evaluator not in args.evaluators:
            continue
        path = args.factuality_root / folder
        if not path.is_dir():
            continue
        metrics, units = load_factuality(path)
        for frame in (metrics, units):
            frame["evaluator_model"] = evaluator
        frames.append(("factuality_error_spans", metrics, units, "statement"))
    metrics, units = load_relevance(args.coverage_root)
    for frame in (metrics, units):
        if frame.empty:
            continue
        frame["evaluator_model"] = frame["metric_file"].map(
            lambda p: COVERAGE_EVALUATORS.get(Path(p).parent.parent.name, ""))
    if not metrics.empty:
        metrics = metrics[metrics["evaluator_model"].ne("")]
        units = units[units["evaluator_model"].ne("")] if not units.empty else units
        if args.evaluators:
            metrics = metrics[metrics["evaluator_model"].isin(args.evaluators)]
            units = units[units["evaluator_model"].isin(args.evaluators)] if not units.empty else units
        frames.append(("coverage_keypoints", metrics, units, "keypoint"))
    for analysis, metrics, units, text_column in frames:
        print(f"Joining {analysis}: {len(metrics)} evaluations, {len(units)} items...", flush=True)
        yield analysis, attach_with_source(metrics, human), attach_with_source(units, human), text_column


def extraction_sets(args, human, documents):
    sets, join_audit, issues = {}, [], []
    transcripts = {}
    for doc in documents.values():
        if doc["field"] == "transcript":
            if doc["item_key"] in transcripts:
                raise ValueError(f"Multiple transcript versions for {doc['item_key']}")
            transcripts[doc["item_key"]] = doc
    keys = ["evaluator_model", "item_key", "summary_id", "keypoint_type", "metric_file"]
    for analysis, metrics, units, column in automatic_frames(args, human):
        for name, frame in (("evaluations", metrics), ("items", units)):
            if frame.empty:
                continue
            for (evaluator, status), group in frame.groupby(["evaluator_model", "join_status"]):
                join_audit.append({"analysis": analysis, "evaluator_model": evaluator,
                                   "table": name, "join_status": status, "rows": len(group)})
        if metrics.empty:
            continue
        matched_units = units[units["join_status"].eq("matched")] if not units.empty else units
        lookup = {key: group[column].dropna().astype(str).tolist()
                  for key, group in matched_units.groupby(keys)} if not matched_units.empty else {}
        for _, row in metrics[metrics["join_status"].eq("matched")].iterrows():
            field = "transcript" if analysis == "coverage_keypoints" else "summary"
            document = transcripts[row["item_key"]] if field == "transcript" else documents[row["summary_id"]]
            if documents[row["summary_id"]]["text"] != row["summary_text"]:
                raise ValueError("Joined summary differs from original annotation text")
            texts = sorted(set(t.strip() for t in lookup.get(tuple(row[k] for k in keys), []) if t.strip()))
            count = row.get("fp_count") if analysis == "factuality_error_spans" else row.get("total_count")
            if not texts and pd.notna(count) and float(count) > 0:
                issues.append({"analysis": analysis, "item_key": row["item_key"],
                               "metric_file": row["metric_file"], "status": "missing_extracted_items"})
                continue
            # Identical transcript extraction sets reused across summaries count once.
            set_id = digest(json_text([analysis, row["evaluator_model"], row["keypoint_type"], document["document_id"], texts]))
            entry = sets.setdefault(set_id, {
                "set_id": set_id, "analysis": analysis, "evaluator_model": row["evaluator_model"],
                "keypoint_type": row["keypoint_type"], "item_key": row["item_key"],
                "summary_id": "" if field == "transcript" else row["summary_id"],
                "document_id": document["document_id"], "field": field, "texts": texts,
                "provenance": [],
            })
            provenance = {k: str(row.get(k, "")) for k in ("metric_file", "summary_id", "summary_model", "profile")}
            if provenance not in entry["provenance"]:
                entry["provenance"].append(provenance)
    return list(sets.values()), join_audit, issues


def summarize(comparisons, n_bootstrap):
    rows = []
    valid = comparisons[comparisons["comparison_status"].eq("scored")]
    for keys, group in valid.groupby(GROUP):
        a, h, overlap = (int(group[c].sum()) for c in ("automatic_tokens", "human_tokens", "overlap_tokens"))
        # Average variants/annotators within transcript before averaging transcripts.
        per_transcript = group.groupby("item_key")["f1"].mean().dropna()
        rng = np.random.default_rng(42)
        if len(per_transcript) >= 2 and n_bootstrap:
            values = per_transcript.to_numpy()
            samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
        else:
            low, high = np.nan, np.nan
        rows.append(dict(zip(GROUP, keys)) | {
            "n_comparisons": len(group), "n_transcripts": group["item_key"].nunique(),
            "n_empty_both": int(((group["automatic_tokens"] == 0) & (group["human_tokens"] == 0)).sum()),
            "automatic_tokens": a, "human_tokens": h, "overlap_tokens": overlap,
            "micro_precision": overlap / a if a else np.nan,
            "micro_recall": overlap / h if h else np.nan,
            "micro_f1": 2 * overlap / (a + h) if a + h else np.nan,
            "macro_f1": per_transcript.mean(), "macro_f1_ci_95_low": low, "macro_f1_ci_95_high": high,
        })
    return rows


def localization_summary(localizations, sets):
    rows = []
    for keys, group in pd.DataFrame(sets).groupby(GROUP):
        subset = localizations[localizations["set_id"].isin(group["set_id"])]
        counts = subset["status"].value_counts().to_dict()
        rows.append(dict(zip(GROUP, keys)) | {
            "extraction_sets": len(group), "automatic_items": len(subset),
            "accepted_items": int(subset["accepted"].sum()),
            "localization_rate": subset["accepted"].mean() if len(subset) else np.nan,
            **{status: int(counts.get(status, 0)) for status in
               ("located", "multiple_locations", "ambiguous", "below_threshold", "empty_query_or_source")},
        })
    return rows


def highlight(text, lo, hi, automatic, human):
    boundaries = {lo, hi}
    for start, end in automatic + human:
        boundaries.update((max(lo, min(hi, start)), max(lo, min(hi, end))))
    parts = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        a = any(left < end and right > start for left, right in automatic)
        h = any(left < end and right > start for left, right in human)
        cls = "both" if a and h else "auto" if a else "human" if h else ""
        part = html.escape(text[start:end])
        parts.append(f'<mark class="{cls}">{part}</mark>' if cls else part)
    return "".join(parts)


def write_report(path, summary, localization, rows, documents, references, title="Fuzzy positional agreement"):
    by_id = {doc["document_id"]: doc for doc in documents.values()}
    sections = ["""<!doctype html><html lang="en"><meta charset="utf-8"><title>Fuzzy positional agreement</title>
    <style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}
    td,th{padding:7px;border:1px solid #ccc}article{border-top:1px solid #bbb;padding:18px 0}.context{white-space:pre-wrap;line-height:1.8}
    mark.auto{background:#ffd39a}mark.human{background:#b6dcff}mark.both{background:#d6b5f5}</style>
    <h1>Fuzzy positional agreement</h1><p>Exploratory thresholds; localization is blind to human highlights.
    Scores cover accepted source positions. Unlocated items are reported separately, not counted as token false positives.
    Multiple locations are counted once per item for localization rate; selected word positions are unioned for scoring.
    Empty-both comparisons have undefined F1.</p>
    <p>Orange: automatic candidate. Blue: human. Purple: overlap. A rejected candidate is shown for inspection only.</p>"""]
    sections[0] = sections[0].replace("Fuzzy positional agreement", html.escape(title))
    if "selected_span_count" in rows and rows["selected_span_count"].le(1).all():
        sections[0] = sections[0].replace(
            "Multiple locations are counted once per item for localization rate; selected word positions are unioned for scoring.",
            "Each accepted item contributes one highest-scoring qualifying location. Word positions are unioned across items for scoring.")
    if "semantic_attempted" in rows.columns:
        sections.append("<p>Semantic fallback uses an exploratory, uncalibrated cosine threshold. "
                        "Accepted fuzzy spans are unchanged. Semantic boundaries follow source sentences/clauses; "
                        "broader passages can change positional scores without more accurate localization. "
                        "Review semantic_localization_review.csv before using these results in a thesis.</p>")
    sections += ["<h2>Position scores</h2>", summary.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
                 "<h2>Localization</h2>", localization.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
                 "<h2>Audit examples</h2><p>Up to two items per evaluator/configuration/status; full results are in the CSV files.</p>"]
    sample_keys = GROUP + ["status"] + (["localization_method"] if "localization_method" in rows else [])
    samples = rows.groupby(sample_keys, sort=True).head(2) if len(rows) else rows
    for _, row in samples.iterrows():
        doc = by_id[row["document_id"]]
        sections.append(f"<article><b>{html.escape(row['evaluator_model'])} / {html.escape(row['analysis'])} / {html.escape(row['keypoint_type'])}</b>"
                        f"<p>{html.escape(row['item_key'])} — {row['status']} — accepted: {row['accepted']}</p>"
                        f"<p>Extracted: {html.escape(row['extracted_text'])}</p>")
        selected = json.loads(row.get("selected_spans_json", "[]"))
        display = selected or ([row] if pd.notna(row.get("start")) else [])
        positions = [(int(s["start"]), int(s["end"])) for s in display]
        for number, span in enumerate(display, 1):
            start, end = int(span["start"]), int(span["end"])
            semantic_score = span.get("semantic_similarity")
            score_text = (f"semantic cosine {semantic_score:.3f}" if pd.notna(semantic_score) else
                          f"fuzzy similarity {span['similarity']:.1f}; query character coverage {span['query_character_coverage']:.3f}")
            sections.append(f"<p>Location {number}/{len(display)}: offsets [{start}, {end}); {score_text}</p>")
            for annotator, reference in references[doc["document_id"]].items():
                excerpt = highlight(doc["text"], max(0, start - 140), min(len(doc["text"]), end + 140), positions, reference["spans"])
                sections.append(f"<p>Human annotator {html.escape(annotator)}</p><div class=context>{excerpt}</div>")
        sections.append("</article>")
    sections.append("</html>")
    path.write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("main_annotation/Results"))
    parser.add_argument("--assignment-dir", type=Path, default=Path("main_annotation/annotator_datasets"))
    parser.add_argument("--pilot-assignment-dir", type=Path, default=Path("pilot/pilot_labelstudio_import"))
    parser.add_argument("--factuality-root", type=Path, default=Path("seval-ex"))
    parser.add_argument("--coverage-root", type=Path, default=Path("relevance"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotation_analysis/analysis_output/fuzzy_position_agreement"))
    parser.add_argument("--min-similarity", type=float, default=65.0)
    parser.add_argument("--min-coverage", type=float, default=0.65)
    parser.add_argument("--ambiguity-margin", type=float, default=3.0)
    parser.add_argument("--ambiguity-policy", choices=("all", "exclude", "best"), default="best",
                        help="Keep one highest-scoring qualifying location by default; other policies reproduce old runs")
    parser.add_argument("--evaluators", nargs="+", choices=sorted(COVERAGE_EVALUATORS.values()))
    parser.add_argument("--limit-sets", type=int, help="Smoke-run limit; never use for final reported scores")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if not 0 <= args.min_similarity <= 100 or not 0 <= args.min_coverage <= 1 or args.ambiguity_margin < 0:
        parser.error("Similarity must be 0..100, coverage 0..1, ambiguity margin nonnegative")
    if args.bootstrap < 0 or (args.limit_sets is not None and args.limit_sets < 1):
        parser.error("bootstrap must be nonnegative and limit-sets positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading original annotation texts and validating human offsets...", flush=True)
    exports = read_exports(args.results_dir)
    human, metadata_audit = human_join_table(exports, args)
    save_csv(metadata_audit, args.output_dir / "human_document_metadata.csv")
    documents = source_documents(exports)
    references, audit = human_references(exports, documents)
    save_csv(audit, args.output_dir / "human_span_validation.csv")
    print(f"Validated {len(audit)} human spans; {sum(a['status'] == 'invalid_offsets_or_text' for a in audit)} invalid.", flush=True)
    print("Loading automatic extraction sets across evaluators...", flush=True)
    sets, joins, issues = extraction_sets(args, human, documents)
    sets.sort(key=lambda s: (s["analysis"], s["evaluator_model"], s["keypoint_type"], s["item_key"], s["set_id"]))
    if args.limit_sets:
        sets = sets[:args.limit_sets]
    if not sets:
        raise ValueError("No extraction sets matched the human documents")
    save_csv(joins, args.output_dir / "automatic_join_audit.csv")
    save_csv(issues, args.output_dir / "input_issues.csv", ["analysis", "item_key", "metric_file", "status"])
    save_csv([{k: json_text(v) if isinstance(v, list) else v for k, v in s.items()} for s in sets], args.output_dir / "extraction_sets.csv")
    with (args.output_dir / "source_documents.jsonl").open("w", encoding="utf-8") as handle:
        for doc in documents.values():
            handle.write(json_text(doc) + "\n")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update({"rapidfuzz_version": rapidfuzz.__version__, "algorithm": "windowed_partial_ratio_full_query_rescore_v1",
                     "factuality_parser": "claim_before_any_verdict_v2",
                     "location_selection": "single_best_qualifying_occurrence_v3" if args.ambiguity_policy == "best" else args.ambiguity_policy,
                     "factuality_parser_file_sha256": hashlib.sha256(Path(__file__).with_name("automatic_analysis.py").read_bytes()).hexdigest(),
                     "window_factor": 3, "stride_factor": 0.5, "minimum_window_tokens": 12,
                     "coverage_definition": "LCS non-space casefolded word characters / query characters",
                     "span_coordinates": "Python Unicode character offsets, end exclusive",
                     "source_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.results_dir.glob('*.csv')}})
    automatic_paths = sorted({p["metric_file"] for s in sets for p in s["provenance"]})
    settings["automatic_files"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in automatic_paths}
    (args.output_dir / "run_config.json").write_text(json_text(settings), encoding="utf-8")
    sources = {doc["document_id"]: SourceText(doc["text"]) for doc in documents.values()}
    cache, localization_rows, comparisons = {}, [], []
    started = time.monotonic()
    for index, extraction in enumerate(sets):
        source = sources[extraction["document_id"]]
        meta = {k: extraction[k] for k in ("set_id", *GROUP, "item_key", "summary_id", "document_id")}
        auto_spans, counts, accepted_items = [], defaultdict(int), 0
        for item_index, text in enumerate(extraction["texts"]):
            key = (extraction["document_id"], text)
            if key not in cache:
                cache[key] = localize(text, source, min_similarity=args.min_similarity,
                                      min_coverage=args.min_coverage, ambiguity_margin=args.ambiguity_margin,
                                      ambiguity_policy=args.ambiguity_policy)
            result = cache[key]
            selected = result["selected_spans"]
            accepted = bool(selected)
            candidate = result["candidate"] or {}
            counts[result["status"]] += 1
            if accepted:
                accepted_items += 1
                auto_spans.extend((span["start"], span["end"]) for span in selected)
            localization_rows.append(meta | {
                "item_index": item_index, "extracted_text": text, "status": result["status"],
                "accepted": accepted, **candidate, "alternatives_json": json_text(result["alternatives"]),
                "selected_span_count": len(selected), "selected_spans_json": json_text(selected),
            })
        auto_positions = source.token_positions(auto_spans)
        for annotator, ref in references[extraction["document_id"]].items():
            status = "invalid_human_spans" if ref["invalid"] else "scored"
            scores = positional_scores(auto_positions, source.token_positions(ref["spans"])) if not ref["invalid"] else {}
            comparisons.append(meta | {
                "annotator": annotator, "comparison_status": status, "automatic_items": len(extraction["texts"]),
                "accepted_items": accepted_items, "accepted_spans": len(auto_spans),
                "multiple_location_items": counts["multiple_locations"], "ambiguous_items": counts["ambiguous"],
                "below_threshold_items": counts["below_threshold"], "invalid_human_spans": ref["invalid"], **scores,
            })
        if (index + 1) % 20 == 0 or index + 1 == len(sets):
            print(f"Localized {index + 1}/{len(sets)} extraction sets; {len(cache)} unique items; {time.monotonic() - started:.1f}s", flush=True)
    loc = save_csv(localization_rows, args.output_dir / "automatic_localizations.csv",
                   ["set_id", *GROUP, "status", "accepted"])
    comparisons = save_csv(comparisons, args.output_dir / "position_agreement_by_unit.csv")
    summary = save_csv(summarize(comparisons, args.bootstrap), args.output_dir / "position_agreement_summary.csv")
    loc_summary = save_csv(localization_summary(loc, sets), args.output_dir / "localization_summary.csv")
    write_report(args.output_dir / "report.html", summary, loc_summary, loc, documents, references)
    print(f"Wrote {len(loc)} localizations and {len(comparisons)} comparisons to {args.output_dir}", flush=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
