"""Add semantic localization ONLY to below-threshold items in a saved fuzzy run.

Default output is separate from the immutable fuzzy baseline. Cosine thresholds
are exploratory, not calibrated against human highlights or localization labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from fuzzy_span_alignment import SourceText, positional_scores
from fuzzy_position_agreement import (GROUP, json_text, localization_summary,
                                      save_csv, summarize, write_report)
from semantic_span_alignment import DEFAULT_MODEL, SemanticLocator, model_settings


def retain_best_locations(localizations):
    """Restrict accepted items to their best qualifying span, never using humans.

    Unaccepted candidates remain unaccepted. Historical alternatives remain for
    inspection, but selected_spans_json alone determines positional scoring.
    """
    rows = localizations.copy(deep=True)
    for index, row in rows.iterrows():
        spans = json.loads(row["selected_spans_json"])
        if not spans:
            continue
        if "semantic_similarity" in spans[0]:
            best = min(spans, key=lambda s: (-s["semantic_similarity"], s["end"]-s["start"], s["start"]))
        else:
            best = min(spans, key=lambda s: (-s["similarity"], -s["query_character_coverage"], s["start"]))
        rows.at[index, "selected_spans_json"] = json_text([best])
        rows.at[index, "selected_span_count"] = 1
        rows.at[index, "status"] = "located"
        if row.get("localization_method") == "semantic":
            rows.at[index, "semantic_status"] = "located"
        for key, value in best.items():
            rows.at[index, key] = value
    return rows


def apply_fallback(localizations, documents, locator, *, cache_dir=None, cache_identity=None):
    """Keep every accepted fuzzy span unchanged; annotate provenance separately."""
    rows = localizations.copy(deep=True)
    rows["fuzzy_status"] = rows["status"]
    rows["localization_method"] = np.where(rows["accepted"], "fuzzy", "unlocated")
    rows["semantic_attempted"] = False
    rows["semantic_status"] = "not_attempted"
    rows["semantic_similarity"] = np.nan
    rows["fuzzy_candidate_json"] = rows.apply(lambda r: json_text({
        k: None if pd.isna(r.get(k)) else r.get(k)
        for k in ("start", "end", "quote", "similarity", "query_character_coverage")}), axis=1)
    rows["fuzzy_alternatives_json"] = rows["alternatives_json"]
    eligible = rows[~rows["accepted"] & rows["status"].eq("below_threshold")]
    started = time.monotonic()
    groups = list(eligible.groupby("document_id", sort=True))
    for number, (document_id, group) in enumerate(groups, 1):
        source = SourceText(documents[document_id]["text"])
        queries = group["extracted_text"].tolist()
        cache_key = hashlib.sha256(json_text([cache_identity, source.text, queries]).encode("utf-8")).hexdigest()
        cache_file = cache_dir / (cache_key + ".json") if cache_dir else None
        if cache_file and cache_file.is_file():
            results = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            results = locator.localize_many(queries, source)
            if cache_file:
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_file.with_suffix(".tmp")
                temporary.write_text(json_text(results), encoding="utf-8")
                temporary.replace(cache_file)
        for index, row in group.iterrows():
            result = results[row["extracted_text"]]
            selected = result["selected_spans"]
            candidate = result["candidate"] or {}
            rows.at[index, "semantic_attempted"] = True
            rows.at[index, "semantic_status"] = result["status"]
            rows.at[index, "accepted"] = bool(selected)
            rows.at[index, "status"] = result["status"] if selected else "below_threshold"
            rows.at[index, "localization_method"] = "semantic" if selected else "unlocated"
            for field in ("start", "end", "quote", "semantic_similarity"):
                rows.at[index, field] = candidate.get(field, "" if field == "quote" else np.nan)
            # Never label a cosine score as a RapidFuzz score or lexical coverage.
            rows.at[index, "similarity"] = np.nan
            rows.at[index, "query_character_coverage"] = np.nan
            rows.at[index, "selected_span_count"] = len(selected)
            rows.at[index, "selected_spans_json"] = json_text(selected)
            rows.at[index, "alternatives_json"] = json_text(result["alternatives"])
        if number % 10 == 0 or number == len(groups):
            print(f"Semantic fallback: {number}/{len(groups)} documents; "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    return rows


def references_from_baseline(comparisons, audit):
    references = defaultdict(dict)
    for row in comparisons.to_dict("records"):
        references[row["document_id"]][row["annotator"]] = {
            "invalid": int(row["invalid_human_spans"]), "spans": []}
    for row in audit.to_dict("records"):
        if row["status"] not in ("valid", "converted_utf16"):
            continue
        ref = references.get(row["document_id"], {}).get(row["annotator"])
        if ref is not None:
            ref["spans"].append((int(row["start"]), int(row["end"])))
    for annotators in references.values():
        for ref in annotators.values():
            ref["spans"] = sorted(set(ref["spans"]))
    return references


def rescore(localizations, baseline_comparisons, documents, references):
    """Use exactly the same units and human-span exclusions as the baseline."""
    by_set = {}
    for set_id, group in localizations.groupby("set_id"):
        spans = [span for value in group["selected_spans_json"] for span in json.loads(value)]
        by_set[set_id] = {
            "spans": [(s["start"], s["end"]) for s in spans],
            "accepted_items": int(group["accepted"].sum()),
            "accepted_spans": len(spans),
            "multiple_location_items": int(group["status"].eq("multiple_locations").sum()),
            "ambiguous_items": int(group["status"].eq("ambiguous").sum()),
            "below_threshold_items": int(group["status"].eq("below_threshold").sum()),
            "semantic_accepted_items": int(group.get("localization_method", pd.Series(dtype=str)).eq("semantic").sum()),
        }
    sources = {key: SourceText(doc["text"]) for key, doc in documents.items()}
    rows = []
    for row in baseline_comparisons.to_dict("records"):
        info = by_set.get(row["set_id"], {"spans": [], "accepted_items": 0,
             "accepted_spans": 0, "multiple_location_items": 0, "ambiguous_items": 0,
             "below_threshold_items": 0, "semantic_accepted_items": 0})
        row.update({k: v for k, v in info.items() if k != "spans"})
        if row["comparison_status"] == "scored":
            source = sources[row["document_id"]]
            human = references[row["document_id"]][row["annotator"]]["spans"]
            row.update(positional_scores(source.token_positions(info["spans"]),
                                         source.token_positions(human)))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_summaries(baseline, hybrid):
    comparison = baseline.merge(hybrid, on=GROUP, suffixes=("_fuzzy", "_hybrid"), validate="one_to_one")
    for metric in ("micro_precision", "micro_recall", "micro_f1", "macro_f1"):
        comparison[metric + "_delta"] = comparison[metric + "_hybrid"] - comparison[metric + "_fuzzy"]
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("annotation_analysis/analysis_output/fuzzy_position_agreement"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotation_analysis/analysis_output/fuzzy_semantic_position_agreement"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model or local directory")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--min-semantic-similarity", type=float, default=None, help="Exploratory cosine threshold: E5 0.85, other models 0.75; not calibrated")
    parser.add_argument("--query-prefix", default=None, help="Defaults to 'query: ' for multilingual E5, empty otherwise")
    parser.add_argument("--passage-prefix", default=None, help="Defaults to 'passage: ' for multilingual E5, empty otherwise")
    parser.add_argument("--semantic-margin", type=float, default=0.03, help="With --location-policy all, retain distinct accepted locations this close to the best cosine score")
    parser.add_argument("--location-policy", choices=("best", "all"), default="best",
                        help="One best qualifying location per item; all reproduces multiple-location runs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="Inference precision; bfloat16 can accelerate supported CPUs/GPUs")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-model-download", action="store_true", help="Otherwise only cached/local model files are loaded")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    defaults = model_settings(args.model)
    if args.min_semantic_similarity is None:
        args.min_semantic_similarity = defaults["min_similarity"]
    if args.query_prefix is None:
        args.query_prefix = defaults["query_prefix"]
    if args.passage_prefix is None:
        args.passage_prefix = defaults["passage_prefix"]
    base, out = args.baseline_dir.resolve(), args.output_dir.resolve()
    if base == out or base in out.parents or out in base.parents:
        parser.error("Use a separate output folder outside the fuzzy baseline")
    if (out / "run_config.json").exists() or (out / "position_agreement_summary.csv").exists():
        parser.error("Output already contains a run; choose a new --output-dir")
    if not np.isfinite(args.min_semantic_similarity) or not -1 <= args.min_semantic_similarity <= 1:
        parser.error("Cosine threshold must be between -1 and 1")
    if not np.isfinite(args.semantic_margin) or not 0 <= args.semantic_margin <= 2:
        parser.error("Semantic margin must be between 0 and 2")
    if args.bootstrap < 0 or args.batch_size < 1 or args.threads < 1:
        parser.error("bootstrap must be nonnegative; batch-size and threads must be positive")
    files = ["automatic_localizations.csv", "position_agreement_by_unit.csv", "position_agreement_summary.csv",
             "extraction_sets.csv", "source_documents.jsonl", "human_span_validation.csv", "run_config.json"]
    missing = [name for name in files if not (base / name).is_file()]
    if missing:
        parser.error("Missing baseline files: " + ", ".join(missing))

    loc = pd.read_csv(base / files[0], keep_default_na=False)
    loc["accepted"] = loc["accepted"].astype(str).str.lower().eq("true")
    for column in ("start", "end", "similarity", "query_character_coverage"):
        loc[column] = pd.to_numeric(loc[column], errors="coerce").astype(float)
    units = pd.read_csv(base / files[1], dtype={"annotator": str})
    baseline_summary = pd.read_csv(base / files[2])
    sets = pd.read_csv(base / "extraction_sets.csv").to_dict("records")
    documents = {d["document_id"]: d for d in map(json.loads, (base / "source_documents.jsonl").read_text(encoding="utf-8").splitlines())}
    audit = pd.read_csv(base / "human_span_validation.csv", dtype={"annotator": str})
    references = references_from_baseline(units, audit)
    # Verify reconstruction before running the new method, avoiding mismatched baselines.
    reproduced = rescore(loc, units, documents, references)
    for column in ("automatic_tokens", "human_tokens", "overlap_tokens", "precision", "recall", "f1"):
        if not np.allclose(reproduced[column], units[column], equal_nan=True):
            raise ValueError(f"Cannot reproduce baseline {column}; check the baseline artifacts")
    if args.location_policy == "best":
        loc = retain_best_locations(loc)
        baseline_summary = pd.DataFrame(summarize(rescore(loc, units, documents, references), args.bootstrap))

    print(f"Loading {args.model} on {args.device}; cosine threshold {args.min_semantic_similarity} (exploratory)...", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(args.model, revision=args.model_revision, device=args.device,
                               local_files_only=not args.allow_model_download,
                               **({"cache_folder": str(out / "model_cache")} if args.allow_model_download else {}))
    if args.dtype == "bfloat16":
        if args.device == "cpu" and not getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)():
            raise ValueError("CPU bfloat16 acceleration is unavailable; use --dtype float32")
        model.to(dtype=torch.bfloat16)
    locator = SemanticLocator(model, min_similarity=args.min_semantic_similarity,
                              margin=args.semantic_margin, batch_size=args.batch_size,
                              query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
                              location_policy=args.location_policy)
    model_commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    cache_identity = {"model": args.model, "revision": args.model_revision, "commit": model_commit,
                      "threshold": args.min_semantic_similarity, "margin": args.semantic_margin,
                      "query_prefix": args.query_prefix, "passage_prefix": args.passage_prefix,
                      "max_tokens": locator.max_tokens, "device": args.device,
                      "dtype": args.dtype, "threads": args.threads,
                      "location_policy": args.location_policy,
                      "batch_size": args.batch_size,
                      "locator_sha256": hashlib.sha256(Path(__file__).with_name("semantic_span_alignment.py").read_bytes()).hexdigest(),
                      "versions": {name: version(name) for name in ("sentence-transformers", "transformers", "torch", "numpy")}}
    hybrid = apply_fallback(loc, documents, locator, cache_dir=out / "localization_cache", cache_identity=cache_identity)
    # The accepted fuzzy items and their exact selected locations must be untouched.
    accepted = loc["accepted"]
    if not hybrid.loc[accepted, "selected_spans_json"].equals(loc.loc[accepted, "selected_spans_json"]):
        raise AssertionError("Semantic fallback changed an accepted fuzzy location")
    comparisons = rescore(hybrid, units, documents, references)
    out.mkdir(parents=True, exist_ok=True)
    summary = save_csv(summarize(comparisons, args.bootstrap), out / "position_agreement_summary.csv")
    save_csv(hybrid, out / "automatic_localizations.csv")
    save_csv(comparisons, out / "position_agreement_by_unit.csv")
    loc_summary = pd.DataFrame(localization_summary(hybrid, sets))
    for name, mask in (("fuzzy_accepted_items", hybrid["localization_method"].eq("fuzzy")),
                       ("semantic_accepted_items", hybrid["localization_method"].eq("semantic")),
                       ("semantic_attempted_items", hybrid["semantic_attempted"])):
        counts = hybrid.assign(**{name: mask.astype(int)}).groupby(GROUP)[name].sum().reset_index()
        loc_summary = loc_summary.merge(counts, on=GROUP, validate="one_to_one")
    save_csv(loc_summary, out / "localization_summary.csv")
    save_csv(compare_summaries(baseline_summary, summary), out / "position_agreement_comparison.csv")
    review = hybrid[hybrid["semantic_attempted"]].groupby(GROUP + ["accepted"], group_keys=False).sample(frac=1, random_state=42)
    review = review.groupby(GROUP + ["accepted"], group_keys=False).head(5).copy()
    review["source_text"] = review["document_id"].map(lambda d: documents[d]["text"])
    for field in ("review_location_correct", "review_boundaries", "review_comment"):
        review[field] = ""
    # No human highlight text or offsets are included in this localization review.
    save_csv(review, out / "semantic_localization_review.csv")
    for name in ("source_documents.jsonl", "human_span_validation.csv", "extraction_sets.csv"):
        shutil.copyfile(base / name, out / name)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update({"algorithm": "fuzzy_then_semantic_passage_cosine_v3", "threshold_validation": "exploratory_not_calibrated",
                     "baseline_location_policy_applied": args.location_policy,
                     "fallback_eligibility": "unaccepted_below_threshold_only", "human_highlights_used_for_localization": False,
                     "candidate_boundaries": "heuristic_sentences_clauses_adjacent_pairs_word_aligned",
                     "overlength_policy": "overlapping_source_windows; reject_overlength_queries_without_truncation",
                     "max_model_tokens": locator.max_tokens,
                     "model_commit": model_commit, "cache_identity": cache_identity,
                     "versions": {name: version(name) for name in ("sentence-transformers", "transformers", "torch", "numpy")},
                     "baseline_files": {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in files},
                     "baseline_config": json.loads((base / "run_config.json").read_text(encoding="utf-8"))})
    (out / "run_config.json").write_text(json_text(settings), encoding="utf-8")
    write_report(out / "report.html", summary, loc_summary, hybrid, documents, references,
                 title=f"Fuzzy + semantic fallback positional agreement ({args.model})")
    print(f"Added {hybrid['localization_method'].eq('semantic').sum()} semantic localizations; results: {out}", flush=True)


if __name__ == "__main__":
    main()
