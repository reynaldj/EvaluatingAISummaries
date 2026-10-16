"""Rescore a saved hybrid run with one best location, without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from fuzzy_position_agreement import GROUP, json_text, save_csv, summarize, localization_summary, write_report
from semantic_fallback_agreement import retain_best_locations, references_from_baseline, rescore, compare_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("annotation_analysis/analysis_output/fuzzy_position_agreement"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, base, out = args.source_dir.resolve(), args.baseline_dir.resolve(), args.output_dir.resolve()
    if out.exists() or source in out.parents or base in out.parents:
        parser.error("Choose a new output folder outside both source runs")
    old = pd.read_csv(source / "automatic_localizations.csv")
    units = pd.read_csv(source / "position_agreement_by_unit.csv", dtype={"annotator": str})
    audit = pd.read_csv(source / "human_span_validation.csv", dtype={"annotator": str})
    documents = {d["document_id"]: d for d in map(json.loads, (source / "source_documents.jsonl").read_text(encoding="utf-8").splitlines())}
    refs = references_from_baseline(units, audit)
    sets = pd.read_csv(source / "extraction_sets.csv").to_dict("records")
    original_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    bootstrap = original_config.get("bootstrap", 1000)
    reproduced = rescore(old, units, documents, refs)
    for key in ("automatic_tokens", "human_tokens", "overlap_tokens", "precision", "recall", "f1"):
        if not np.allclose(reproduced[key], units[key], equal_nan=True):
            raise ValueError(f"Saved run does not reproduce: {key}")
    new = retain_best_locations(old)
    assert new["accepted"].equals(old["accepted"])
    assert new["selected_span_count"].le(1).all()
    for row in new.to_dict("records"):
        for span in json.loads(row["selected_spans_json"]):
            assert documents[row["document_id"]]["text"][span["start"]:span["end"]] == span["quote"]
    comparisons = rescore(new, units, documents, refs)
    valid = comparisons["comparison_status"].eq("scored")
    assert (comparisons.loc[valid, "automatic_tokens"] <= units.loc[valid, "automatic_tokens"]).all()
    assert (comparisons.loc[valid, "overlap_tokens"] <= units.loc[valid, "overlap_tokens"]).all()
    assert np.allclose(comparisons["human_tokens"], units["human_tokens"], equal_nan=True)
    fuzzy = retain_best_locations(pd.read_csv(base / "automatic_localizations.csv"))
    fuzzy_units = pd.read_csv(base / "position_agreement_by_unit.csv", dtype={"annotator": str})
    fuzzy_summary = pd.DataFrame(summarize(rescore(fuzzy, fuzzy_units, documents, refs), bootstrap))
    out.mkdir(parents=True)
    summary = save_csv(summarize(comparisons, bootstrap), out / "position_agreement_summary.csv")
    save_csv(new, out / "automatic_localizations.csv")
    save_csv(comparisons, out / "position_agreement_by_unit.csv")
    save_csv(fuzzy_summary, out / "fuzzy_best_position_agreement_summary.csv")
    save_csv(compare_summaries(fuzzy_summary, summary), out / "position_agreement_comparison.csv")
    loc_summary = pd.DataFrame(localization_summary(new, sets))
    for name, mask in (("fuzzy_accepted_items", new["localization_method"].eq("fuzzy")),
                       ("semantic_accepted_items", new["localization_method"].eq("semantic")),
                       ("semantic_attempted_items", new["semantic_attempted"])):
        counts = new.assign(**{name: mask.astype(int)}).groupby(GROUP)[name].sum().reset_index()
        loc_summary = loc_summary.merge(counts, on=GROUP, validate="one_to_one")
    save_csv(loc_summary, out / "localization_summary.csv")
    review = new[new["semantic_attempted"]].groupby(GROUP + ["accepted"], group_keys=False).sample(frac=1, random_state=42)
    review = review.groupby(GROUP + ["accepted"], group_keys=False).head(5).copy()
    review["source_text"] = review["document_id"].map(lambda d: documents[d]["text"])
    for key in ("review_location_correct", "review_boundaries", "review_comment"):
        review[key] = ""
    save_csv(review, out / "semantic_localization_review.csv")
    for name in ("extraction_sets.csv", "source_documents.jsonl", "human_span_validation.csv"):
        shutil.copyfile(source / name, out / name)
    config = dict(original_config)
    config.update({"algorithm": "fuzzy_then_semantic_passage_cosine_v3", "location_policy": "best",
                   "baseline_location_policy_applied": "best", "semantic_margin": None,
                   "location_selection": "single_best_qualifying_occurrence",
                   "output_dir": str(out), "reselected_from": str(source),
                   "reselection_script": str(Path(__file__).resolve()),
                   "embedding_run_config": original_config,
                   "reselection_inputs": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                          for name in ("automatic_localizations.csv", "position_agreement_by_unit.csv", "run_config.json")}})
    config["embedding_cache_identity"] = config.pop("cache_identity", None)
    (out / "run_config.json").write_text(json_text(config), encoding="utf-8")
    verification = {"model": config["model"], "location_policy": "best", "accepted_items_unchanged": True,
                    "accepted_items": int(new["accepted"].sum()), "maximum_spans_per_item": int(new["selected_span_count"].max()),
                    "previous_selected_spans": int(old["selected_span_count"].sum()),
                    "new_selected_spans": int(new["selected_span_count"].sum()),
                    "comparison_rows": len(comparisons), "source_offsets_verified": True,
                    "embedding_inference_rerun": False}
    (out / "verification.json").write_text(json_text(verification), encoding="utf-8")
    write_report(out / "report.html", summary, loc_summary, new, documents, refs,
                 title=f"Single-location fuzzy + semantic agreement ({config['model']})")
    print(json_text(verification))


if __name__ == "__main__":
    main()
