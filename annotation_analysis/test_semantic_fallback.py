"""Deterministic regression tests; no model download or human-guided localization."""
import json
import unittest

import numpy as np
import pandas as pd

from fuzzy_span_alignment import SourceText
from semantic_span_alignment import SemanticLocator, passage_spans, select_candidates, model_settings
from semantic_fallback_agreement import apply_fallback, rescore, retain_best_locations


class FakeEncoder:
    max_seq_length = 12

    def __init__(self):
        self.encoded = []

    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        # A synonym pair has the same vector despite different lexical wording.
        return np.array([[1., 0.] if any(w in t.casefold() for w in ("vertagt", "verschoben"))
                         else [0., 1.] for t in texts])


class SemanticTests(unittest.TestCase):
    def test_e5_prefixes_count_toward_budget_but_not_source_offsets(self):
        model = FakeEncoder()
        settings = model_settings("intfloat/multilingual-e5-large")
        self.assertEqual(settings["min_similarity"], .85)
        self.assertEqual(model_settings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")["query_prefix"], "")
        locator = SemanticLocator(model, **settings)
        source = SourceText("Der Rat vertagt die Abstimmung.")
        query = "Die Entscheidung wurde verschoben"
        result = locator.localize_many([query], source)[query]
        self.assertIn("query: " + query, model.encoded)
        self.assertIn("passage: Der Rat vertagt die Abstimmung", model.encoded)
        span = result["selected_spans"][0]
        self.assertEqual(source.text[span["start"]:span["end"]], span["quote"])
        self.assertNotIn("passage:", span["quote"])
        too_long_with_prefix = " ".join(["Wort"] * 10)
        self.assertEqual(locator.length(too_long_with_prefix), model.max_seq_length)
        self.assertEqual(locator.localize_many([too_long_with_prefix], source)[too_long_with_prefix]["status"], "semantic_query_too_long")
        long_source = SourceText(" ".join(["Wort"] * 50))
        fitted = locator.fitted_spans(long_source)
        self.assertTrue(all(locator.length("passage: " + long_source.text[a:b]) <= model.max_seq_length for a,b in fitted))
        self.assertEqual(long_source.token_positions(fitted), set(range(50)))

    def test_source_offsets_unicode_and_repeated_occurrences(self):
        source = SourceText("😀 Der Rat vertagt die Abstimmung. Andere Themen. Der Rat vertagt die Abstimmung.")
        model = FakeEncoder()
        result = SemanticLocator(model, location_policy="all").localize_many(["Die Entscheidung wurde verschoben"], source)
        selected = result["Die Entscheidung wurde verschoben"]["selected_spans"]
        self.assertEqual(len(selected), 2)
        for span in selected:
            self.assertEqual(source.text[span["start"]:span["end"]], span["quote"])
            self.assertEqual(span["quote"], "Der Rat vertagt die Abstimmung")
        positions = source.token_positions([(s["start"], s["end"]) for s in selected])
        self.assertNotIn(5, positions)  # The intervening 'Andere' is not selected.

    def test_reject_weak_candidates_and_collapse_boundaries(self):
        candidates = [{"start": 0, "end": 10, "semantic_similarity": .9},
                      {"start": 0, "end": 12, "semantic_similarity": .89},
                      {"start": 20, "end": 30, "semantic_similarity": .88},
                      {"start": 40, "end": 50, "semantic_similarity": .70}]
        selected = select_candidates(candidates, .75, .03, "all")["selected_spans"]
        self.assertEqual([(c["start"], c["end"]) for c in selected], [(0, 10), (20, 30)])
        self.assertEqual(len(select_candidates(candidates, .75, .03)["selected_spans"]), 1)
        self.assertEqual(select_candidates(candidates, .75, .03)["status"], "located")
        self.assertEqual(select_candidates(candidates, .95, .03)["selected_spans"], [])

    def test_no_silent_truncation_and_long_query_abstention(self):
        model = FakeEncoder()
        source = SourceText(" ".join(["Wort"] * 50))
        locator = SemanticLocator(model)
        locator.localize_many(["Kurze Aussage", " ".join(["lang"] * 30)], source)
        self.assertTrue(model.encoded)
        self.assertTrue(all(locator.length(t) <= model.max_seq_length for t in model.encoded))
        result = locator.localize_many([" ".join(["lang"] * 30)], source)
        self.assertEqual(next(iter(result.values()))["status"], "semantic_query_too_long")
        covered = source.token_positions(locator.fitted_spans(source))
        self.assertEqual(covered, set(range(50)))

    def test_empty_source_and_invalid_parameters(self):
        locator = SemanticLocator(FakeEncoder())
        self.assertEqual(locator.localize_many(["query"], SourceText(""))["query"]["selected_spans"], [])
        self.assertEqual(passage_spans(SourceText("...")), [])
        for kwargs in ({"min_similarity": float("nan")}, {"min_similarity": 1.1},
                       {"margin": -1}, {"batch_size": 0}):
            with self.assertRaises(ValueError):
                SemanticLocator(FakeEncoder(), **kwargs)

    def test_hybrid_preserves_fuzzy_and_rescores_union(self):
        source = "Der Rat vertagt die Abstimmung."
        span = {"start": 0, "end": 3, "quote": "Der", "similarity": 90., "query_character_coverage": .9}
        common = {"document_id": "doc", "set_id": "set", "alternatives_json": "[]", **span}
        rows = pd.DataFrame([
            common | {"status": "located", "accepted": True, "extracted_text": "Already found",
                      "selected_span_count": 1, "selected_spans_json": json.dumps([span])},
            common | {"status": "below_threshold", "accepted": False, "extracted_text": "Entscheidung verschoben",
                      "selected_span_count": 0, "selected_spans_json": "[]"}])
        documents = {"doc": {"text": source}}
        model = FakeEncoder()
        result = apply_fallback(rows, documents, SemanticLocator(model))
        self.assertEqual(result.loc[0, "selected_spans_json"], rows.loc[0, "selected_spans_json"])
        self.assertFalse(result.loc[0, "semantic_attempted"])
        self.assertNotIn("Already found", model.encoded)
        self.assertEqual(result.loc[1, "localization_method"], "semantic")
        self.assertTrue(pd.isna(result.loc[1, "similarity"]))
        self.assertFalse(rows.loc[1, "accepted"])  # Input is immutable.
        units = pd.DataFrame([{"set_id": "set", "document_id": "doc", "annotator": "1", "comparison_status": "scored"},
                              {"set_id": "set", "document_id": "doc", "annotator": "2", "comparison_status": "invalid_human_spans", "f1": np.nan}])
        refs = {"doc": {"1": {"spans": [(0, 7)]}, "2": {"spans": []}}}
        scores = rescore(result, units, documents, refs)
        self.assertEqual(scores.loc[0, "automatic_tokens"], 5)
        self.assertEqual(scores.loc[0, "overlap_tokens"], 2)
        self.assertEqual(scores.loc[0, "recall"], 1)
        self.assertTrue(pd.isna(scores.loc[1, "f1"]))

    def test_saved_selection_keeps_best_and_never_accepts_a_rejection(self):
        a = {"start": 0, "end": 8, "quote": "original", "semantic_similarity": .91}
        b = {"start": 20, "end": 29, "quote": "repeated", "semantic_similarity": .92}
        rows = pd.DataFrame([
            {"accepted": True, "status": "multiple_locations", "localization_method": "semantic",
             "semantic_status": "multiple_locations", "selected_spans_json": json.dumps([a,b]), "selected_span_count": 2},
            {"accepted": False, "status": "below_threshold", "localization_method": "unlocated",
             "semantic_status": "semantic_below_threshold", "selected_spans_json": "[]", "selected_span_count": 0}])
        selected = retain_best_locations(rows)
        self.assertEqual(json.loads(selected.iloc[0].selected_spans_json), [b])
        self.assertEqual(selected.iloc[0].semantic_status, "located")
        self.assertEqual(selected.iloc[1].selected_span_count, 0)
        self.assertTrue(selected.accepted.equals(rows.accepted))
        self.assertTrue(retain_best_locations(selected).equals(selected))


if __name__ == "__main__":
    unittest.main()
