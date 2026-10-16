"""Synthetic checks for localization and coordinate-based scoring."""
import unittest
from unittest.mock import patch

from fuzzy_span_alignment import SourceText, localize, positional_scores, validate_human_span


class FuzzyAlignmentTests(unittest.TestCase):
    def test_edited_claim_localizes_to_original_coordinates(self):
        text = "An unrelated introduction. The team plans to release the new application in June. Other business follows."
        result = localize("The team plans to release the application in June", SourceText(text))
        self.assertEqual(result["status"], "located")
        candidate = result["candidate"]
        self.assertEqual(text[candidate["start"]:candidate["end"]], candidate["quote"])
        self.assertIn("new application", candidate["quote"])
        self.assertGreater(candidate["similarity"], 85)

    def test_repeated_occurrences_are_retained(self):
        claim = "The application will launch in June"
        text = claim + ". A different subject was discussed for quite some time. " + claim + "."
        result = localize(claim, SourceText(text), ambiguity_policy="all")
        self.assertEqual(result["status"], "multiple_locations")
        self.assertEqual(len(result["selected_spans"]), 2)
        positions = SourceText(text).token_positions([(s["start"], s["end"]) for s in result["selected_spans"]])
        self.assertEqual(len(positions), 12)  # Two six-word occurrences; intervening words excluded.
        self.assertEqual(result["candidate"]["similarity"], 100)
        self.assertTrue(any(c["start"] != result["candidate"]["start"] for c in result["alternatives"]))

    def test_legacy_policies_remain_reproducible(self):
        claim = "The application will launch in June"
        source = SourceText(claim + ". A different subject was discussed for quite some time. " + claim)
        self.assertEqual(localize(claim, source, ambiguity_policy="exclude")["selected_spans"], [])
        self.assertEqual(len(localize(claim, source, ambiguity_policy="best")["selected_spans"]), 1)
        self.assertEqual(localize(claim, source)["status"], "located")
        self.assertEqual(len(localize(claim, source)["selected_spans"]), 1)

    def test_only_near_best_qualifying_candidates_selected(self):
        candidates = [{"start": i * 100, "end": i * 100 + 20, "quote": "example",
                       "similarity": score, "query_character_coverage": coverage}
                      for i, (score, coverage) in enumerate([(80, .8), (79, .5), (77, .7), (76, .8), (64, 1)])]
        with patch("fuzzy_span_alignment.fuzzy_candidates", return_value=candidates):
            result = localize("example", SourceText("example"), ambiguity_policy="all")
        self.assertEqual([s["start"] for s in result["selected_spans"]], [0, 200])

    def test_multiple_locations_do_not_duplicate_overlap(self):
        source = SourceText("one two three four five")
        positions = source.token_positions([(0, 13), (8, 23), (0, 13)])
        scores = positional_scores(positions, {0, 1, 2})
        self.assertEqual(scores["automatic_tokens"], 5)
        self.assertEqual(scores["overlap_tokens"], 3)

    def test_tiny_matching_fragment_is_rejected(self):
        result = localize("The team plans to release the application in June after extensive testing", SourceText("The team"))
        self.assertEqual(result["status"], "below_threshold")
        self.assertLess(result["candidate"]["query_character_coverage"], 0.65)

    def test_unicode_offsets_survive_casefold_and_punctuation(self):
        text = "😀 Hinweis: Die STRAẞE bleibt bis nächsten Montag gesperrt. Ende."
        result = localize("Die Straße bleibt bis nächsten Montag gesperrt", SourceText(text))
        candidate = result["candidate"]
        self.assertEqual(candidate["start"], text.index("Die"))
        self.assertEqual(candidate["quote"], "Die STRAẞE bleibt bis nächsten Montag gesperrt")

    def test_empty_query(self):
        self.assertEqual(localize("...", SourceText("Some text"))["status"], "empty_query_or_source")

    def test_positions_not_word_identity(self):
        source = SourceText("budget approved and budget approved")
        left = source.token_positions([(0, 15)])
        right = source.token_positions([(20, len(source.text))])
        self.assertEqual(positional_scores(left, right)["overlap_tokens"], 0)

    def test_union_and_end_exclusive(self):
        source = SourceText("one two three four")
        self.assertEqual(source.token_positions([(0, 7), (4, 13)]), {0, 1, 2})
        self.assertEqual(source.token_positions([(3, 4)]), set())
        self.assertEqual(source.token_positions([(4, 7)]), {1})

    def test_empty_score_conventions(self):
        self.assertIsNone(positional_scores(set(), set())["f1"])
        self.assertEqual(positional_scores(set(), {1})["f1"], 0)
        self.assertEqual(positional_scores({1}, set())["f1"], 0)

    def test_utf16_and_invalid_offsets(self):
        self.assertEqual(validate_human_span("😀 hello", 3, 8, "hello"), (2, 7, "converted_utf16"))
        self.assertEqual(validate_human_span("hello", 0, 4, "hello")[2], "invalid_offsets_or_text")


if __name__ == "__main__":
    unittest.main()
