"""Regression tests for claim-only factuality ingestion; no filesystem fixtures."""
import json
import unittest
from unittest.mock import Mock

from automatic_analysis import factuality_claim_text, load_factuality


class FactualityParserTests(unittest.TestCase):
    def test_all_verdict_labels_and_multiline_explanations(self):
        for verdict in ("FP", "CONTRADICTED", "NOT_SUPPORTED", "UNSUPPORTED"):
            with self.subTest(verdict=verdict):
                self.assertEqual(factuality_claim_text(
                    f"[7] Die Zustimmung ist nicht erfolgt.\nVERDICT: {verdict}\n(Explanation\ncontinued)"),
                    "Die Zustimmung ist nicht erfolgt.")

    def test_case_and_marker_spacing(self):
        self.assertEqual(factuality_claim_text("Claim. verdict : contradicted (reason)"), "Claim.")

    def test_preserves_plain_claims_numbers_and_negation(self):
        for claim in ("2026 war das Ziel.", "Es sind exakt 565 Kilogramm.", "Die Zustimmung ist nicht erfolgt."):
            self.assertEqual(factuality_claim_text(claim), claim)

    def test_empty_or_missing_claim(self):
        for raw in (None, "", "VERDICT: CONTRADICTED (reason)", "[8] VERDICT: FP"):
            self.assertEqual(factuality_claim_text(raw), "")

    def test_loader_keeps_raw_provenance_and_metrics(self):
        raw = "[11] Die Menge ist 565 Kilogramm. VERDICT: CONTRADICTED (565 Tonnen.)"
        path = Mock()
        path.name = "results_short_degraded_model.json"
        path.stem = "results_short_degraded_model"
        path.read_text.return_value = json.dumps([{
            "filename": "sample", "profile": "test", "metrics": {"precision": 0.5, "fp_count": 2},
            "verdict_reasoning": {"FP": [raw, "VERDICT: FP (no claim)"]},
        }])
        directory = Mock()
        directory.glob.return_value = [path]
        metrics, statements = load_factuality(directory)
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements.iloc[0].statement, "Die Menge ist 565 Kilogramm.")
        self.assertEqual(statements.iloc[0].raw_statement, raw)
        self.assertEqual(metrics.iloc[0].fp_count, 2)
        self.assertEqual(metrics.iloc[0].automatic_score, 0.5)


if __name__ == "__main__":
    unittest.main()
