"""Regression checks for shuffled summary slots and annotation provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from fuzzy_position_agreement import human_join_table, human_references, source_documents


class SourceIdentityTests(unittest.TestCase):
    def exports(self):
        rows = []
        for annotator, summaries in (("1", ["First claim", "Second claim", "Third claim"]),
                                     ("2", ["Second claim", "First claim", "Third claim"])):
            row = {"source": "short", "filename": "example", "input_file": annotator + ".csv",
                   "annotator": annotator, "annotation_id": annotator, "transcript": "First second third",
                   "transcript_labels": "[]"}
            for slot, text in zip("abc", summaries):
                row[f"summary_{slot}"] = text
                row[f"{slot}_error_labels"] = json.dumps([{"start": 0, "end": len(text), "text": text, "labels": ["Error"]}])
            rows.append(row)
        return pd.DataFrame(rows)

    def test_shuffled_slots_share_document_and_correct_human_spans(self):
        exports = self.exports()
        documents = source_documents(exports)
        self.assertEqual(len(documents), 4)
        references, audit = human_references(exports, documents)
        self.assertTrue(all(row["status"] == "valid" for row in audit))
        first = next(doc for doc in documents.values() if doc["text"] == "First claim")
        self.assertEqual(set(references[first["document_id"]]), {"1", "2"})
        self.assertEqual(references[first["document_id"]]["2"]["spans"], [(0, 11)])
        self.assertEqual({alias["slot"] for alias in first["export_locations"]}, {"summary_a", "summary_b"})

    def test_assignment_metadata_follows_text_not_slot(self):
        # Keep fixtures in memory; temporary-directory ACLs can prevent this test
        # from running in the Windows sandbox even though joins need no writes.
        root = Path("source_identity_fixture")
        record = {"source": "short", "filename": "example"}
        for slot, text in zip("abc", ["First claim", "Second claim", "Third claim"]):
            record.update({f"summary_{slot}": text, f"model_{slot}": "model_" + slot,
                           f"profile_{slot}": "profile_" + slot, f"degraded_{slot}": ""})
        import_copy = {k: v for k, v in record.items() if not k.startswith(("model_", "profile_", "degraded_"))}
        fixtures = {root / "test_annotations.json": json.dumps([record]),
                    root / "copy_annotations.json": json.dumps([import_copy])}
        with patch.object(Path, "glob", autospec=True, side_effect=lambda p, pattern: list(fixtures) if p == root else []), \
                patch.object(Path, "read_text", autospec=True, side_effect=lambda p, **kwargs: fixtures[p]):
            human, audit = human_join_table(self.exports(), SimpleNamespace(assignment_dir=root, pilot_assignment_dir=root / "absent"))
            self.assertEqual(len(human), 3)
            self.assertTrue(all(row["metadata_status"] == "matched" for row in audit))
            first = human[human["summary_text"].eq("First claim")].iloc[0]
            self.assertEqual(first["summary_model"], "model_a")
            self.assertEqual(first["n_annotators"], 2)

    def test_invalid_annotation_is_retained_as_invalid(self):
        exports = self.exports()
        exports.at[0, "a_error_labels"] = json.dumps([{"start": 0, "end": 2, "text": "wrong", "labels": []}])
        documents = source_documents(exports)
        references, audit = human_references(exports, documents)
        invalid = [row for row in audit if row["status"] == "invalid_offsets_or_text"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(references[invalid[0]["document_id"]]["1"]["invalid"], 1)


if __name__ == "__main__":
    unittest.main()
