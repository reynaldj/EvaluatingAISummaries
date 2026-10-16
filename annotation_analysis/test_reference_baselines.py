"""Regression tests for reference identity, scoring, and matched comparison units."""
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from reference_baselines import (GermanWordTokenizer, resolve_reference, summary_uid,
    prepare_unieval_jobs, correlation_stats, baseline_long_form, score_rouge, load_proposed)


class BaselineTests(unittest.TestCase):
    def test_duplicate_id_requires_transcript_identity(self):
        refs = {'speech': [{'text': 'Erster Beitrag.', 'summary': 'Ref A'},
                           {'text': 'Zweiter Beitrag.', 'summary': 'Ref B'}]}
        self.assertEqual(resolve_reference('speech', ' Zweiter  Beitrag. ', refs)[:2], ('Ref B', 'matched'))
        self.assertEqual(resolve_reference('speech', 'Anderer Beitrag.', refs)[1], 'transcript_mismatch')
        refs['speech'].append({'text': 'Zweiter Beitrag.', 'summary': 'Conflicting ref'})
        self.assertEqual(resolve_reference('speech', 'Zweiter Beitrag.', refs)[1], 'ambiguous_reference')

    def test_summary_identity_ignores_slot_but_not_text(self):
        self.assertEqual(summary_uid('long', r'folder\file', 'Ein Satz.'),
                         summary_uid('long', 'folder/file', 'Ein  Satz.'))
        self.assertNotEqual(summary_uid('short', 'id', 'Gut.'), summary_uid('short', 'id', 'Schlecht.'))

    def test_german_rouge_keeps_umlauts_and_sharp_s(self):
        self.assertEqual(GermanWordTokenizer().tokenize('Grüße, größere Äpfel!'), ['grüße', 'größere', 'äpfel'])
        data = pd.DataFrame([{'uid': 'a', 'summary_text': 'Grüße größere Äpfel', 'reference': 'Grüße größere Äpfel'},
                             {'uid': 'b', 'summary_text': 'Grüße', 'reference': 'Grüße größere Äpfel'}])
        values = score_rouge(data)
        self.assertEqual(values['a']['rouge1_f1'], 1.)
        self.assertEqual(values['b']['rouge1_precision'], 1.)
        self.assertAlmostEqual(values['b']['rouge1_recall'], 1/3)

    def test_unieval_uses_correct_context_and_sentence_aggregation_units(self):
        record = {'summary_text': 'Satz A. Satz B.', 'transcript': 'SOURCE', 'reference': 'REFERENCE'}
        jobs = prepare_unieval_jobs(record, lambda _: ['Satz A.', 'Satz B.'])
        by_dim = defaultdict(list)
        for dim, prompt in jobs:
            by_dim[dim].append(prompt)
        self.assertEqual({k: len(v) for k,v in by_dim.items()},
                         {'coherence': 1, 'factuality': 2, 'fluency': 2, 'relevance': 1})
        self.assertIn('SOURCE', by_dim['factuality'][0])
        self.assertIn('REFERENCE', by_dim['relevance'][0])
        self.assertNotIn('SOURCE', by_dim['relevance'][0])
        self.assertNotIn('SOURCE', by_dim['fluency'][0])

    def test_paired_bootstrap_identical_methods_have_zero_delta(self):
        frame = pd.DataFrame({'baseline_score': [1, 2, 4, 3, 5, 6],
            'proposed_score': [1, 2, 4, 3, 5, 6], 'human_score': [1, 2, 3, 4, 5, 6],
            'cluster_id': ['a', 'a', 'b', 'b', 'c', 'c']})
        stats = correlation_stats(frame, 40, 7, paired=True)
        self.assertEqual(stats['n_transcripts'], 3)
        for coefficient in ['spearman_rho', 'kendall_tau_b']:
            for suffix in ['', '_ci_low', '_ci_high']:
                self.assertEqual(stats['delta_'+coefficient+suffix], 0.)
        frame['baseline_score'] = 1
        self.assertTrue(np.isnan(correlation_stats(frame, 0, 7)['baseline_spearman_rho']))

    def test_dimension_specific_unieval_and_truncation(self):
        data = pd.DataFrame([{'uid': 'x', 'cluster_id': 'doc', 'coherence': 3, 'fluency': 4,
                              'factuality': 2, 'relevance': 1}])
        scores = pd.DataFrame([{'uid': 'x', 'rouge1_f1': .5, 'unieval_factuality': .2,
                               'unieval_factuality_truncated_inputs': 1}])
        frame = baseline_long_form(data, scores)
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.loc[frame.baseline.eq('unieval_factuality'), 'dimension'].tolist(), ['factuality'])
        self.assertEqual(frame.input_truncated.sum(), 1)

    def test_proposed_duplicate_and_conflicting_scores_are_not_averaged(self):
        record = {'join_status': 'matched', 'source': 'short', 'filename': 'doc',
                  'summary_text': 'Ein Satz.', 'dimension': 'relevance',
                  'evaluator_model': 'model', 'method_variant': 'narrow',
                  'keypoint_type': 'narrow', 'automatic_score': .5}
        frame = pd.DataFrame([record, record, record | {'method_variant': 'wide', 'automatic_score': .8},
                              record | {'method_variant': 'wide', 'automatic_score': .9}])
        data = pd.DataFrame({'uid': [summary_uid('short', 'doc', 'Ein Satz.')]})
        with patch('reference_baselines.pd.read_csv', return_value=frame):
            scores, audit = load_proposed(Path('.'), data)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores.iloc[0].proposed_score, .5)
        self.assertEqual(audit.status.tolist(), ['matched', 'conflicting_scores'])


if __name__ == '__main__':
    unittest.main()
