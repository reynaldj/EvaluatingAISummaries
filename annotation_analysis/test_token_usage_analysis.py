import unittest

import pandas as pd

from token_usage_analysis import CumulativeCounter, aggregate, parse_usage, token_counts


class TokenUsageTests(unittest.TestCase):
    def test_cumulative_counters_continue_across_files_and_restart(self):
        counter = CumulativeCounter()
        self.assertEqual(counter.advance((100, 20, 120)), ((100, 20, 120), 'counter_start'))
        self.assertEqual(counter.advance((200, 40, 240)), ((100, 20, 120), 'counter_increment'))
        # The first record of the next file is still cumulative.
        self.assertEqual(counter.advance((250, 50, 300)), ((50, 10, 60), 'counter_increment'))
        self.assertEqual(counter.advance((40, 5, 45)), ((40, 5, 45), 'counter_restart'))

    def test_zero_placeholder_does_not_reset_counter(self):
        counter = CumulativeCounter()
        counter.advance((100, 20, 120))
        self.assertIsNone(counter.advance((0, 0, 0))[0])
        self.assertIsNone(counter.advance(None)[0])
        self.assertEqual(counter.advance((150, 30, 180))[0], (50, 10, 60))
        self.assertIsNone(counter.advance((150, 30, 180))[0])

    def test_usage_parsing_and_component_validation(self):
        usage = parse_usage("{'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}")
        self.assertEqual(token_counts(usage), (10, 2, 12))
        self.assertIsNone(token_counts(parse_usage('{"prompt_tokens": null}')))
        with self.assertRaises(ValueError):
            token_counts({'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 99})

    def test_missing_and_duplicate_records_do_not_become_zero_cost(self):
        rows = pd.DataFrame([
            {'model': 'a', 'included': True, 'usage_status': 'recorded', 'counter_event': '',
             'raw_total_tokens': 12, 'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
            {'model': 'a', 'included': False, 'usage_status': 'exact_duplicate', 'counter_event': '',
             'raw_total_tokens': 12, 'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None},
            {'model': 'a', 'included': True, 'usage_status': 'missing', 'counter_event': '',
             'raw_total_tokens': None, 'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None},
        ])
        row = aggregate(rows, ['model']).iloc[0]
        self.assertEqual(row.total_tokens, 12)
        self.assertEqual(row.records_counted, 2)
        self.assertEqual(row.records_without_usable_counts, 1)
        self.assertEqual(row.exact_duplicate_records_excluded, 1)
        self.assertEqual(row.mean_total_tokens_usable_records, 12)
        self.assertEqual(row.accounting_status, 'recorded_partial')


if __name__ == '__main__':
    unittest.main()
