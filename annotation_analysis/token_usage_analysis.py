"""Audit saved LLM token counters for the four thesis evaluation pipelines.

This reads logs only; it does not load models, retokenize text, or call APIs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


GROUP = ['dimension', 'stage', 'method_variant', 'evaluator_model']
MODELS = {
    'gemma-12b': ('gemma12b', 'coherence_geval_ablation_gemma12b', 'fluency_evaluation_results_gemma12b', 'results-factuality-evaluation-gemma-12b'),
    'gemma-31b': ('gemma31b', 'coherence_geval_ablation_gemma31b', 'fluency_evaluation_results_gemma_31b', 'results-factuality-evaluation-gemma-31b'),
    'llama-3.1-8b': ('llama-3.1-8b', 'coherence_geval_ablation_llama3.1-8b-instruct', 'fluency_evaluation_results_llama-3.1-8b', 'results-factuality-evaluation-llama8b'),
    'llama-70b': ('llama-70b', 'coherence_geval_ablation_llama70b', 'fluency_evaluation_results_llama70b', 'results-factuality-evaluation-llama70b-correct'),
}
EXTRACTIONS = {
    ('gemma-12b', 'narrow'): 'gemma12b/entailment_keypoints_results_summaries_strict_9_gemma12b_narrow/keypoints_gemma.json',
    ('gemma-12b', 'wide'): 'gemma12b/entailment_keypoints_results_summaries_strict_9_gemma12b_wide/keypoints_gemma.json',
    ('gemma-31b', 'narrow'): 'gemma31b/entailment_keypoints_results_summaries_strict_9_gemma31b_narrow/keypoints_gemma.json',
    ('gemma-31b', 'wide'): 'gemma31b/entailment_keypoints_results_summaries_strict_9_gemma31b_wide/keypoints_gemma.json',
    ('llama-3.1-8b', 'narrow'): 'llama-3.1-8b/entailment_keypoints_results_summaries_strict_9_llama3.1-8b-instruct_narrow/entailment_kp_dedup.json',
    ('llama-3.1-8b', 'wide'): 'llama-3.1-8b/entailment_keypoints_results_summaries_strict_9_llama3.1-8b-instruct_wide/entailment_kp_dedup.json',
    ('llama-70b', 'narrow'): 'llama-70b/entailment_keypoints_results_summaries_strict_9_llama70b_narrow/keypoints_llama.json',
    ('llama-70b', 'wide'): 'llama-70b/entailment_keypoint_results_summaries_strict_9_llama70b_wide/keypoints_llama.json',
    ('shared_model_unrecorded', 'statistical'): 'statistical_keypoints_results_summaries_strict_9/statistical_kp_dedup.json',
}


def parse_usage(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, dict) else {}
        except (ValueError, SyntaxError):
            return {}


def token_counts(usage):
    values = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        return None
    if values[0] + values[1] != values[2]:
        raise ValueError(f'Inconsistent token counters: {values}')
    return tuple(values)


class CumulativeCounter:
    """Reconstruct deltas in runner order, preserving state across result files.

    Zero placeholders do not reset a previously positive counter. A decrease in
    both components identifies a restart. Session IDs were not saved, so this
    reconstruction is explicitly an estimate of the recorded usage.
    """
    def __init__(self):
        self.previous = (0, 0, 0)

    def advance(self, current):
        if current is None or current[2] == 0:
            return None, 'missing_or_zero_counter'
        before = self.previous
        p_drop, c_drop = current[0] < before[0], current[1] < before[1]
        if p_drop != c_drop:
            raise ValueError(f'Conflicting cumulative counter direction: {before} -> {current}')
        if p_drop:
            before = (0, 0, 0)
            event = 'counter_restart'
        elif before[2] == 0:
            event = 'counter_start'
        else:
            event = 'counter_increment'
        delta = tuple(a-b for a, b in zip(current, before))
        self.previous = current
        if delta[2] == 0:
            return None, 'unchanged_counter'
        return delta, event


class Collector:
    def __init__(self, root):
        self.root = root
        self.rows = []
        self.files = []
        self.seen = {}

    def add_file(self, path, dimension, stage, variant, model, counter=None):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(self.root).as_posix()
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as handle:
                records = list(csv.DictReader(handle))
        else:
            records = json.loads(path.read_text(encoding='utf-8'))
        self.files.append({'source_file': relative, 'dimension': dimension, 'stage': stage,
                           'evaluator_model': model, 'records': len(records),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for index, record in enumerate(records):
            method = variant or ('auto_cot' if 'auto_cot' in record['prompt_name'] else 'no_auto_cot')
            group = (dimension, stage, method, model)
            fingerprint = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
            previous_id = self.seen.get((group, fingerprint))
            row_id = f'{relative}#{index}'
            included = previous_id is None
            if included:
                self.seen[group, fingerprint] = row_id
            usage = parse_usage(record.get('token_usage'))
            raw = token_counts(usage)
            numeric = None
            status = 'missing'
            event = ''
            if not included:
                status = 'exact_duplicate'
            elif counter is not None:
                numeric, event = counter.advance(raw)
                status = 'reconstructed' if numeric is not None else 'unavailable_counter'
            elif raw is not None and raw[2] > 0:
                numeric, status = raw, 'recorded'
            elif raw is not None and raw[2] == 0:
                # Empty relevance evaluations can legitimately invoke no LLM.
                if stage == 'evaluation' and dimension == 'relevance' and not record.get('key_items'):
                    numeric, status = raw, 'recorded_zero_no_items'
                else:
                    status = 'zero_unverified'
            profile = record.get('profile', '')
            if not isinstance(profile, str):
                profile = json.dumps(profile, ensure_ascii=False, sort_keys=True)
            self.rows.append(dict(zip(GROUP, group)) | {
                'record_id': row_id, 'source_file': relative, 'record_index': index,
                'filename': record.get('filename', ''), 'profile': profile,
                'summary_model': record.get('summary_model', ''),
                'transcript_hash': record.get('transcript_hash', ''),
                'included': included, 'duplicate_of': previous_id or '',
                'usage_status': status, 'counter_event': event,
                'raw_prompt_tokens': raw[0] if raw else None,
                'raw_completion_tokens': raw[1] if raw else None,
                'raw_total_tokens': raw[2] if raw else None,
                'prompt_tokens': numeric[0] if numeric else None,
                'completion_tokens': numeric[1] if numeric else None,
                'total_tokens': numeric[2] if numeric else None,
            })


def aggregate(frame, keys):
    rows = []
    for key, raw_group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        group = raw_group[raw_group.included]
        known = group[group.total_tokens.notna()]
        incomplete = len(known) < len(group)
        reconstructed = group.usage_status.eq('reconstructed').any()
        status = ('unavailable' if known.empty else
                  ('reconstructed_partial' if incomplete else 'reconstructed') if reconstructed else
                  ('recorded_partial' if incomplete else 'recorded'))
        row = dict(zip(keys, key)) | {
            'saved_records': len(raw_group), 'exact_duplicate_records_excluded': len(raw_group)-len(group),
            'records_counted': len(group), 'records_with_usable_counts': len(known),
            'records_without_usable_counts': len(group)-len(known),
            'zero_counter_records': int(group.raw_total_tokens.eq(0).sum()),
            'counter_restarts': int(group.counter_event.eq('counter_restart').sum()),
            'accounting_status': status,
        }
        for token in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            row[token] = int(known[token].sum()) if len(known) else None
        # Partial cumulative increments can span records whose counters are
        # missing. Do not present them as individual evaluation distributions.
        allow_distribution = len(known) > 0 and not (reconstructed and incomplete)
        row['mean_total_tokens_usable_records'] = known.total_tokens.mean() if allow_distribution else None
        row['median_total_tokens_usable_records'] = known.total_tokens.median() if allow_distribution else None
        row['p95_total_tokens_usable_records'] = known.total_tokens.quantile(.95) if allow_distribution else None
        rows.append(row)
    return pd.DataFrame(rows)


def save(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6f')


def grouped_evaluation_usage(methods):
    """Pool records with interpretable per-evaluation means, preserving coverage.

    Partial factuality counters can cover unlogged intervening evaluations; their
    method means are absent and must not be divided by the positive-counter count.
    """
    result = []
    for (dimension, variant), group in methods[methods.stage.eq('evaluation')].groupby(
            ['dimension', 'method_variant'], sort=True):
        eligible = group[group.mean_total_tokens_usable_records.notna()]
        omitted = group[group.mean_total_tokens_usable_records.isna()]
        n_used = int(eligible.records_with_usable_counts.sum())
        if not n_used:
            continue
        row = {'dimension': dimension, 'method_variant': variant,
               'models_included': '; '.join(eligible.evaluator_model),
               'models_omitted': '; '.join(omitted.evaluator_model),
               'n_models_included': len(eligible),
               'records_in_scope': int(group.records_counted.sum()),
               'records_used_for_average': n_used,
               'records_excluded_from_average': int(group.records_counted.sum()) - n_used,
               'includes_reconstructed_counts': eligible.accounting_status.str.contains('reconstructed').any()}
        for component in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            row['sum_' + component + '_in_average'] = int(eligible[component].sum())
            row['mean_' + component + '_per_evaluation'] = eligible[component].sum() / n_used
        assert abs(row['mean_prompt_tokens_per_evaluation'] + row['mean_completion_tokens_per_evaluation']
                   - row['mean_total_tokens_per_evaluation']) < 1e-6
        result.append(row)
    return pd.DataFrame(result)


def coverage_pipeline_usage(methods):
    """Combine recorded stage totals, counting shared extraction only once.

    These are inventory totals, not matched per-summary pipeline averages:
    extraction is per transcript and may be reused across evaluations.
    """
    result = []
    for variant, group in methods[methods.dimension.eq('relevance')].groupby('method_variant'):
        row = {'method_variant': variant}
        for stage in ('evaluation', 'keypoint_extraction'):
            subset = group[group.stage.eq(stage)]
            row[stage + '_records'] = int(subset.records_counted.sum())
            row[stage + '_records_with_usable_counts'] = int(subset.records_with_usable_counts.sum())
            row[stage + '_records_without_usable_counts'] = int(subset.records_without_usable_counts.sum())
            for component in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                row[stage + '_recorded_' + component] = subset[component].sum(min_count=1)
        for component in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            row['combined_recorded_' + component] = group[component].sum(min_count=1)
            extraction_count = row['keypoint_extraction_records_with_usable_counts']
            row['mean_keypoint_extraction_' + component + '_per_usable_record'] = (
                row['keypoint_extraction_recorded_' + component] / extraction_count
                if extraction_count else None)
        row['accounting_status'] = ('recorded_partial' if group.records_without_usable_counts.sum()
                                    else 'recorded')
        result.append(row)
    return pd.DataFrame(result)


def coverage_pipeline_report(out, pipeline):
    lines = ['# Coverage token usage including keypoint extraction', '',
        'Recorded totals across the selected saved runs, not complete historical spending. '
        'Extraction is counted once per saved extraction record; statistical extraction is shared '
        'across evaluator models and counted once overall. Missing counts are not treated as zero. '
        'These totals combine stage inventories, not matched transcript-summary pairs, so no '
        'per-summary pipeline mean is reported.', '',
        '| Strategy | Evaluation tokens | Extraction tokens | Combined recorded tokens | Extraction records with counts / total | Mean extraction tokens per usable record |',
        '|---|---:|---:|---:|---:|---:|']
    for row in pipeline.itertuples():
        lines.append(f'| {row.method_variant} | {row.evaluation_recorded_total_tokens:,.0f} | '
            f'{row.keypoint_extraction_recorded_total_tokens:,.0f} | '
            f'{row.combined_recorded_total_tokens:,.0f} | '
            f'{row.keypoint_extraction_records_with_usable_counts} / {row.keypoint_extraction_records} | '
            f'{row.mean_keypoint_extraction_total_tokens_per_usable_record:,.0f} |')
    lines += ['', 'Extraction means divide recorded extraction tokens by the number of extraction '
        'records with usable counts, pooling available models with equal weight per record. '
        'They are per transcript extraction, not per summary evaluation; missing counts are excluded '
        'from both numerator and denominator. Input, output, and total extraction means are in the CSV.']
    lines += ['', 'Complete extraction logs are available for Llama-3.1-8B narrow and wide '
        'and Gemma-31B wide. Statistical extraction has partial counts. Gemma-12B narrow/wide, '
        'Gemma-31B narrow, and Llama-70B narrow/wide have no usable extraction counts. '
        'See keypoint_extraction_usage.csv for the model-level breakdown.', '',
        'The existing token_usage_by_dimension_strategy.csv still reports evaluation-only means. '
        'Adding the extraction mean to that mean would mix per-transcript and per-summary units '
        'and different model subsets. A complete end-to-end average requires usable logs for '
        'both stages and an explicit allocation of reused extraction costs.']
    (out/'COVERAGE_PIPELINE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def grouped_report(out, grouped):
    lines = ['# Token usage grouped by dimension and strategy', '',
             'Each mean pools input plus output token counts across usable evaluation records '
             'from the listed models. Records have equal weight; model means are not averaged equally. '
             'One record evaluates one summary with one model and strategy, potentially using several LLM calls. '
             'Keypoint extraction is excluded from these means. Combined recorded stage totals '
             'are available in coverage_pipeline_token_usage.csv and COVERAGE_PIPELINE_REPORT.md; '
             'model-level extraction counts remain in keypoint_extraction_usage.csv.', '',
             '| Dimension | Strategy | Models contributing | Records used / in scope | Mean input | Mean output | Mean total |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in grouped.itertuples():
        lines.append('| ' + ' | '.join([row.dimension, row.method_variant, str(row.n_models_included),
            f'{row.records_used_for_average:,} / {row.records_in_scope:,}',
            f'{row.mean_prompt_tokens_per_evaluation:,.0f}', f'{row.mean_completion_tokens_per_evaluation:,.0f}',
            f'{row.mean_total_tokens_per_evaluation:,.0f}']) + ' |')
    lines += ['', 'Coherence, fluency and relevance pool available records from all four evaluator models. '
              'Factuality uses only Llama-3.1-8B and Llama-70B, with reconstructed cumulative counts. '
              'Gemma-12B has unavailable counters; Gemma-31B has partial counters for which a '
              'per-evaluation mean cannot be established. Missing or unverified records are not assigned zero.', '',
              'These are descriptive pooled averages, not comparisons on an identical set of inputs. '
              'Grouping does not recover missing usage or remove differences in model tokenizers. '
              'The original method CSV and full ledger preserve model-level details.', '',
              '`mean_total_tokens_usable_records` in token_usage_by_method.csv equals '
              '`total_tokens / records_with_usable_counts` when a per-record mean is interpretable. '
              'The extraction-stage version is per transcript extraction, not per summary. '
              'For example, Gemma-12B narrow relevance uses 5,508,727 / 657 = 8,384.67 tokens per evaluation.', '']
    (out/'GROUPED_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def report(out, summaries, dimensions, model_totals, inputs):
    def tokens(value):
        return '--' if pd.isna(value) else f'{value/1e6:.3f}'
    lines = [
        '# Recorded LLM token usage', '',
        'Scope: all saved records in the current thesis result folders, not only the human-annotated subset. '
        'Tokens are input/prompt plus output/completion tokens. No model calls were made.', '',
        'For the compact dimension/strategy view, see [GROUPED_REPORT.md](GROUPED_REPORT.md) '
        'and token_usage_by_dimension_strategy.csv.', '',
        '## Evaluation totals (millions of tokens)', '',
        '| Task | Gemma-12B | Gemma-31B | Llama-3.1-8B | Llama-70B |',
        '|---|---:|---:|---:|---:|',
    ]
    for dimension in ('coherence', 'fluency', 'factuality', 'relevance'):
        values = []
        for model in MODELS:
            row = dimensions[(dimensions.dimension == dimension) & (dimensions.evaluator_model == model)].iloc[0]
            value = tokens(row.total_tokens)
            if 'partial' in row.accounting_status:
                value += '*'
            if 'reconstructed' in row.accounting_status:
                value += '~'
            values.append(value)
        lines.append('| ' + ' | '.join([dimension] + values) + ' |')
    lines += ['', '* Partial recorded usage; ~ reconstructed from cumulative counters; -- unavailable. '
              'Coherence and fluency include both prompt variants. Relevance includes all three '
              'keypoint strategies and excludes separately saved extraction costs.', '',
              '## Usage by method', '',
              '| Task / stage | Variant | Model | Counted records | Usable counts | Input | Output | Total | Mean total per usable record |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries.itertuples():
        fmt = lambda v: '--' if pd.isna(v) else f'{v:,.0f}'
        lines.append('| ' + ' | '.join([f'{row.dimension} / {row.stage}', row.method_variant,
            row.evaluator_model, str(row.records_counted), str(row.records_with_usable_counts),
            fmt(row.prompt_tokens), fmt(row.completion_tokens), fmt(row.total_tokens),
            fmt(row.mean_total_tokens_usable_records)]) + ' |')
    lines += ['', '## Interpretation and accounting', '',
        '- A record is a summary evaluation, or one transcript extraction. Factuality and relevance '
        'can use multiple LLM calls per record. The means are not tokens per API call.',
        '- All totals describe retained results, not a complete historical bill. Failed attempts, '
        'overwritten runs, and unlogged usage cannot be recovered. No monetary cost or GPU time is inferred.',
        '- Only detailed coherence/fluency CSVs are counted. Summary CSVs, alternate extraction folders, '
        'old experiments and NLI/QA alternatives are listed as excluded in file_inventory.csv.',
        '- Byte-identical parsed records within a task/model/method are counted once. '
        'Distinct records with repeated filenames/profile labels remain counted. Without call IDs, '
        'identical repeat calls cannot be distinguished from duplicate exports; the full ledger and '
        'direct_counter_duplicate_sensitivity.csv retain the alternative totals.',
        '- Factuality counters are cumulative across files. Files follow the runners\' sorted input order; '
        'positive counters are differenced across file boundaries and restarted when both counters decrease. '
        'Zero placeholders are flagged, not treated as zero-cost evaluations or counter resets. '
        'Session IDs/timestamps are absent, so reconstructed counts are estimates. Inspect counter_events.csv.',
        '- Gemma-12B factuality has zero counters throughout; Gemma-31B factuality is only partly logged. '
        'The local Gemma factuality generator initializes counters but does not increment them in _chat. '
        'Those zero values cannot support an efficiency comparison.',
        '- Missing usage, zero counters on nonempty evaluations, and unavailable extraction counts are '
        'not imputed. Zero relevance counters with no key_items are retained as recorded zero-use cases.',
        '- Relevance evaluation logs contain row-level usage. Pre-extraction is reported separately '
        'and is not multiplied by the number of summaries. A row may include on-demand work if its '
        'preloaded keypoints were absent; the saved counts do not provide a call-level breakdown.',
        '- The statistical extraction file records llm_curation_mode=curate and positive LLM token '
        'counts. Its shared extraction cost is reported once; the actual curation model is not recorded '
        'in the result metadata. The model field there describes the input summary, not the evaluator.',
        '- Tokenizers differ across models. Token counts are usage measures, not directly comparable '
        'compute costs. Model means are also affected by input length, missing records and prompt variants.', '',
        '## Files', '',
        '- token_usage_by_method.csv: input/output/total counts and mean, median, p95 by model and method.',
        '- token_usage_by_dimension_model.csv: aggregate evaluation counts by task/model.',
        '- token_usage_by_model_evaluation.csv: known evaluation totals across all task variants; extraction excluded.',
        '- keypoint_extraction_usage.csv: separate extraction-stage accounting, including missing counts.',
        '- token_usage_records.csv: per-record counters, inclusion decisions and provenance.',
        '- counter_events.csv: starts, restarts, gaps and file-boundary increments for factuality.',
        '- file_inventory.csv and run_config.json: selected files, exclusions and input hashes.',
        '- token_usage_table.tex: LaTeX table with total tokens by evaluation method.', '',
        'Run: `python annotation_analysis/token_usage_analysis.py`',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tex = [r'% Requires booktabs and longtable.', r'\begin{longtable}{lllrrr}',
           r'\caption{Recorded LLM token usage for saved summary evaluations. Counts include all saved thesis-run inputs. Tokens are reported in thousands. A dash indicates unavailable usage; * denotes partial logging and \(\sim\) denotes cumulative-counter reconstruction. Extraction costs are reported separately.}\label{tab:llm-token-usage}\\',
           r'\toprule Task & Method & Evaluator & Input & Output & Total \\', r'\midrule\endhead']
    for row in summaries[summaries.stage.eq('evaluation')].itertuples():
        values = ['--' if pd.isna(v) else f'{v/1000:,.1f}' for v in (row.prompt_tokens, row.completion_tokens, row.total_tokens)]
        if 'partial' in row.accounting_status:
            values[-1] += '*'
        if 'reconstructed' in row.accounting_status:
            values[-1] += r'\(\sim\)'
        tex.append(' & '.join([row.dimension, row.method_variant.replace('_', r'\_'), row.evaluator_model] + values) + r' \\')
    tex += [r'\bottomrule', r'\end{longtable}']
    (out / 'token_usage_table.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir', type=Path, default=Path('annotation_analysis/analysis_output/token_usage'))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    collector = Collector(root)
    for model, (rel, coherence, fluency, factuality) in MODELS.items():
        collector.add_file(root/'coherence'/coherence/'coherence_ablation_detailed.csv', 'coherence', 'evaluation', None, model)
        collector.add_file(root/'fluency'/fluency/'fluency_evaluation_detailed.csv', 'fluency', 'evaluation', None, model)
        counter = CumulativeCounter()
        for path in sorted((root/'seval-ex'/factuality).glob('*.json')):
            collector.add_file(path, 'factuality', 'evaluation', 'default', model, counter)
        for variant in ('narrow', 'wide', 'statistical'):
            folders = list((root/'relevance'/rel).glob(f'entailment_evaluation_results_*_log_reconsider_{variant}_keypoints'))
            if len(folders) != 1:
                raise ValueError(f'Expected one current evaluation folder for {model}/{variant}')
            for path in sorted(folders[0].glob('*.json')):
                collector.add_file(path, 'relevance', 'evaluation', variant, model)
    for (model, variant), relative in EXTRACTIONS.items():
        collector.add_file(root/'relevance'/relative, 'relevance', 'keypoint_extraction', variant, model)
    rows = pd.DataFrame(collector.rows)
    usable = rows[rows.total_tokens.notna()]
    assert (usable.prompt_tokens + usable.completion_tokens == usable.total_tokens).all()
    methods = aggregate(rows, GROUP)
    evaluation = rows[rows.stage.eq('evaluation')]
    dimensions = aggregate(evaluation, ['dimension', 'evaluator_model'])
    model_totals = aggregate(evaluation, ['evaluator_model'])
    assert methods.total_tokens.sum() == usable.total_tokens.sum()
    assert dimensions.total_tokens.sum() == model_totals.total_tokens.sum()
    out.mkdir(parents=True, exist_ok=True)
    save(rows, out/'token_usage_records.csv')
    save(methods, out/'token_usage_by_method.csv')
    save(dimensions, out/'token_usage_by_dimension_model.csv')
    save(model_totals, out/'token_usage_by_model_evaluation.csv')
    save(methods[methods.stage.eq('keypoint_extraction')], out/'keypoint_extraction_usage.csv')
    grouped = grouped_evaluation_usage(methods)
    save(grouped, out/'token_usage_by_dimension_strategy.csv')
    grouped_report(out, grouped)
    pipeline = coverage_pipeline_usage(methods)
    save(pipeline, out/'coverage_pipeline_token_usage.csv')
    coverage_pipeline_report(out, pipeline)
    cumulative = rows[rows.dimension.eq('factuality')].copy()
    cumulative['file_boundary'] = cumulative.source_file.ne(cumulative.groupby('evaluator_model').source_file.shift())
    save(cumulative[cumulative.file_boundary | ~cumulative.counter_event.eq('counter_increment')], out/'counter_events.csv')
    direct = rows[~rows.dimension.eq('factuality')]
    sensitivity = direct.groupby(GROUP, dropna=False).agg(
        all_saved_record_tokens=('raw_total_tokens', lambda s: s.sum(min_count=1)),
        deduplicated_raw_tokens=('raw_total_tokens', lambda s: s[direct.loc[s.index, 'included']].sum(min_count=1)))
    save(sensitivity.reset_index(), out/'direct_counter_duplicate_sensitivity.csv')
    inventory = {entry['source_file']: entry | {'included': True, 'reason': 'Selected current thesis result file'} for entry in collector.files}
    for folder in ('coherence', 'fluency', 'seval-ex', 'relevance'):
        for path in (root/folder).rglob('*'):
            if not path.is_file() or '.git' in path.parts or path.suffix not in ('.json', '.jsonl', '.csv'):
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in inventory:
                inventory[relative] = {'source_file': relative, 'included': False,
                    'reason': 'Aggregate/export copy, alternate run or method, checkpoint, cache, or input; outside selected thesis runs'}
    save(pd.DataFrame(inventory.values()).sort_values('source_file'), out/'file_inventory.csv')
    config = {'scope': 'All retained records in selected current thesis result folders; not restricted to annotated subset',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_files': collector.files, 'model_mapping': MODELS, 'extraction_files': {f'{a}/{b}': c for (a,b),c in EXTRACTIONS.items()},
              'recorded_totals_are_not_full_historical_spend': True,
              'cumulative_order': 'Lexicographic file order, then original array order, per evaluator',
              'duplicate_policy': 'Exclude exact parsed-record duplicates within task/stage/method/model; retain differing records',
              'verification': {'component_sums_equal_total': True, 'group_totals_reconcile': True}}
    (out/'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report(out, methods, dimensions, model_totals, collector.files)
    print(dimensions[['dimension', 'evaluator_model', 'records_counted', 'records_without_usable_counts', 'total_tokens', 'accounting_status']].to_string(index=False))
    print(f'\nSaved {len(rows)} ledger rows and {len(methods)} method summaries to {out}')


if __name__ == '__main__':
    main()
