"""Build a shared statistical-keypoint row from canonical extraction sets.

Saved model-specific results remain unchanged. No embedding inference is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fuzzy_position_agreement import GROUP, save_csv, summarize, digest, json_text
from fuzzy_span_alignment import SourceText, localize
from human_analysis import normalized_filename
from semantic_fallback_agreement import references_from_baseline, rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(
        'annotation_analysis/analysis_output/fuzzy_semantic_position_agreement'))
    parser.add_argument('--keypoints', type=Path, default=Path(
        'relevance/statistical_keypoints_results_summaries_strict_9/statistical_kp_dedup.json'))
    args = parser.parse_args()
    run = args.run_dir
    canonical = {}
    for record in json.loads(args.keypoints.read_text(encoding='utf-8')):
        key = (record['variant'].lower() + '::' + normalized_filename(record['filename']),
               record['transcript_hash'])
        texts = tuple(sorted({text.strip() for text in record['key_points'] if text.strip()}))
        if key in canonical and canonical[key] != texts:
            raise ValueError(f'Conflicting canonical keypoints: {key}')
        canonical[key] = texts

    sets = pd.read_csv(run / 'extraction_sets.csv')
    statistical = sets[sets.keypoint_type.eq('statistical')].copy()
    documents = {doc['document_id']: doc for doc in map(json.loads,
                 (run / 'source_documents.jsonl').read_text(encoding='utf-8').splitlines())}
    localizations = pd.read_csv(run / 'automatic_localizations.csv')
    all_copies = localizations[localizations.keypoint_type.eq('statistical')]
    saved = {}
    for key, group in all_copies.groupby(['document_id', 'extracted_text']):
        for column in ('accepted', 'selected_spans_json'):
            assert group[column].nunique(dropna=False) == 1, column
        saved[key] = group.iloc[0].to_dict()
    units = pd.read_csv(run / 'position_agreement_by_unit.csv', dtype={'annotator': str})
    statistical_units = units[units.keypoint_type.eq('statistical')]
    for _, group in statistical_units.groupby(['document_id', 'annotator']):
        for column in ('comparison_status', 'human_tokens'):
            assert group[column].nunique(dropna=False) == 1, column
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    fuzzy_config = config.get('baseline_config', {})
    selected_sets, items, comparison_rows, mismatches = [], [], [], []
    new_localizations = 0
    for doc_id, group in statistical.groupby('document_id'):
        document = documents[doc_id]
        source_hash = hashlib.sha256(document['text'].strip().encode('utf-8')).hexdigest()
        texts = canonical[(normalized_filename(document['item_key']), source_hash)]
        set_id = digest(json_text(['coverage_keypoints', 'not_applicable', 'statistical', doc_id, texts]))
        meta = {'set_id': set_id, 'analysis': 'coverage_keypoints',
                'evaluator_model': 'not_applicable', 'keypoint_type': 'statistical',
                'document_id': doc_id, 'item_key': document['item_key'], 'summary_id': ''}
        selected_sets.append(meta | {'texts': json_text(texts), 'transcript_hash': source_hash})
        for _, row in group.iterrows():
            if tuple(json.loads(row.texts)) != texts:
                mismatches.append({'set_id': row.set_id, 'evaluator_model': row.evaluator_model,
                                   'item_key': row.item_key})
        for text in texts:
            if (doc_id, text) in saved:
                item = dict(saved[doc_id, text])
            else:
                result = localize(text, SourceText(document['text']),
                                  min_similarity=fuzzy_config.get('min_similarity', 65),
                                  min_coverage=fuzzy_config.get('min_coverage', .65),
                                  ambiguity_policy='best')
                if not result['selected_spans']:
                    raise ValueError(f'New canonical item requires semantic fallback: {document["item_key"]}')
                item = {'extracted_text': text, 'accepted': True,
                        'status': result['status'], 'selected_span_count': 1,
                        'selected_spans_json': json_text(result['selected_spans']),
                        'localization_method': 'fuzzy', 'semantic_attempted': False,
                        **result['selected_spans'][0]}
                new_localizations += 1
            item.update(meta)
            items.append(item)
        for row in statistical_units[statistical_units.document_id.eq(doc_id)].drop_duplicates('annotator').to_dict('records'):
            row.update(meta)
            row['automatic_items'] = len(texts)
            comparison_rows.append(row)
    shared = pd.DataFrame(items)
    assert not shared.duplicated(['document_id', 'extracted_text']).any()
    audit_spans = pd.read_csv(run / 'human_span_validation.csv', dtype={'annotator': str})
    refs = references_from_baseline(units, audit_spans)
    comparisons = rescore(shared, pd.DataFrame(comparison_rows), documents, refs)
    assert not comparisons.duplicated(['document_id', 'annotator']).any()
    for row in shared.to_dict('records'):
        for span in json.loads(row['selected_spans_json']):
            assert documents[row['document_id']]['text'][span['start']:span['end']] == span['quote']
    summary = pd.DataFrame(summarize(comparisons, config.get('bootstrap', 1000)))
    summary['localization_rate'] = shared.accepted.mean()
    summary['automatic_items'] = len(shared)
    summary['accepted_items'] = int(shared.accepted.sum())
    save_csv(summary, run / 'statistical_shared_summary.csv')
    save_csv(shared, run / 'statistical_shared_localizations.csv')
    save_csv(comparisons, run / 'statistical_shared_comparisons.csv')
    save_csv(selected_sets, run / 'statistical_shared_extraction_sets.csv')
    remaining = pd.read_csv(run / 'position_agreement_summary.csv')
    remaining = remaining[~remaining.keypoint_type.eq('statistical')]
    rates = pd.read_csv(run / 'localization_summary.csv')
    remaining = remaining.merge(rates[GROUP + ['localization_rate', 'automatic_items', 'accepted_items']],
                                on=GROUP, validate='one_to_one')
    save_csv(pd.concat([remaining, summary], ignore_index=True), run / 'thesis_position_agreement_summary.csv')
    audit = {'canonical_file': str(args.keypoints.resolve()),
             'canonical_sha256': hashlib.sha256(args.keypoints.read_bytes()).hexdigest(),
             'source_hashes': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                               for name in ('extraction_sets.csv', 'automatic_localizations.csv',
                                            'position_agreement_by_unit.csv', 'run_config.json')},
             'shared_documents': len(selected_sets), 'automatic_items': len(shared),
             'canonical_selection': 'filename and SHA256 of stripped source transcript',
             'new_fuzzy_localizations': new_localizations,
             'saved_item_locations_verified_identical': True,
             'excluded_noncanonical_sets': mismatches}
    (run / 'statistical_shared_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
