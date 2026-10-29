"""Reference baselines and paired comparisons with saved thesis evaluators.

No generation API calls, translation, or modification of source results.
UniEval follows https://github.com/maszhongming/UniEval (summarization):
official question formats, first-token Yes/(Yes+No), sentence means for
consistency/fluency, summary scores for coherence/relevance. German NLTK
sentence segmentation replaces its English default. UniEval is exploratory
because the released checkpoint was developed for English tasks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t as student_t

from data_inventory import read_csv, transcript_payload, digest

DIMENSIONS = ('coherence', 'factuality', 'fluency', 'relevance')
ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ('precision', 'recall', 'f1')


def name_key(value):
    return unicodedata.normalize('NFC', str(value).strip().replace('\\', '/'))


def summary_uid(source, filename, summary):
    return digest(json.dumps([source, name_key(filename), digest(summary)], ensure_ascii=False))


class GermanWordTokenizer:
    """Unicode words, lowercase, no English Porter stemming. Keeps ä/ö/ü/ß."""
    def tokenize(self, text):
        return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).lower(), flags=re.UNICODE)


def resolve_reference(filename, transcript, references):
    candidates = references.get(name_key(filename), [])
    exact = [r for r in candidates if digest(r['text']) == digest(transcript)]
    values = sorted({r['summary'].strip() for r in exact if r.get('summary', '').strip()})
    status = ('matched' if len(values) == 1 else 'ambiguous_reference' if len(values) > 1
              else 'transcript_mismatch' if candidates else 'reference_unavailable')
    return (values[0] if status == 'matched' else ''), status, len(candidates), len(exact)


def build_dataset(root):
    transcripts, source_summaries = {}, defaultdict(set)
    for path in sorted((root/'main_annotation/Results').glob('*.csv')):
        for row in read_csv(path):
            key = (row['source'], name_key(row['filename']))
            text = transcript_payload(row['transcript'])
            if key in transcripts and digest(transcripts[key]) != digest(text):
                raise ValueError(f'Conflicting annotated transcripts: {key}')
            transcripts[key] = text
            source_summaries[key].update(digest(row[f'summary_{slot}']) for slot in 'abc')
    references = defaultdict(list)
    for row in read_csv(root/'resources/summarization_corpus_20230815.csv', ';'):
        references[name_key(row['speaker_id'])].append(row)
    human = pd.read_csv(root/'annotation_analysis/analysis_output/human_analysis/human_scores_aggregated.csv')
    rows = []
    for record in human.to_dict('records'):
        key = (record['source'], name_key(record['filename']))
        if key not in transcripts or digest(record['summary_text']) not in source_summaries[key]:
            raise ValueError(f'Human aggregate does not match source annotations: {key}')
        reference, status, n_candidates, n_exact = resolve_reference(key[1], transcripts[key], references)
        rows.append(record | {'uid': summary_uid(*key, record['summary_text']),
            'cluster_id': json.dumps(key), 'transcript': transcripts[key],
            'reference': reference, 'reference_status': status,
            'reference_id_candidates': n_candidates, 'reference_text_matches': n_exact})
    data = pd.DataFrame(rows)
    if data.uid.duplicated().any():
        raise ValueError('Duplicate human summaries; refusing to give them extra weight.')
    return data


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.8f')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def cached_results(path, signature):
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('signature') == signature:
            return value
    return {'signature': signature, 'scores': {}, 'provenance': {}}


def dataset_signature(data, settings):
    inputs = [(r.uid, digest(r.transcript), digest(r.reference), digest(r.summary_text))
              for r in data.itertuples()]
    return digest(json.dumps({'inputs': inputs, 'settings': settings}, sort_keys=True))


def score_rouge(data):
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=GermanWordTokenizer(), use_stemmer=False)
    rows = {}
    for r in data.itertuples():
        scores = scorer.score(r.reference, r.summary_text)
        rows[r.uid] = {f'{metric}_{component}': float(getattr(value, 'fmeasure' if component == 'f1' else component))
                       for metric, value in scores.items() for component in COMPONENTS}
    return rows


def score_bertscore(data, args, out):
    import torch
    from bert_score import BERTScorer
    settings = {'model': args.bert_model, 'layer': args.bert_layer, 'idf': False,
                'rescale': False, 'package': importlib.metadata.version('bert-score'),
                'transformers': importlib.metadata.version('transformers')}
    path = out/'bertscore_cache.json'
    state = cached_results(path, dataset_signature(data, settings))
    pending = data[~data.uid.isin(state['scores'])]
    if pending.empty:
        print('BERTScore: using cached scores', flush=True)
        return state
    scorer = BERTScorer(model_type=args.bert_model, num_layers=args.bert_layer,
                       lang='de', idf=False, rescale_with_baseline=False,
                       batch_size=args.batch_size, device=args.device, use_fast_tokenizer=False)
    tokenizer = scorer._tokenizer
    limit = min(tokenizer.model_max_length, getattr(scorer._model.config, 'max_position_embeddings', 512))
    state['provenance'] = settings | {'hash': scorer.hash, 'token_limit': limit,
        'model_revision': getattr(scorer._model.config, '_commit_hash', None)}
    records = pending.to_dict('records')
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start+args.batch_size]
        p, r, f = scorer.score([x['summary_text'] for x in batch], [x['reference'] for x in batch])
        for row, precision, recall, f1 in zip(batch, p.tolist(), r.tolist(), f.tolist()):
            n_summary = len(tokenizer.encode(row['summary_text'], truncation=False))
            n_reference = len(tokenizer.encode(row['reference'], truncation=False))
            state['scores'][row['uid']] = {'bertscore_precision': precision, 'bertscore_recall': recall,
                'bertscore_f1': f1, 'bertscore_summary_tokens': n_summary,
                'bertscore_reference_tokens': n_reference,
                'bertscore_truncated': n_summary > limit or n_reference > limit}
        write_json(path, state)
        print(f'BERTScore: {len(state["scores"])}/{len(data)}', flush=True)
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def unieval_input(dimension, summary, source='', reference=''):
    # Fixed published UniEval input interface, not newly designed judge prompts.
    questions = {
        'coherence': ('Is this a coherent summary to the document?', 'summary', 'document', source),
        'factuality': ('Is this claim consistent with the document?', 'claim', 'document', source),
        'fluency': ('Is this a fluent paragraph?', 'paragraph', None, None),
        'relevance': ('Is this summary relevant to the reference?', 'summary', 'reference', reference),
    }
    question, label, context_label, context = questions[dimension]
    prompt = f'question: {question} </s> {label}: {summary}'
    return prompt if context_label is None else prompt + f' </s> {context_label}: {context}'


def prepare_unieval_jobs(record, sent_tokenize):
    sentences = sent_tokenize(record['summary_text']) or [record['summary_text']]
    jobs = []
    for dim in DIMENSIONS:
        units = sentences if dim in ('factuality', 'fluency') else [record['summary_text']]
        for unit in units:
            jobs.append((dim, unieval_input(dim, unit, record['transcript'], record['reference'])))
    return jobs


def score_unieval(data, args, out):
    import nltk
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    settings = {'model': args.unieval_model, 'max_length': args.unieval_max_length,
                'sentence_language': 'german', 'formula': 'first_token_yes_over_yes_plus_no',
                'transformers': importlib.metadata.version('transformers')}
    path = out/'unieval_cache.json'
    state = cached_results(path, dataset_signature(data, settings))
    pending = data[~data.uid.isin(state['scores'])]
    if pending.empty:
        print('UniEval: using cached scores', flush=True)
        return state
    nltk_path = out/'nltk_data'
    nltk.data.path.insert(0, str(nltk_path))
    try:
        nltk.sent_tokenize('Ein Satz. Noch ein Satz.', language='german')
    except LookupError:
        nltk.download('punkt_tab', download_dir=str(nltk_path), raise_on_error=True, quiet=True)
    tokenizer = AutoTokenizer.from_pretrained(args.unieval_model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.unieval_model).to(args.device).eval()
    yes = tokenizer.encode('Yes', add_special_tokens=False)
    no = tokenizer.encode('No', add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError('UniEval requires single-token Yes and No labels.')
    state['provenance'] = settings | {'model_revision': getattr(model.config, '_commit_hash', None),
        'exploratory_german_transfer': True, 'truncation_policy': 'official_right_truncation_audited',
        'source_code': 'https://github.com/maszhongming/UniEval'}
    for row in pending.to_dict('records'):
        jobs = prepare_unieval_jobs(row, lambda text: nltk.sent_tokenize(text, language='german'))
        values, lengths = defaultdict(list), defaultdict(list)
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            texts = [text for _, text in batch]
            original_lengths = [len(tokenizer.encode(text, truncation=False)) for text in texts]
            inputs = tokenizer(texts, padding=True, truncation=True,
                               max_length=args.unieval_max_length, return_tensors='pt').to(args.device)
            decoder = torch.full((len(batch), 1), model.config.decoder_start_token_id,
                                 device=args.device, dtype=torch.long)
            with torch.inference_mode():
                logits = model(**inputs, decoder_input_ids=decoder, use_cache=False).logits[:, 0, :]
                # Mathematically identical to P(Yes)/(P(Yes)+P(No)), more numerically stable.
                probabilities = torch.softmax(logits[:, [yes[0], no[0]]].float(), dim=-1)[:, 0].tolist()
            for (dim, _), value, length in zip(batch, probabilities, original_lengths):
                values[dim].append(value)
                lengths[dim].append(length)
        result = {}
        for dim in DIMENSIONS:
            result[f'unieval_{dim}'] = float(np.mean(values[dim]))
            result[f'unieval_{dim}_n_inputs'] = len(values[dim])
            result[f'unieval_{dim}_max_input_tokens'] = max(lengths[dim])
            result[f'unieval_{dim}_truncated_inputs'] = sum(n > args.unieval_max_length for n in lengths[dim])
        state['scores'][row['uid']] = result
        write_json(path, state)
        print(f'UniEval: {len(state["scores"])}/{len(data)}', flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def correlation_pair(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.array([np.nan, np.nan])
    return np.array([spearmanr(x, y).statistic, kendalltau(x, y, variant='b').statistic])


@lru_cache(maxsize=256)
def bootstrap_correlations(x_tuple, y_tuple, cluster_tuple, n_bootstrap, seed):
    """Reuse identical cohort/score calculations across comparator methods."""
    x, y, clusters = np.asarray(x_tuple), np.asarray(y_tuple), np.asarray(cluster_tuple)
    groups = [np.flatnonzero(clusters == k) for k in dict.fromkeys(cluster_tuple)]
    rng, draws = np.random.default_rng(seed), []
    if len(groups) >= 2:
        for _ in range(n_bootstrap):
            indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            draws.append(correlation_pair(x[indices], y[indices]))
    return correlation_pair(x, y), np.asarray(draws).reshape(-1, 2)


def correlation_stats(frame, n_bootstrap, seed, paired=False):
    """Resample whole transcripts. Paired differences use the SAME resamples."""
    human, clusters = tuple(frame.human_score.astype(float)), tuple(frame.cluster_id)
    base, draws = bootstrap_correlations(tuple(frame.baseline_score.astype(float)), human, clusters, n_bootstrap, seed)
    point = base
    if paired:
        proposed, proposed_draws = bootstrap_correlations(tuple(frame.proposed_score.astype(float)), human, clusters, n_bootstrap, seed)
        point = np.r_[base, proposed, proposed-base]
        draws = np.concatenate([draws, proposed_draws, proposed_draws-draws], axis=1)
    labels = ['baseline_spearman_rho', 'baseline_kendall_tau_b']
    if paired:
        labels += ['proposed_spearman_rho', 'proposed_kendall_tau_b', 'delta_spearman_rho', 'delta_kendall_tau_b']
    result = {'n_summaries': len(frame), 'n_transcripts': len(set(clusters))}
    for index, label in enumerate(labels):
        result[label] = point[index]
        valid = [v[index] for v in draws if np.isfinite(v[index])]
        low, high = np.quantile(valid, [.025, .975]) if valid else (np.nan, np.nan)
        result[label+'_ci_low'], result[label+'_ci_high'] = low, high
        result[label+'_valid_bootstraps'] = len(valid)
        cluster_se = np.std(valid, ddof=1) if len(valid) > 1 else np.nan
        result[label+'_cluster_bootstrap_se'] = cluster_se
        n_clusters = result['n_transcripts']
        if np.isfinite(cluster_se) and cluster_se > 0 and n_clusters > 1 and np.isfinite(point[index]):
            result[label+'_cluster_adjusted_p'] = float(
                2 * student_t.sf(abs(point[index] / cluster_se), df=n_clusters - 1))
        else:
            result[label+'_cluster_adjusted_p'] = np.nan
    return result


def baseline_long_form(data, scores):
    joined = data.merge(scores, on='uid', validate='one_to_one')
    rows = []
    for r in joined.to_dict('records'):
        for dim in DIMENSIONS:
            metrics = [m for m in ('rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bertscore_f1', f'unieval_{dim}') if m in r]
            for metric in metrics:
                truncated = (r.get('bertscore_truncated', False) if metric == 'bertscore_f1' else
                             r.get(f'unieval_{dim}_truncated_inputs', 0) > 0 if metric.startswith('unieval_') else False)
                rows.append({'uid': r['uid'], 'cluster_id': r['cluster_id'], 'dimension': dim,
                    'baseline': metric, 'baseline_score': r[metric], 'human_score': r[dim],
                    'input_truncated': bool(truncated)})
    return pd.DataFrame(rows)


def load_proposed(root, data):
    path = root/'annotation_analysis/analysis_output/automatic_analysis/human_automatic_merged.csv'
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame.join_status.eq('matched')].copy()
    frame['uid'] = [summary_uid(r.source, r.filename, r.summary_text) for r in frame.itertuples()]
    frame = frame[frame.uid.isin(data.uid)].copy()
    keys = ['uid', 'dimension', 'evaluator_model', 'method_variant', 'keypoint_type']
    frame[keys] = frame[keys].fillna('not_applicable')
    rows, audit = [], []
    for key, group in frame.groupby(keys, sort=True):
        values = pd.to_numeric(group.automatic_score, errors='coerce').dropna().unique()
        status = 'matched' if len(values) == 1 else 'conflicting_scores' if len(values) > 1 else 'missing_score'
        audit.append(dict(zip(keys, key)) | {'status': status, 'saved_rows': len(group)})
        if len(values) == 1:
            rows.append(dict(zip(keys, key)) | {'proposed_score': float(values[0])})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def analyse(data, scores, root, out, n_bootstrap, seed):
    baselines = baseline_long_form(data, scores)
    save_csv(baselines, out/'baseline_human_scores.csv')
    validity = []
    for (dim, metric), group in baselines.groupby(['dimension', 'baseline']):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=['baseline_score', 'human_score'])
        for subset, sample in [('all_scored', group), ('untruncated', group[~group.input_truncated])]:
            if sample.empty:
                continue
            rho, rho_p_iid = spearmanr(sample.baseline_score, sample.human_score)
            tau, tau_p_iid = kendalltau(sample.baseline_score, sample.human_score, variant='b')
            cluster_stats = correlation_stats(sample, n_bootstrap, seed)
            validity.append({'dimension': dim, 'baseline': metric, 'subset': subset,
                'n_unique_summaries': sample.uid.nunique(),
                'spearman_rho': rho, 'spearman_p': cluster_stats['baseline_spearman_rho_cluster_adjusted_p'],
                'spearman_p_iid': rho_p_iid,
                'kendall_tau_b': tau, 'kendall_p': cluster_stats['baseline_kendall_tau_b_cluster_adjusted_p'],
                'kendall_p_iid': tau_p_iid,
                'n_truncated': int(sample.input_truncated.sum()),
                **cluster_stats})
    validity = pd.DataFrame(validity)
    save_csv(validity, out/'baseline_metric_validity.csv')
    proposed, audit = load_proposed(root, data)
    save_csv(audit, out/'proposed_join_audit.csv')
    matched = baselines.merge(proposed, on=['uid', 'dimension'], how='inner', validate='many_to_many')
    save_csv(matched, out/'paired_comparison_records.csv')
    results = []
    keys = ['dimension', 'baseline', 'evaluator_model', 'method_variant', 'keypoint_type']
    for key, group in matched.groupby(keys, sort=True):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=['baseline_score', 'human_score', 'proposed_score'])
        if group.uid.duplicated().any():
            raise ValueError('Duplicate summary within paired comparison.')
        for subset, sample in [('all_scored', group), ('untruncated', group[~group.input_truncated])]:
            if sample.empty:
                continue
            # Reuse statistics when the sensitivity subset is identical.
            stats = (previous if subset == 'untruncated' and len(sample) == len(group)
                     else correlation_stats(sample, n_bootstrap, seed, paired=True))
            previous = stats
            results.append(dict(zip(keys, key)) | {'subset': subset,
                'n_truncated': int(sample.input_truncated.sum()), **stats})
        print('Compared ' + ' / '.join(key), flush=True)
    comparisons = pd.DataFrame(results)
    save_csv(comparisons, out/'paired_metric_comparisons.csv')
    return validity, comparisons


def render_report(out, data, audit, scores, validity, errors, args):
    counts = audit.drop_duplicates('cluster_id').reference_status.value_counts().to_dict()
    lines = ['# Reference baseline comparisons', '',
        f'Reference audit: {counts}. Main cohort: {len(data)} summaries from {data.cluster_id.nunique()} transcripts.', '',
        'References require matching transcript ID AND whitespace-normalized transcript text. '
        'No references are fabricated for other transcripts. Human ratings come from the existing consensus file. '
        'All proposed-metric correlations in paired_metric_comparisons.csv are recomputed on precisely the '
        'same summaries as that baseline. Original full-dataset correlations are not used as comparators.', '',
        'ROUGE uses Unicode word tokenization without stemming; main scores are ROUGE-1/2/L F1. '
        'BERTScore uses bert-base-multilingual-cased layer 9 by default, German, no IDF, no rescaling; '
        'the exact model hash and settings are stored in the cache. Precision and recall are also saved.', '',
        'UniEval uses MingZhong/unieval-sum, original English question templates on untranslated German text, '
        'German NLTK sentence segmentation, and a mean sentence score for factuality (consistency) and fluency. '
        'Coherence and relevance use full-summary inputs. This is exploratory transfer: the original model '
        'was developed for English, so poor German results do not establish inferiority on its intended domain.', '',
        f'UniEval uses the official right-truncation policy at {args.unieval_max_length} input tokens. '
        'BERTScore also has a model input limit. Lengths and truncation flags are saved per summary/dimension; '
        'both all_scored and untruncated sensitivity comparisons are reported. Truncation can omit source evidence.', '',
        f'Confidence intervals and cluster-adjusted p-values use {args.bootstrap_samples} transcript-cluster bootstrap draws (seed {args.seed}). '
        'Paired correlation differences (proposed minus baseline) share identical resamples. '
        'P-values use two-sided Wald tests with the cluster-bootstrap standard error and a t reference distribution with '
        'degrees of freedom equal to the number of transcripts minus one; IID Spearman/Kendall p-values are retained in '
        'separate *_p_iid columns. P-values are not adjusted for multiple comparisons. Constant/insufficient samples have undefined correlations.', '',
        'Main file: paired_metric_comparisons.csv. Baseline-only correlations: baseline_metric_validity.csv. '
        'Inspect n_summaries and n_transcripts: different proposed methods may have different availability. '
        'Reference similarity scores are not direct factual-error detectors or span-localization metrics.', '']
    if errors:
        lines += ['## Incomplete scoring', '', *[f'- {metric}: {message}' for metric, message in errors.items()], '']
    if not validity.empty:
        table = validity[validity.subset.eq('all_scored')]
        lines += ['| Dimension | Baseline | n | Spearman | Kendall | Truncated |', '|---|---|---:|---:|---:|---:|']
        tex = [r'% Requires booktabs. UniEval is exploratory German transfer.', r'\begin{tabular}{llrrr}',
               r'\toprule', r'Dimension & Baseline & $n$ & $\rho$ & $\tau_b$ \\', r'\midrule']
        for r in table.itertuples():
            lines.append(f'| {r.dimension} | {r.baseline} | {r.n_summaries} | {r.baseline_spearman_rho:.3f} | {r.baseline_kendall_tau_b:.3f} | {r.n_truncated} |')
            tex.append(' & '.join([r.dimension, r.baseline.replace('_', r'\_'), str(r.n_summaries),
                       f'{r.baseline_spearman_rho:.3f}', f'{r.baseline_kendall_tau_b:.3f}']) + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}']
        (out/'baseline_table.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    lines += ['', '## Sources', '',
        '- https://github.com/google-research/google-research/tree/master/rouge',
        '- https://github.com/Tiiiger/bert_score',
        '- https://github.com/maszhongming/UniEval',
        '- https://aclanthology.org/2022.emnlp-main.131/ (including English-language limitation)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--metrics', nargs='+', choices=['rouge', 'bertscore', 'unieval'], default=['rouge', 'bertscore', 'unieval'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--score-only', action='store_true', help='Score/cache without bootstrap comparisons.')
    parser.add_argument('--analysis-only', action='store_true', help='Recompute comparisons from saved baseline_dataset.csv and baseline_scores.csv.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--bert-model', default='bert-base-multilingual-cased')
    parser.add_argument('--bert-layer', type=int, default=9)
    parser.add_argument('--unieval-model', default='MingZhong/unieval-sum')
    parser.add_argument('--unieval-max-length', type=int, default=1024)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 0 or args.torch_threads < 1 or args.unieval_max_length < 1:
        parser.error('Batch size, threads and max length must be positive; bootstrap count cannot be negative.')
    out = args.output_dir or args.root/'annotation_analysis/analysis_output/reference_baselines'
    out.mkdir(parents=True, exist_ok=True)
    if args.analysis_only:
        data = pd.read_csv(out/'baseline_dataset.csv', low_memory=False)
        scores = pd.read_csv(out/'baseline_scores.csv', low_memory=False)
        audit = pd.read_csv(out/'reference_match_audit.csv', low_memory=False)
        config_path = out/'run_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        errors = config.get('errors', {})
        validity, _ = analyse(data, scores, args.root, out, args.bootstrap_samples, args.seed)
        render_report(out, data, audit, scores, validity, errors, args)
        config['analysis'] = {'bootstrap_samples': args.bootstrap_samples, 'seed': args.seed,
            'cluster_unit': 'transcript', 'p_value_method': 'two-sided Wald t test using cluster-bootstrap SE; df=n_transcripts-1'}
        config['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(config_path, config)
        print(f'Recomputed cluster-adjusted results from saved scores in {out}', flush=True)
        return
    data = build_dataset(args.root)
    audit = data.drop(columns=['transcript', 'reference', 'summary_text'])
    save_csv(audit, out/'reference_match_audit.csv')
    data = data[data.reference_status.eq('matched')].copy().sort_values('uid')
    if data.empty:
        raise ValueError('No matched references.')
    save_csv(data, out/'baseline_dataset.csv')
    print(f'Reference-matched cohort: {len(data)} summaries / {data.cluster_id.nunique()} transcripts.', flush=True)
    if args.prepare_only:
        return
    if any(m != 'rouge' for m in args.metrics):
        import torch
        torch.set_num_threads(args.torch_threads)
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scores = data[['uid']].copy()
    errors, provenance = {}, {}
    for metric in args.metrics:
        try:
            if metric == 'rouge':
                state = {'scores': score_rouge(data), 'provenance': {'tokenizer': 'UnicodeWordLowercase', 'stemmer': False}}
            else:
                state = (score_bertscore if metric == 'bertscore' else score_unieval)(data, args, out)
            metric_frame = pd.DataFrame.from_dict(state['scores'], orient='index').rename_axis('uid').reset_index()
            scores = scores.merge(metric_frame, on='uid', validate='one_to_one')
            provenance[metric] = state['provenance']
            save_csv(scores, out/'baseline_scores.csv')
        except Exception as exc:
            # Preserve other baselines and report failure instead of substituting scores.
            errors[metric] = f'{type(exc).__name__}: {exc}'
            print(f'{metric} FAILED: {errors[metric]}', flush=True)
    save_csv(scores, out/'baseline_scores.csv')
    versions = {}
    for package in ('numpy', 'pandas', 'scipy', 'torch', 'transformers', 'bert-score', 'rouge-score', 'nltk'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'provenance': provenance, 'versions': versions, 'errors': errors,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256((out/'baseline_dataset.csv').read_bytes()).hexdigest()}
    write_json(out/'run_config.json', config)
    validity = pd.DataFrame()
    if not args.score_only and len(scores.columns) > 1:
        validity, _ = analyse(data, scores, args.root, out, args.bootstrap_samples, args.seed)
    render_report(out, data, audit, scores, validity, errors, args)
    if errors:
        raise SystemExit('Incomplete baseline run; see REPORT.md and run_config.json. Successful scores were saved.')
    print(f'Saved results to {out}', flush=True)


if __name__ == '__main__':
    main()
