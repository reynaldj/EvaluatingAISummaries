import argparse
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    import nltk
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


DECISION_MARKERS = {
    r'\bbeschlossen\b': 0.15, r'\bbeschließen\b': 0.15,
    r'\bEntscheidung\b': 0.12, r'\bentschieden\b': 0.12,
    r'\bErgebnis\b': 0.12, r'\bfestgelegt\b': 0.10,
    r'\bvereinbart\b': 0.12, r'\bEinigung\b': 0.10,
    r'\beinigt\b': 0.10, r'\babgestimmt\b': 0.10,
    r'\bMehrheit\b': 0.08, r'\babgelehnt\b': 0.10,
    r'\bangenommen\b': 0.10, r'\bzugestimmt\b': 0.10,
}

ACTION_MARKERS = {
    r'\bMaßnahme\b': 0.12, r'\bnächster Schritt\b': 0.15,
    r'\bnächste Schritte\b': 0.15, r'\bAufgabe\b': 0.08,
    r'\bzuständig\b': 0.08, r'\bverantwortlich\b': 0.08,
    r'\bDeadline\b': 0.10, r'\bbis zum\b': 0.06,
    r'\bumsetzen\b': 0.08, r'\bbeauftragt\b': 0.10,
    r'\bUmsetzung\b': 0.08, r'\bPlanung\b': 0.06,
}

RESULT_MARKERS = {
    r'\bFazit\b': 0.10, r'\babschließend\b': 0.06,
    r'\binsgesamt\b': 0.05, r'\bletztendlich\b': 0.06,
    r'\bZusammenfassung\b': 0.08, r'\bSchluss\b': 0.06,
    r'\bfesthalten\b': 0.08, r'\bfeststellen\b': 0.06,
}

POSITION_MARKERS = {
    r'\bfordern?\b': 0.08, r'\blehnen?\s+ab\b': 0.10,
    r'\bkritisieren?\b': 0.08, r'\bunterstützen?\b': 0.08,
    r'\bbefürworten?\b': 0.08, r'\bablehnen\b': 0.10,
    r'\bsprechen?\s+sich\s+(für|gegen)\b': 0.10,
    r'\bPosition\b': 0.06, r'\bStandpunkt\b': 0.06,
}

INFORMAL_DECISION_MARKERS = {
    r'\bmachen wir\b': 0.12, r'\bnehmen wir mit\b': 0.12,
    r'\bbleibt so\b': 0.10, r'\bso machen\b': 0.10,
    r'\bist geklärt\b': 0.10, r'\bsteht fest\b': 0.12,
    r'\bwird gemacht\b': 0.10, r'\bist erledigt\b': 0.10,
    r'\bkümmere? (ich|mich|sich)\b': 0.10, r'\bübernehme\b': 0.10,
    r'\bTODO\b': 0.12, r'\baction item\b': 0.12,
    r'\bfestgehalten\b': 0.10, r'\bnotiert\b': 0.08,
    r'\bmüssen wir\b': 0.08, r'\bsollten wir\b': 0.08,
    r'\bwichtig ist\b': 0.08, r'\bentscheidend\b': 0.10,
}

GERMAN_FILLER_WORDS = {
    'also', 'halt', 'nochmal', 'einfach', 'irgendwie', 'sozusagen',
    'quasi', 'eigentlich', 'natürlich', 'eben', 'ja', 'ne', 'na',
    'gut', 'genau', 'okay', 'naja', 'ähm', 'äh', 'hmm', 'mhm',
    'bisschen', 'irgendwann', 'irgendwas', 'irgendein', 'vielleicht',
    'sagen', 'mal', 'jetzt', 'dann', 'schon', 'noch', 'so', 'da',
    'hier', 'wirklich', 'praktisch', 'grundsätzlich',
}

RHETORIC_MARKERS = {
    r'\bich bin (fassungslos|dankbar|überzeugt|sicher|der Meinung)\b': 0.15,
    r'\b(wir|ich) (setzen|senden|geben)\s+ein\w*\s+(Signal|Zeichen)\b': 0.15,
    r'\bes ist (unsere|meine) (Aufgabe|Pflicht|Verantwortung)\b': 0.12,
    r'\bnie wieder\b': 0.10,
    r'\b(ganz )?(klar|ehrlich|offen) (gesagt|formuliert)\b': 0.06,
    r'\b(ausdrücklich|unbedingt|dringend|nachdrücklich)\b': 0.06,
    r'\bdas (ist|war) (wichtig|essentiell|fundamental)\b': 0.08,
    r'\bich (appelliere|bitte|werbe|danke)\b': 0.10,
    r'\b(meine Damen und Herren|liebe Kolleg|werte Beigeordnete)\b': 0.08,
    r'\b(Danke|Dankeschön|Vielen Dank)\b': 0.10,
    r'\bich (finde|glaube|meine|denke),?\s+(dass|es)\b': 0.08,
    r'\bin diesem Sinne\b': 0.06,
    r'\blassen Sie (mich|uns)\b': 0.06,
    r'\b(steht|stehen) (uns|dem Stadtrat) gut zu Gesicht\b': 0.08,
}

VAGUE_PATTERNS = {
    r'\bman (sollte|müsste|könnte|muss)\b': 0.08,
    r'\b(es|das) (braucht|muss|ist nötig|gilt)\b': 0.06,
    r'\bwir (müssen|sollten|dürfen nicht|können nicht)\b': 0.06,
    r'\b(es|das) ist (Zeit|an der Zeit|notwendig)\b': 0.06,
}

ALL_DISCOURSE_MARKERS = {}
ALL_DISCOURSE_MARKERS.update(DECISION_MARKERS)
ALL_DISCOURSE_MARKERS.update(ACTION_MARKERS)
ALL_DISCOURSE_MARKERS.update(RESULT_MARKERS)
ALL_DISCOURSE_MARKERS.update(POSITION_MARKERS)
ALL_DISCOURSE_MARKERS.update(INFORMAL_DECISION_MARKERS)


class StatisticalKeyPointExtractor:
    def __init__(
        self,
        embedding_model_name="intfloat/multilingual-e5-large",
        llm_model_name="meta-llama-3.1-8b-instruct",
        use_llm_compression=True,
        llm_curation_mode=None,
        mmr_lambda=0.7,
        target_percentage=0.05,
        seg_threshold=0.3,
    ):
        self.api_key = "ce64487341616c6d2455a926db59e9c4"
        self.base_url = "https://chat-ai.academiccloud.de/v1"
        self.llm_model_name = llm_model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.use_llm_compression = use_llm_compression
        if llm_curation_mode is None:
            llm_curation_mode = "curate" if use_llm_compression else "none"
        if llm_curation_mode not in {"none", "curate", "curate_informative"}:
            raise ValueError("llm_curation_mode must be one of: none, curate, curate_informative")
        self.llm_curation_mode = llm_curation_mode
        self.mmr_lambda = mmr_lambda
        self.target_percentage = target_percentage
        self.seg_threshold = seg_threshold
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        print("Loading SentenceTransformer model...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as exc:
            print(f"  [Warning] Failed to load {embedding_model_name} ({exc}); falling back to paraphrase-multilingual-MiniLM-L12-v2.")
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # ─── API ────────────────────────────────────────────────────────────

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    extra_body={}
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                last_error = "empty response body"
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if "429" in error_msg:
                    wait = 5 + 3 * attempt + random.uniform(0, 2)
                    print(f"    [Rate Limit] Waiting {wait:.0f}s (attempt {attempt+1}/{max_attempts})...")
                elif any(s in error_msg for s in ("500", "502", "503", "timeout", "timed out", "connection")):
                    wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"    [Server Error] Retrying in {wait:.0f}s (attempt {attempt+1}/{max_attempts})...")
                else:
                    wait = 2 + 2 * attempt + random.uniform(0, 1)
                    print(f"    [API Error] {error_msg[:100]} (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)

        raise SystemExit(
            f"\n[FATAL] API call failed after {max_attempts} attempts. Last error: {last_error}"
        )

    # ─── Stage 1: Transcript Parsing ────────────────────────────────────

    def parse_transcript_text(self, content):
        turns = []
        speaker_bracket_pattern = r'\[(Speaker \d+)\]:\s*(.*?)(?=\n\[Speaker \d+\]:|\Z)'
        timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*?)\n(.*?)(?=\n\n\[|\Z)'
        cleaned_pattern = r'(Sprecher \d+|[\w\s]+)\n(.*?)(?=\n(?:Sprecher \d+|[\w\s]+)\n|\Z)'

        matches = list(re.finditer(speaker_bracket_pattern, content, re.DOTALL))
        if len(matches) > 1:
            for i, match in enumerate(matches):
                turns.append({
                    'timestamp': f"Turn-{i+1:03d}",
                    'speaker': match.group(1).strip(),
                    'text': match.group(2).strip()
                })
        else:
            matches = list(re.finditer(timestamp_pattern, content, re.DOTALL))
            if matches:
                for match in matches:
                    turns.append({
                        'timestamp': match.group(1),
                        'speaker': match.group(2).strip(),
                        'text': match.group(3).strip().replace('\n', ' ')
                    })
            else:
                matches = list(re.finditer(cleaned_pattern, content, re.DOTALL))
                if len(matches) > 1:
                    for i, match in enumerate(matches):
                        turns.append({
                            'timestamp': f"Turn-{i+1:03d}",
                            'speaker': match.group(1).strip(),
                            'text': match.group(2).strip().replace('\n', ' ')
                        })
                else:
                    sentences = re.split(r'(?<=[.!?]) +', content)
                    for i, sent in enumerate(sentences):
                        if len(sent.strip()) > 5:
                            turns.append({
                                'timestamp': f"Sent-{i+1:03d}",
                                'speaker': "Speaker",
                                'text': sent.strip()
                            })
        return turns

    # ─── Stage 1: Topic Segmentation ───────────────────────────────────

    def run_topic_segmentation(self, turns):
        if not turns or len(turns) < 5:
            return [turns] if turns else []

        texts = [turn['text'] for turn in turns]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        num_turns = len(turns)
        if num_turns > 100:
            window_size = 7
            min_segment_size = 10
            max_turns = 50
        elif num_turns > 30:
            window_size = 5
            min_segment_size = 7
            max_turns = 35
        else:
            window_size = 3
            min_segment_size = 4
            max_turns = 20

        distances = []
        for i in range(window_size, len(embeddings) - window_size):
            window_before = embeddings[i - window_size:i].mean(axis=0).reshape(1, -1)
            window_after = embeddings[i:i + window_size].mean(axis=0).reshape(1, -1)
            sim = cosine_similarity(window_before, window_after)[0][0]
            distances.append(1 - sim)

        full_distances = [0] * window_size + distances + [0] * window_size

        # Use adaptive threshold: if fixed threshold yields no boundaries, use percentile
        boundaries = set(np.where(np.array(full_distances) > self.seg_threshold)[0])
        if not boundaries and distances:
            adaptive_threshold = np.percentile(distances, 75)
            boundaries = set(np.where(np.array(full_distances) > adaptive_threshold)[0])

        segments = []
        current_segment = []
        for i, turn in enumerate(turns):
            current_segment.append(turn)
            is_boundary = (i in boundaries and len(current_segment) >= min_segment_size)
            is_hard_limit = len(current_segment) >= max_turns
            if is_boundary or is_hard_limit:
                segments.append(current_segment)
                current_segment = []

        if current_segment:
            if segments and len(current_segment) < min_segment_size:
                segments[-1].extend(current_segment)
            else:
                segments.append(current_segment)

        # Fallback: if still only 1 segment with many turns, split by speaker changes
        if len(segments) == 1 and num_turns > 10:
            segments = self._split_by_speaker_blocks(turns, min_segment_size)

        return segments

    def _split_by_speaker_blocks(self, turns, min_size=4):
        segments = []
        current = []
        prev_speaker = None
        for turn in turns:
            if turn['speaker'] != prev_speaker and len(current) >= min_size:
                segments.append(current)
                current = []
            current.append(turn)
            prev_speaker = turn['speaker']
        if current:
            if segments and len(current) < min_size:
                segments[-1].extend(current)
            else:
                segments.append(current)
        return segments if len(segments) > 1 else [turns]

    # ─── Stage 1: Sentence Splitting ───────────────────────────────────

    def segment_into_sentences(self, segments):
        candidates = []
        for seg_id, segment in enumerate(segments):
            for turn_idx, turn in enumerate(segment):
                text = turn['text']
                sents = self._split_german_sentences(text)
                for sent_idx, sent in enumerate(sents):
                    words = sent.split()
                    if len(words) < 6:
                        continue
                    filler_ratio = self._compute_filler_ratio(words)
                    if filler_ratio > 0.45:
                        continue
                    content_word_count = len(words) - int(filler_ratio * len(words))
                    if content_word_count < 8:
                        continue
                    if sent.count('...') >= 2 or sent.rstrip().endswith(('...', '…', ',')):
                        continue
                    candidates.append({
                        'text': sent,
                        'speaker': turn['speaker'],
                        'turn_index': turn_idx,
                        'sentence_index': sent_idx,
                        'segment_id': seg_id,
                        'position_in_segment': turn_idx / max(1, len(segment) - 1),
                        'is_first_in_segment': (turn_idx == 0 and sent_idx == 0),
                        'is_last_in_segment': (turn_idx == len(segment) - 1 and sent_idx == len(sents) - 1),
                        'filler_ratio': filler_ratio,
                    })
        return candidates

    def _compute_filler_ratio(self, words):
        if not words:
            return 1.0
        lower_words = [w.lower().strip('.,;:!?()[]"\'') for w in words]
        filler_count = sum(1 for w in lower_words if w in GERMAN_FILLER_WORDS)
        return filler_count / len(lower_words)

    def _split_german_sentences(self, text):
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text, language='german')
        except Exception:
            sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if s.strip()]

    # ─── Stage 2: Salience Scoring ─────────────────────────────────────

    def score_salience(self, candidates, full_transcript_text):
        if not candidates:
            return []

        candidate_texts = [c['text'] for c in candidates]

        # Split full transcript into sentences for BM25 corpus
        all_sentences = self._split_german_sentences(full_transcript_text)
        if not all_sentences:
            all_sentences = [full_transcript_text]

        # --- BM25 (length-normalized) ---
        tokenized_corpus = [self._bm25_tokenize(s) for s in all_sentences]
        tokenized_corpus = [t for t in tokenized_corpus if t]
        if not tokenized_corpus:
            bm25_scores = np.zeros(len(candidates))
        else:
            bm25 = BM25Okapi(tokenized_corpus)
            raw_bm25 = np.array([
                np.mean(bm25.get_scores(self._bm25_tokenize(c['text'])))
                for c in candidates
            ])
            # Penalize long sentences: normalize by sqrt of word count
            lengths = np.array([max(1, len(c['text'].split())) for c in candidates])
            length_penalty = np.sqrt(lengths) / np.sqrt(np.median(lengths))
            bm25_scores = raw_bm25 / length_penalty
        bm25_norm = self._sigmoid_normalize(bm25_scores)

        # --- Embedding centrality ---
        candidate_embeddings = self.embedding_model.encode(candidate_texts, show_progress_bar=False)
        all_embeddings = self.embedding_model.encode(all_sentences, show_progress_bar=False)
        centroid = all_embeddings.mean(axis=0).reshape(1, -1)
        centrality_scores = cosine_similarity(candidate_embeddings, centroid).flatten()
        centrality_norm = (centrality_scores - centrality_scores.min()) / (centrality_scores.max() - centrality_scores.min() + 1e-8)

        # --- Position bias ---
        position_scores = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            score = 0.0
            if c['is_first_in_segment']:
                score += 0.10
            if c['is_last_in_segment']:
                score += 0.05
            if c['position_in_segment'] < 0.2:
                score += 0.03
            position_scores[i] = min(score, 0.15)

        # --- Discourse markers ---
        discourse_scores = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            score = 0.0
            for pattern, weight in ALL_DISCOURSE_MARKERS.items():
                if re.search(pattern, c['text'], re.IGNORECASE):
                    score += weight
            discourse_scores[i] = min(score, 0.25)

        # --- Specificity: penalize filler-heavy, reward content-dense ---
        specificity_scores = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            filler_ratio = c.get('filler_ratio', 0.0)
            words = c['text'].split()
            content_words = [w for w in words[1:] if w[0].isupper() and len(w) > 2] if len(words) > 1 else []
            noun_density = len(content_words) / max(1, len(words))
            has_numbers = 1.0 if re.search(r'\d{2,}', c['text']) else 0.0
            specificity_scores[i] = (
                0.4 * (1.0 - filler_ratio) +
                0.4 * min(noun_density * 3, 1.0) +
                0.2 * has_numbers
            )
        specificity_norm = (specificity_scores - specificity_scores.min()) / (specificity_scores.max() - specificity_scores.min() + 1e-8)

        # --- Rhetoric penalty: penalize aspirational/emotional, reward factual ---
        rhetoric_scores = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            score = 0.0
            for pattern, weight in RHETORIC_MARKERS.items():
                if re.search(pattern, c['text'], re.IGNORECASE):
                    score += weight
            for pattern, weight in VAGUE_PATTERNS.items():
                if re.search(pattern, c['text'], re.IGNORECASE):
                    score += weight
            rhetoric_scores[i] = min(score, 0.60)
        rhetoric_norm = 1.0 - (rhetoric_scores / 0.60)

        # --- Combined score ---
        combined = (
            0.20 * bm25_norm +
            0.25 * centrality_norm +
            0.10 * position_scores +
            0.15 * discourse_scores +
            0.15 * specificity_norm +
            0.15 * rhetoric_norm
        )

        for i, c in enumerate(candidates):
            c['bm25_score'] = float(bm25_norm[i])
            c['centrality_score'] = float(centrality_norm[i])
            c['position_score'] = float(position_scores[i])
            c['discourse_score'] = float(discourse_scores[i])
            c['specificity_score'] = float(specificity_norm[i])
            c['rhetoric_score'] = float(rhetoric_norm[i])
            c['combined_score'] = float(combined[i])

        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates

    def _bm25_tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _sigmoid_normalize(self, scores):
        if len(scores) == 0:
            return scores
        mean = np.mean(scores)
        std = np.std(scores) + 1e-8
        z = (scores - mean) / std
        return 1 / (1 + np.exp(-z))

    # ─── Stage 2: MMR Selection ────────────────────────────────────────

    _ACTOR_PATTERNS = re.compile(
        r'\b(CDU|SPD|FDP|AfD|BSW|Gr[üu]ne[n]?|Linke|PVP|Team Zastrow'
        r'|Oberb[üu]rgermeister|Verwaltung|Globus|Freistaat|Stadtrat'
        r'|Fraktion|Verein|F[öo]rderkreis|Kulturausschuss)\b',
        re.IGNORECASE
    )

    def _extract_actors(self, text):
        return {m.group(0).lower() for m in self._ACTOR_PATTERNS.finditer(text)}

    def _extract_numbers(self, text):
        return set(re.findall(r'\d[\d.,]*', text))

    def select_with_mmr(self, scored_candidates, target_count):
        if not scored_candidates:
            return []
        if len(scored_candidates) <= target_count:
            return scored_candidates

        texts = [c['text'] for c in scored_candidates]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        candidate_actors = [self._extract_actors(t) for t in texts]
        candidate_numbers = [self._extract_numbers(t) for t in texts]

        selected_indices = [0]
        remaining = list(range(1, len(scored_candidates)))

        while len(selected_indices) < target_count and remaining:
            best_idx = None
            best_mmr = -float('inf')

            selected_embeddings = embeddings[selected_indices]
            selected_speakers = {scored_candidates[i]['speaker'] for i in selected_indices}
            selected_actors = set()
            selected_numbers = set()
            for i in selected_indices:
                selected_actors.update(candidate_actors[i])
                selected_numbers.update(candidate_numbers[i])

            for idx in remaining:
                relevance = scored_candidates[idx]['combined_score']
                sim_to_selected = cosine_similarity(
                    embeddings[idx].reshape(1, -1), selected_embeddings
                ).max()

                speaker_bonus = 0.08 if scored_candidates[idx]['speaker'] not in selected_speakers else 0.0

                new_actors = candidate_actors[idx] - selected_actors
                actor_bonus = min(0.06 * len(new_actors), 0.12)

                has_new_number = bool(candidate_numbers[idx] - selected_numbers)
                number_bonus = 0.05 if has_new_number else 0.0

                mmr = (self.mmr_lambda * relevance
                       - (1 - self.mmr_lambda) * sim_to_selected
                       + speaker_bonus + actor_bonus + number_bonus)

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return [scored_candidates[i] for i in selected_indices]

    # ─── Stage 2: LLM Curation (selection only) ─────────────────────────

    def curate_with_llm(self, selected_candidates, target_count=None):
        if not selected_candidates:
            return []

        if target_count is None:
            target_count = len(selected_candidates)

        if len(selected_candidates) <= target_count:
            return selected_candidates

        items_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(selected_candidates)])

        prompt = f"""Du erhältst {len(selected_candidates)} nummerierte Sätze aus einem deutschen Sitzungstranskript.
Wähle die {target_count} informativsten davon aus.

Sätze:
{items_text}

Auswahlkriterien (Priorität):
1. Konkrete Entscheidungen, Beschlüsse, Abstimmungsergebnisse
2. Spezifische Fakten mit Zahlen, Daten, Namen, Geldbeträgen
3. Positionen verschiedener Fraktionen/Akteure (wer fordert/blockiert/unterstützt was)
4. Konflikte und Meinungsverschiedenheiten zwischen Akteuren

Verwirf:
- Rein rhetorische Appelle ohne konkreten Inhalt
- Wiederholungen bereits abgedeckter Punkte
- Prozedurale Aussagen (Begrüßungen, Tagesordnung, Redeübergaben)

Antworte NUR mit den Nummern der ausgewählten Sätze, kommagetrennt. Beispiel: 1,3,5,7,9
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du wählst die informativsten Sätze aus. Antworte nur mit Nummern."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=100)

        if not content:
            return selected_candidates[:target_count]

        try:
            indices = [int(x.strip()) - 1 for x in re.findall(r'\d+', content)]
            indices = [i for i in indices if 0 <= i < len(selected_candidates)]
            if len(indices) >= target_count // 2:
                return [selected_candidates[i] for i in indices[:target_count]]
        except (ValueError, IndexError):
            pass

        return selected_candidates[:target_count]

    # ─── Stage 2: Rule-based Cleanup ──────────────────────────────────

    def curate_informative_with_llm(self, selected_candidates, target_count=None):
        if not selected_candidates:
            return []

        if target_count is None:
            target_count = len(selected_candidates)

        items_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(selected_candidates)])

        prompt = f"""Du erhältst {len(selected_candidates)} nummerierte Sätze aus einem deutschen Transkript.

Wähle höchstens {target_count} Sätze aus, die zugleich:
- konkrete, eigenstÃ¤ndige Sachaussagen enthalten
- für eine knappe Zusammenfassung des Transkripts relevant wÃ¤ren
- als Key Points verwendbar sind

Sätze:
{items_text}

Priorität:
1. Entscheidungen, Beschlüsse, Abstimmungsergebnisse
2. Konkrete nÃ¤chste Schritte, Aufgaben oder Ziele
3. Spezifische Fakten mit Zahlen, Daten, Namen oder GeldbetrÃ¤gen
4. Positionen verschiedener Akteure
5. Technische Planungsinhalte, offene Kernfragen oder begrÃ¼ndete VorschlÃ¤ge

Verwirf:
- Begrüßungen, Redeübergaben und reine Moderation
- Fragen ohne inhaltliche Aussage
- persönliche Kommentare, Anekdoten und rhetorische Ausschmückungen
- vage Zustimmung wie "passt", "okay", "genau"
- Fragmente, die ohne Kontext nicht verständlich sind
- Wiederholungen bereits ausgewählter Punkte

Wichtig:
- Bewahre Unsicherheit: Wenn ein Satz nur einen Vorschlag oder eine offene Frage enthält, darf er ausgewählt werden, aber nur wenn er inhaltlich wichtig ist.
- Schreibe keine Sätze um.
- Antworte NUR mit den Nummern der ausgewählten Sätze, kommagetrennt. Beispiel: 1,3,5,7
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du wählst konkrete, informative und zusammenfassungswürdige Transkriptsätze aus. Antworte nur mit Nummern."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=100)

        if not content:
            return selected_candidates[:target_count]

        try:
            seen = set()
            indices = []
            for value in re.findall(r'\d+', content):
                idx = int(value.strip()) - 1
                if 0 <= idx < len(selected_candidates) and idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
            if indices:
                return [selected_candidates[i] for i in indices[:target_count]]
        except (ValueError, IndexError):
            pass

        return selected_candidates[:target_count]

    _ORAL_PREFIX = re.compile(
        r'^(Also,?\s*|Und\s+|Na gut,?\s*|Ja,?\s*|Genau,?\s*|Gut,?\s*'
        r'|Okay,?\s*|Naja,?\s*|Nun,?\s*|Ähm,?\s*|Also ich denke,?\s*'
        r'|Ich denke,?\s*|Ich sage mal,?\s*|Ich würde sagen,?\s*)',
        re.IGNORECASE
    )

    _ORAL_SUFFIX = re.compile(
        r'(,?\s*(oder so|und so weiter|und und und|sage ich mal|'
        r'würde ich sagen|muss man sagen|das muss man sagen)\.?\s*)$',
        re.IGNORECASE
    )

    def _clean_sentence(self, text):
        text = self._ORAL_PREFIX.sub('', text).strip()
        text = self._ORAL_SUFFIX.sub('.', text).strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    # ─── Main Pipeline ─────────────────────────────────────────────────

    def extract_key_points(self, transcript_text, filename=None):
        print(f"  [Stage 1] Parsing transcript...")
        turns = self.parse_transcript_text(transcript_text)
        if not turns:
            return {'key_points': [], 'metadata': {}}

        print(f"    - {len(turns)} turns parsed")

        print(f"  [Stage 1] Topic segmentation...")
        segments = self.run_topic_segmentation(turns)
        print(f"    - {len(segments)} segments created")

        print(f"  [Stage 2A] Generating sentence candidates...")
        candidates = self.segment_into_sentences(segments)
        print(f"    - {len(candidates)} candidate sentences")

        if not candidates:
            return {'key_points': [], 'metadata': {'num_turns': len(turns), 'num_segments': len(segments)}}

        print(f"  [Stage 2B] Scoring salience (BM25 + centrality + position + discourse + specificity)...")
        scored = self.score_salience(candidates, transcript_text)

        pre_gate = len(scored)
        scored = [c for c in scored if not (
            c['specificity_score'] < 0.3
            and c['discourse_score'] == 0
            and not re.search(r'\d{2,}', c['text'])
        )]
        if pre_gate != len(scored):
            print(f"    - Information gate removed {pre_gate - len(scored)} low-content candidates")

        word_count = len(transcript_text.split())
        min_count = max(3, min(8, word_count // 200))
        target_count = max(min_count, min(25, int(word_count * self.target_percentage / 25)))
        print(f"    - Target: {target_count} key points (from {word_count} words)")

        overselect_count = min(len(scored), int(target_count * 1.5))
        print(f"  [Stage 2C] MMR selection (lambda={self.mmr_lambda}, overselect={overselect_count})...")
        selected = self.select_with_mmr(scored, overselect_count)
        print(f"    - Selected {len(selected)} points")

        if self.llm_curation_mode == "curate_informative":
            print(f"  [Stage 2D] Combined LLM curation + informative filter (picking up to {target_count} from {len(selected)})...")
            curated = self.curate_informative_with_llm(selected, target_count=target_count)
            print(f"    - Combined LLM selected {len(curated)} points")
        elif self.llm_curation_mode == "curate":
            print(f"  [Stage 2D] LLM curation (picking {target_count} from {len(selected)})...")
            curated = self.curate_with_llm(selected, target_count=target_count)
            print(f"    - LLM selected {len(curated)} points")
        else:
            curated = sorted(selected, key=lambda x: x['combined_score'], reverse=True)[:target_count]

        print(f"  [Stage 2E] Rule-based cleanup...")
        key_points = [self._clean_sentence(c['text']) for c in curated]

        metadata = {
            'num_turns': len(turns),
            'num_segments': len(segments),
            'num_candidates': len(candidates),
            'target_count': target_count,
            'word_count': word_count,
            'mmr_lambda': self.mmr_lambda,
            'seg_threshold': self.seg_threshold,
            'llm_curation_mode': self.llm_curation_mode,
            'scored_selections': [
                {
                    'text': c['text'],
                    'combined_score': round(c['combined_score'], 4),
                    'bm25_score': round(c['bm25_score'], 4),
                    'centrality_score': round(c['centrality_score'], 4),
                    'position_score': round(c['position_score'], 4),
                    'discourse_score': round(c['discourse_score'], 4),
                    'specificity_score': round(c.get('specificity_score', 0.0), 4),
                    'rhetoric_score': round(c.get('rhetoric_score', 1.0), 4),
                    'segment_id': c['segment_id'],
                    'speaker': c['speaker'],
                }
                for c in selected
            ],
        }

        return {'key_points': key_points, 'metadata': metadata}


def process_csv(csv_path, extractor, output_dir, limit=None, filenames=None, force=False):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"statistical_kp_{base_name}.json")

    results = []
    processed_ids = set()
    if os.path.exists(output_path) and not force:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {f"{r['filename']}_{r.get('profile', '')}_{r.get('model', '')}" for r in results}
                print(f"Loaded {len(results)} existing results")
        except Exception:
            pass

    try:
        df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if filenames:
        df = df[df['filename'].isin(filenames)]
        if force:
            filenames_set = set(filenames)
            results = [r for r in results if r.get('filename') not in filenames_set]
            processed_ids = {f"{r['filename']}_{r.get('profile', '')}_{r.get('model', '')}" for r in results}

    mask = df.apply(
        lambda r: f"{r['filename']}_{r.get('profile', '')}_{r.get('model', '')}" not in processed_ids, axis=1
    )
    df_to_process = df[mask]

    if limit:
        df_to_process = df_to_process.head(limit)

    if len(df_to_process) == 0:
        print(f"No new rows to process in {base_name}.")
        return

    print(f"\nProcessing {len(df_to_process)} rows from {base_name}...")

    keypoints_cache = {}

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        filename = row['filename']
        profile = row.get('profile', '')
        model = row.get('model', '')
        transcript = str(row['transcript'])
        summary = str(row.get('summary', ''))

        print(f"\n{'='*60}")
        print(f"Extracting key points: {filename} ({model}, {profile})")
        print(f"{'='*60}")

        if filename in keypoints_cache:
            print(f"  [Cache] Reusing key points from previous extraction of {filename}")
            extraction = keypoints_cache[filename]
        else:
            extraction = extractor.extract_key_points(transcript, filename)
            keypoints_cache[filename] = extraction

        result = dict(extraction)
        result['filename'] = filename
        result['profile'] = profile
        result['model'] = model
        result['summary'] = summary[:200] + "..." if len(summary) > 200 else summary

        results.append(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical key-point extraction from transcripts (Stages 1 & 2)")
    parser.add_argument("--csv", type=str, default="pilot_summaries.csv", help="Path to CSV file")
    parser.add_argument("--output-dir", type=str, default="statistical_keypoints_results", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit rows to process")
    parser.add_argument("--filenames", type=str, nargs="+", help="Specific filenames to process")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM compression step")
    parser.add_argument(
        "--llm-curation-mode",
        choices=["none", "curate", "curate_informative"],
        help="LLM selection mode: none, current curation, or combined curation+informative filtering",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR lambda (0=diversity, 1=relevance)")
    parser.add_argument("--target-pct", type=float, default=0.05, help="Target percentage for key points")
    parser.add_argument("--seg-threshold", type=float, default=0.3, help="Segmentation distance threshold")
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-large", help="Embedding model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = StatisticalKeyPointExtractor(
        embedding_model_name=args.embedding_model,
        use_llm_compression=not args.no_llm,
        llm_curation_mode=args.llm_curation_mode,
        mmr_lambda=args.mmr_lambda,
        target_percentage=args.target_pct,
        seg_threshold=args.seg_threshold,
    )

    import atexit
    def print_usage():
        pt = extractor.total_prompt_tokens
        ct = extractor.total_completion_tokens
        if pt + ct > 0:
            print(f"\n=== Token Usage ===")
            print(f"  Prompt tokens:     {pt:>8}")
            print(f"  Completion tokens: {ct:>8}")
            print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    process_csv(
        args.csv, extractor, args.output_dir,
        limit=args.limit, filenames=args.filenames, force=args.force,
    )
