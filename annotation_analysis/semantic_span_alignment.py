"""Embedding-based source localization. No human annotations enter this module."""
from __future__ import annotations

import re

import numpy as np

from fuzzy_span_alignment import SourceText, same_occurrence


DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def model_settings(model_name):
    """E5 retrieval needs role prefixes and its own exploratory score threshold."""
    if "multilingual-e5" in str(model_name).lower():
        return {"query_prefix": "query: ", "passage_prefix": "passage: ", "min_similarity": 0.85}
    return {"query_prefix": "", "passage_prefix": "", "min_similarity": 0.75}


def passage_spans(source):
    """Heuristic sentences, clauses, and adjacent sentence pairs at source offsets.

    Boundaries are source word boundaries, not predictions made by the embedding
    model. Sentence splitting is deliberately lightweight (not a German parser).
    """
    if not source.tokens:
        return []
    sentences, spans = [], set()

    def add(left, right):
        positions = sorted(source.token_positions([(left, right)]))
        if not positions:
            return None
        span = (source.starts[positions[0]], source.ends[positions[-1]])
        spans.add(span)
        return span

    start = 0
    for boundary in [*re.finditer(r"(?<=[.!?])\s+|\n+", source.text), None]:
        end = boundary.start() if boundary else len(source.text)
        sentence = add(start, end)
        if sentence:
            sentences.append(sentence)
            clause_start = start
            for separator in re.finditer(r"[,;:]|\s[–—]\s", source.text[start:end]):
                clause_end = start + separator.start()
                # Very short fragments remain available through their sentence.
                if len(source.token_positions([(clause_start, clause_end)])) >= 3:
                    add(clause_start, clause_end)
                clause_start = start + separator.end()
            if len(source.token_positions([(clause_start, end)])) >= 3:
                add(clause_start, end)
        start = boundary.end() if boundary else end
    for left, right in zip(sentences, sentences[1:]):
        add(left[0], right[1])
    return sorted(spans)


def select_candidates(candidates, min_similarity, margin, location_policy="best"):
    """Keep one best qualifying occurrence; all is a historical opt-in policy."""
    if location_policy not in ("best", "all"):
        raise ValueError("location_policy must be best or all")
    distinct = []
    for candidate in sorted(candidates, key=lambda c: (
            -c["semantic_similarity"], c["end"] - c["start"], c["start"])):
        if not any(same_occurrence(candidate, previous) for previous in distinct):
            distinct.append(candidate)
    eligible = [c for c in distinct if c["semantic_similarity"] >= min_similarity]
    if not eligible:
        return {"status": "semantic_below_threshold", "candidate": distinct[0] if distinct else None,
                "alternatives": distinct[1:4], "selected_spans": []}
    best = eligible[0]
    selected = [c for c in eligible
                if best["semantic_similarity"] - c["semantic_similarity"] <= margin + 1e-12]
    if location_policy == "best":
        selected = [best]
    return {"status": "multiple_locations" if len(selected) > 1 else "located",
            "candidate": best, "alternatives": selected[1:] or eligible[1:4],
            "selected_spans": selected}


class SemanticLocator:
    """Batch cosine search with whole-word offsets and no silent input truncation."""

    def __init__(self, model, *, min_similarity=0.75, margin=0.03, batch_size=32,
                 query_prefix="", passage_prefix="", location_policy="best"):
        if not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("Semantic similarity threshold must be between -1 and 1")
        if not np.isfinite(margin) or not 0 <= margin <= 2:
            raise ValueError("Semantic margin must be between 0 and 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if location_policy not in ("best", "all"):
            raise ValueError("location_policy must be best or all")
        self.location_policy = location_policy
        self.model = model
        self.min_similarity = min_similarity
        self.margin = margin
        self.batch_size = batch_size
        self.max_tokens = int(model.max_seq_length)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def length(self, text):
        return len(self.model.tokenizer(text, add_special_tokens=True,
                                        truncation=False, verbose=False)["input_ids"])

    def fitted_spans(self, source):
        """Split overlength passages into overlapping word windows, never truncate."""
        fitted, seen = set(), set()
        pending = list(passage_spans(source))
        while pending:
            start, end = pending.pop()
            if (start, end) in seen:
                continue
            seen.add((start, end))
            if self.length(self.passage_prefix + source.text[start:end]) <= self.max_tokens:
                fitted.add((start, end))
                continue
            words = sorted(source.token_positions([(start, end)]))
            if len(words) < 2:
                continue
            width = max(1, (len(words) + 1) // 2)
            stride = max(1, width // 2)
            for i in range(0, len(words), stride):
                part = words[i:i + width]
                pending.append((source.starts[part[0]], source.ends[part[-1]]))
        return sorted(fitted)

    def encode(self, texts, prefix=""):
        vectors = np.asarray(self.model.encode(
            [prefix + text for text in texts], prompt="", batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned invalid vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Embedding model returned a zero vector")
        return vectors / norms

    def localize_many(self, queries, source):
        """Encode each source candidate once for all rejected items in its document."""
        results, valid = {}, []
        for query in dict.fromkeys(queries):
            if not SourceText(query).tokens or not source.tokens:
                status = "empty_query_or_source"
            elif self.length(self.query_prefix + query) > self.max_tokens:
                status = "semantic_query_too_long"
            else:
                valid.append(query)
                continue
            results[query] = {"status": status, "candidate": None,
                              "alternatives": [], "selected_spans": []}
        spans = self.fitted_spans(source) if valid else []
        if not spans:
            for query in valid:
                results[query] = {"status": "semantic_no_candidates", "candidate": None,
                                  "alternatives": [], "selected_spans": []}
            return results
        candidates = [{"start": start, "end": end, "quote": source.text[start:end]}
                      for start, end in spans]
        passage_vectors = self.encode([c["quote"] for c in candidates], self.passage_prefix)
        query_vectors = self.encode(valid, self.query_prefix)
        scores = np.clip(query_vectors @ passage_vectors.T, -1, 1)
        for query, similarities in zip(valid, scores):
            scored = [c | {"semantic_similarity": float(score)}
                      for c, score in zip(candidates, similarities)]
            results[query] = select_candidates(scored, self.min_similarity, self.margin, self.location_policy)
        return results
