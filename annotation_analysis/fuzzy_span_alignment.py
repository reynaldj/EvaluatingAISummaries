"""Fuzzy source localization, independent of human spans (no LLM or exact-match gate)."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import LCSseq


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class SourceText:
    text: str

    def __post_init__(self):
        self.tokens = list(TOKEN_RE.finditer(self.text))
        self.starts = [m.start() for m in self.tokens]
        self.ends = [m.end() for m in self.tokens]
        words = [m.group().casefold() for m in self.tokens]
        self.normalized = " ".join(words)
        self.normalized_starts = []
        self.normalized_ends = []
        offset = 0
        for word in words:
            self.normalized_starts.append(offset)
            offset += len(word)
            self.normalized_ends.append(offset)
            offset += 1

    def token_positions(self, spans):
        """Any positive character overlap selects a source token; union avoids double counting."""
        selected = set()
        for start, end in spans:
            if end > start:
                selected.update(range(bisect_right(self.ends, start), bisect_left(self.starts, end)))
        return selected


def fuzzy_candidates(query, source, *, window_factor=3, stride_factor=0.5):
    """Generate approximate substring alignments in overlapping token windows.

    RapidFuzz searches character sequences; final boundaries snap outwards to source
    word boundaries. Every item takes this path, including verbatim items. Casefolding
    and punctuation removal are just the fuzzy representation, never an exact gate.
    Search is windowed/heuristic, not an exhaustive guarantee of the global optimum.
    """
    query_words = [m.group().casefold() for m in TOKEN_RE.finditer(query)]
    if not query_words or not source.tokens:
        return []
    needle = " ".join(query_words)
    query_chars = "".join(query_words)
    width = max(12, window_factor * len(query_words))
    stride = max(1, int(len(query_words) * stride_factor))
    candidates = {}
    for token_start in range(0, len(source.tokens), stride):
        token_end = min(len(source.tokens), token_start + width)
        window_start = source.normalized_starts[token_start]
        window_end = source.normalized_ends[token_end - 1]
        alignment = fuzz.partial_ratio_alignment(needle, source.normalized[window_start:window_end])
        if alignment is None or alignment.dest_start == alignment.dest_end:
            continue
        left = window_start + alignment.dest_start
        right = window_start + alignment.dest_end
        first = bisect_right(source.normalized_ends, left)
        last = bisect_left(source.normalized_starts, right)
        if first >= last:
            continue
        start, end = source.starts[first], source.ends[last - 1]
        if (start, end) in candidates:
            continue
        matched = source.normalized[source.normalized_starts[first]:source.normalized_ends[last - 1]]
        # Rescore the ENTIRE query, not just RapidFuzz's possibly short query fragment.
        similarity = fuzz.ratio(needle, matched)
        # Explicit query-side lexical coverage: in-order shared non-space characters.
        # This is not semantic coverage, nor a calibrated probability.
        coverage = LCSseq.similarity(query_chars, matched.replace(" ", "")) / len(query_chars)
        candidates[start, end] = {
            "start": start, "end": end, "quote": source.text[start:end],
            "similarity": similarity, "query_character_coverage": coverage,
        }
    # Collapse alternate boundaries for the same occurrence, retaining the best score.
    distinct = []
    for candidate in sorted(candidates.values(), key=lambda c: (-c["similarity"], -c["query_character_coverage"], c["start"])):
        if any(same_occurrence(candidate, previous) for previous in distinct):
            continue
        distinct.append(candidate)
    return distinct


def same_occurrence(left, right):
    overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    return overlap / min(left["end"] - left["start"], right["end"] - right["start"]) >= 0.5


def localize(query, source, *, min_similarity=65.0, min_coverage=0.65, ambiguity_margin=3.0,
             ambiguity_policy="best"):
    """Retain the highest-scoring qualifying occurrence, without human input.

    The all/exclude policies remain available to reproduce earlier runs.
    Each occurrence is a separate span; text between occurrences is not selected.
    """
    if ambiguity_policy not in ("all", "exclude", "best"):
        raise ValueError("Unknown ambiguity policy")
    candidates = fuzzy_candidates(query, source)
    eligible = [c for c in candidates if c["similarity"] >= min_similarity
                and c["query_character_coverage"] >= min_coverage]
    if not candidates:
        return {"status": "empty_query_or_source", "candidate": None, "alternatives": [], "selected_spans": []}
    if not eligible:
        return {"status": "below_threshold", "candidate": candidates[0], "alternatives": candidates[1:4], "selected_spans": []}
    best = eligible[0]
    close = [c for c in eligible[1:] if best["similarity"] - c["similarity"] <= ambiguity_margin]
    return {
        "status": ("multiple_locations" if ambiguity_policy == "all" else
                   "ambiguous" if ambiguity_policy == "exclude" else "located") if close else "located", "candidate": best,
        "alternatives": close if close else eligible[1:4],
        "selected_spans": ([best, *close] if ambiguity_policy == "all" else
                           [] if close and ambiguity_policy == "exclude" else [best]),
    }


def positional_scores(automatic, human):
    overlap = len(automatic & human)
    a, h = len(automatic), len(human)
    return {
        "automatic_tokens": a, "human_tokens": h, "overlap_tokens": overlap,
        "precision": overlap / a if a else None,
        "recall": overlap / h if h else None,
        "f1": 2 * overlap / (a + h) if a + h else None,
    }


def validate_human_span(text, start, end, quote):
    """Validate Python offsets or losslessly convert verified UTF-16 offsets."""
    start, end = int(start), int(end)
    if 0 <= start < end <= len(text) and text[start:end] == quote:
        return start, end, "valid"
    units, boundaries = 0, {0: 0}
    for index, character in enumerate(text):
        units += 2 if ord(character) > 0xFFFF else 1
        boundaries[units] = index + 1
    if start in boundaries and end in boundaries:
        left, right = boundaries[start], boundaries[end]
        if left < right and text[left:right] == quote:
            return left, right, "converted_utf16"
    return None, None, "invalid_offsets_or_text"
