import pandas as pd
import numpy as np
import re
import os
import json
import tiktoken
import time
import random
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import csv
from extract_keypoints_entailment import EntailmentKeyPointExtractor


class NLIHybridRelevanceEvaluator:
    """
    NLI-Hybrid evaluation pipeline (Stages 3-7):
    - Stage 3: Summary segmentation
    - Stage 4: Embedding-based retrieval
    - Stage 5: NLI verification
    - Stage 6: LLM refinement for neutral cases
    - Stage 7: Scoring and aggregation
    """

    def __init__(
        self,
        model_name="meta-llama-3.1-8b-instruct",
        nli_model_name="joeddav/xlm-roberta-large-xnli",
        top_k=3,
        entailment_threshold=0.7,
        contradiction_threshold=0.7,
        batch_size=16
    ):
        # LLM setup
        self.api_key = os.environ.get("ACADEMIC_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("Set the ACADEMIC_CLOUD_API_KEY environment variable before running this evaluator.")
        self.base_url = "https://chat-ai.academiccloud.de/v1"
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Embedding model
        print("Loading SentenceTransformer model...")
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        except Exception as exc:
            print(f"  [Warning] Failed to load intfloat/multilingual-e5-large ({exc}); falling back to all-MiniLM-L6-v2.")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # NLI model
        print(f"Loading NLI model: {nli_model_name}...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_model.to(self.device)
        print(f"  NLI model loaded on: {self.device}")

        # Label mapping: 0=contradiction, 1=neutral, 2=entailment
        self.nli_labels = ["contradiction", "neutral", "entailment"]

        # Configuration
        self.top_k = top_k
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.batch_size = batch_size

        # Cache
        self.kp_cache = {}
        self.llm_refinement_cache = {}

        # Key point extractor (lazy initialization)
        self._kp_extractor = None

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        """Call the API with retry logic."""
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
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

    def count_tokens(self, text):
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    # ─── Stage 3: Summary Segmentation ─────────────────────────────────────

    def segment_summary(self, summary):
        """
        Split summary into sentences, with sub-splitting for long sentences.
        Returns list of summary sentence segments.
        """
        if not summary or not summary.strip():
            return []

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

        segments = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            words = sent.split()
            if len(words) > 40:
                # Sub-split long sentences at punctuation or conjunctions
                sub_parts = re.split(r'(?:;|\s–\s|\s-\s|,\s+(?:und|aber|oder|denn|weil|da|sodass)\s+)', sent)
                for part in sub_parts:
                    part = part.strip()
                    if len(part) >= 10:
                        segments.append(part)
            else:
                segments.append(sent)

        return segments

    # ─── Stage 4: Retrieval ────────────────────────────────────────────────

    def retrieve_candidates(self, key_points, summary_sentences, top_k=None):
        """
        Retrieve candidate pairs using embedding similarity.
        Returns list of (kp_idx, sent_idx, similarity) tuples.
        """
        if top_k is None:
            top_k = self.top_k

        if not key_points or not summary_sentences:
            return []

        # Encode with E5 prefixes
        kp_texts = ["query: " + kp for kp in key_points]
        sent_texts = ["passage: " + sent for sent in summary_sentences]

        try:
            kp_embeddings = self.embedding_model.encode(kp_texts)
            sent_embeddings = self.embedding_model.encode(sent_texts)
        except Exception as e:
            print(f"    [Warning] Embedding failed: {e}")
            return []

        # Compute similarity matrix
        sim_matrix = cosine_similarity(kp_embeddings, sent_embeddings)

        # For each key point, retrieve top-k sentences
        candidate_pairs = []
        for kp_idx in range(len(key_points)):
            sims = sim_matrix[kp_idx]
            # Get top-k indices above threshold
            top_indices = np.argsort(sims)[::-1][:top_k]
            for sent_idx in top_indices:
                sim_score = float(sims[sent_idx])
                if sim_score > 0.3:  # Threshold to avoid noise
                    candidate_pairs.append((kp_idx, sent_idx, sim_score))

        return candidate_pairs

    # ─── Stage 5: NLI Verification ─────────────────────────────────────────

    def run_nli_batch(self, premises, hypotheses):
        """
        Run NLI model on batches of (premise, hypothesis) pairs.
        Returns list of dicts with label, scores, confidence.
        """
        if not premises or not hypotheses or len(premises) != len(hypotheses):
            return []

        results = []
        num_pairs = len(premises)

        with torch.no_grad():
            for batch_start in range(0, num_pairs, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_pairs)
                batch_premises = premises[batch_start:batch_end]
                batch_hypotheses = hypotheses[batch_start:batch_end]

                # Tokenize
                inputs = self.nli_tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass
                outputs = self.nli_model(**inputs)
                logits = outputs.logits

                # Softmax to get probabilities
                probs = F.softmax(logits, dim=-1).cpu().numpy()

                # Parse results
                for prob_dist in probs:
                    label_idx = int(np.argmax(prob_dist))
                    label = self.nli_labels[label_idx]
                    confidence = float(prob_dist[label_idx])

                    # Apply thresholds
                    entailment_score = float(prob_dist[2])
                    contradiction_score = float(prob_dist[0])

                    if entailment_score >= self.entailment_threshold:
                        final_label = "entailment"
                    elif contradiction_score >= self.contradiction_threshold:
                        final_label = "contradiction"
                    else:
                        final_label = "neutral"

                    results.append({
                        "label": final_label,
                        "scores": {
                            "contradiction": float(prob_dist[0]),
                            "neutral": float(prob_dist[1]),
                            "entailment": float(prob_dist[2])
                        },
                        "confidence": confidence
                    })

        return results

    def verify_pairs_nli(self, key_points, summary_sentences, candidate_pairs):
        """
        Run NLI on all candidate pairs.
        Returns list of dicts with detailed results.
        """
        if not candidate_pairs:
            return []

        # Prepare batches
        premises = []
        hypotheses = []
        pair_info = []

        for kp_idx, sent_idx, sim_score in candidate_pairs:
            premises.append(summary_sentences[sent_idx])
            hypotheses.append(key_points[kp_idx])
            pair_info.append((kp_idx, sent_idx, sim_score))

        # Run NLI
        nli_results = self.run_nli_batch(premises, hypotheses)

        # Combine with pair info
        detailed_results = []
        for (kp_idx, sent_idx, sim_score), nli_result in zip(pair_info, nli_results):
            detailed_results.append({
                "kp_idx": kp_idx,
                "sent_idx": int(sent_idx),
                "key_point": key_points[kp_idx],
                "summary_sentence": summary_sentences[sent_idx],
                "cosine_similarity": sim_score,
                "nli_label": nli_result["label"],
                "nli_scores": nli_result["scores"],
                "nli_confidence": nli_result["confidence"]
            })

        return detailed_results

    # ─── Stage 6: LLM Refinement ───────────────────────────────────────────

    def refine_neutral_with_llm(self, key_point, summary_sentence):
        """
        For NLI-neutral cases, ask LLM to decide: partial | missing | unclear.
        Returns dict with label, confidence, reason.
        """
        cache_key = f"{key_point}|||{summary_sentence}"
        if cache_key in self.llm_refinement_cache:
            return self.llm_refinement_cache[cache_key]

        prompt = f"""Du erhältst einen Kernpunkt aus einem Transkript und einen Satz aus einer Zusammenfassung.
Das NLI-Modell konnte keine klare Entailment- oder Widerspruchs-Beziehung feststellen.

Kernpunkt:
{key_point}

Zusammenfassungs-Satz:
{summary_sentence}

Aufgabe:
Beurteile, ob der Zusammenfassungs-Satz den Kernpunkt teilweise abdeckt oder ob er ihn gänzlich verfehlt.

Antwortmöglichkeiten:
- "partial": Der Satz deckt einen wesentlichen Teil des Kernpunkts ab, aber nicht vollständig.
- "missing": Der Satz hat keinen inhaltlichen Bezug zum Kernpunkt oder deckt ihn nicht substantiv ab.
- "unclear": Die Beziehung ist nicht eindeutig bestimmbar.

Antworte ausschließlich im JSON-Format:
{{"label": "partial|missing|unclear", "confidence": 0.0-1.0, "reason": "kurze Begründung"}}
"""

        try:
            content = self._safe_api_call([
                {"role": "system", "content": "Du bist ein präziser Textabgleich-Assistent. Antworte nur mit JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, max_tokens=150)
        except Exception as e:
            print(f"    [LLM Refinement Error] {e}")
            return {"label": "unclear", "confidence": 0.0, "reason": f"API error: {e}"}

        # Parse JSON
        try:
            result = json.loads(content)
            if "label" not in result:
                raise ValueError("Missing 'label' field")
        except Exception:
            # Fallback: regex extraction
            label_match = re.search(r'"label"\s*:\s*"(partial|missing|unclear)"', content, re.IGNORECASE)
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)

            result = {
                "label": label_match.group(1).lower() if label_match else "unclear",
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "reason": reason_match.group(1) if reason_match else content[:100]
            }

        self.llm_refinement_cache[cache_key] = result
        return result

    # ─── Stage 7: Aggregation ──────────────────────────────────────────────

    def aggregate_per_keypoint(self, key_points, nli_results, llm_refinements):
        """
        Aggregate results per key point and compute coverage score.
        Returns dict with per-key-point details and overall metrics.
        """
        # Group results by key point
        kp_matches = {i: [] for i in range(len(key_points))}
        for result in nli_results:
            kp_idx = result["kp_idx"]
            kp_matches[kp_idx].append(result)

        per_key_point = []
        covered_count = 0
        partially_covered_count = 0
        missing_count = 0
        contradicted_count = 0
        excluded_count = 0
        earned_score = 0.0

        for kp_idx, key_point in enumerate(key_points):
            matches = kp_matches[kp_idx]

            if not matches:
                # No matches found
                per_key_point.append({
                    "key_point": key_point,
                    "final_label": "missing",
                    "final_score": 0.0,
                    "best_match": None,
                    "all_matches": []
                })
                missing_count += 1
                continue

            # Aggregation logic
            best_label = None
            best_score = -999
            best_match = None

            for match in matches:
                nli_label = match["nli_label"]

                if nli_label == "entailment":
                    label = "covered"
                    score = 1.0
                elif nli_label == "contradiction":
                    label = "contradicted"
                    score = -1.0
                else:  # neutral
                    # Get LLM refinement
                    cache_key = f"{match['key_point']}|||{match['summary_sentence']}"
                    llm_result = llm_refinements.get(cache_key, {"label": "unclear"})

                    if llm_result["label"] == "partial":
                        label = "partially_covered"
                        score = 0.5
                    elif llm_result["label"] == "missing":
                        label = "missing"
                        score = 0.0
                    else:  # unclear
                        label = "unclear"
                        score = None

                    match["llm_refinement"] = llm_result

                # Update best match (prefer higher scores, entailment > partial > missing)
                if score is not None and score > best_score:
                    best_score = score
                    best_label = label
                    best_match = {
                        "summary_sentence": match["summary_sentence"],
                        "nli_label": match["nli_label"],
                        "nli_confidence": match["nli_confidence"],
                        "llm_refinement": match.get("llm_refinement")
                    }

            # Assign final label and score
            if best_label is None:
                best_label = "unclear"
                best_score = 0.0
                excluded_count += 1
            else:
                if best_label == "covered":
                    covered_count += 1
                    earned_score += 1.0
                elif best_label == "partially_covered":
                    partially_covered_count += 1
                    earned_score += 0.5
                elif best_label == "missing":
                    missing_count += 1
                    earned_score += 0.0
                elif best_label == "contradicted":
                    contradicted_count += 1
                    earned_score += max(0.0, best_score)  # Clamp to 0 for coverage
                elif best_label == "unclear":
                    excluded_count += 1

            per_key_point.append({
                "key_point": key_point,
                "final_label": best_label,
                "final_score": round(best_score, 4) if best_score is not None else 0.0,
                "best_match": best_match,
                "all_matches": matches
            })

        # Compute coverage score
        total_key_points = len(key_points)
        max_score = total_key_points - excluded_count
        coverage_score = earned_score / max_score if max_score > 0 else 0.0

        return {
            "per_key_point": per_key_point,
            "coverage_score": round(coverage_score, 4),
            "total_key_points": total_key_points,
            "covered_count": covered_count,
            "partially_covered_count": partially_covered_count,
            "missing_count": missing_count,
            "contradicted_count": contradicted_count,
            "excluded_count": excluded_count,
            "earned_score": round(earned_score, 2),
            "max_score": max_score
        }

    # ─── Key Point Loading ─────────────────────────────────────────────────

    def _get_kp_extractor(self):
        """Lazy initialization of key point extractor."""
        if self._kp_extractor is None:
            print("  [Initializing] Key point extractor for on-the-fly extraction...")
            self._kp_extractor = EntailmentKeyPointExtractor(model_name=self.model_name)
            # Share the embedding model to avoid loading it twice
            self._kp_extractor.embedding_model = self.embedding_model
        return self._kp_extractor

    def load_preextracted_keypoints(self, keypoints_dir, csv_basename):
        """
        Load pre-extracted key points from JSON file or a direct JSON path.
        Returns dict: {(filename, profile, model): [key_points]}
        """
        if os.path.isfile(keypoints_dir) or str(keypoints_dir).lower().endswith(".json"):
            json_path = keypoints_dir
        else:
            json_path = os.path.join(keypoints_dir, f"entailment_kp_{csv_basename}.json")

        if not os.path.exists(json_path):
            print(f"  [Warning] Pre-extracted keypoints not found: {json_path}")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            keypoints_map = {}
            for entry in data:
                key = (entry["filename"], entry.get("profile", ""), entry.get("model", ""))
                keypoints_map[key] = entry["key_points"]

            print(f"  [Loaded] {len(keypoints_map)} pre-extracted keypoint sets from {json_path}")
            return keypoints_map
        except Exception as e:
            print(f"  [Warning] Failed to load keypoints from {json_path}: {e}")
            return {}

    # ─── Main Evaluation ───────────────────────────────────────────────────

    def evaluate_row(self, transcript, summary, filename, profile="", model="", key_points=None):
        """
        Full pipeline for one (transcript, summary) pair.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {filename} ({profile}, {model})")
        print(f"{'='*60}")

        # Validate summary
        if not summary or summary.strip().lower() in ["no summary generated.", "error in summarization.", "nan"]:
            print("  [Skip] Summary is missing or invalid.")
            return {
                "coverage_score": 0.0,
                "total_key_points": 0,
                "covered_count": 0,
                "partially_covered_count": 0,
                "missing_count": 0,
                "contradicted_count": 0,
                "excluded_count": 0,
                "key_items": []
            }

        # Get key points (from cache, parameter, or extract on-the-fly)
        if key_points is None:
            cache_key = filename
            if cache_key in self.kp_cache:
                key_points = self.kp_cache[cache_key]
                print(f"  [Cache] Reusing {len(key_points)} key points")
            else:
                # Extract key points on-the-fly
                print("  [Extracting] Key points not found, extracting on-the-fly...")
                extractor = self._get_kp_extractor()
                short = len(transcript.split()) > 2000
                key_points = extractor.get_key_items_for_transcript(transcript, filename, short=short)
                self.kp_cache[filename] = key_points
                print(f"  [Extracted] {len(key_points)} key points")
        else:
            self.kp_cache[filename] = key_points

        if not key_points:
            print("  [Skip] No key points available.")
            return {
                "coverage_score": 0.0,
                "total_key_points": 0,
                "covered_count": 0,
                "partially_covered_count": 0,
                "missing_count": 0,
                "contradicted_count": 0,
                "excluded_count": 0,
                "key_items": []
            }

        print(f"  [Stage 3] Segmenting summary...")
        summary_sentences = self.segment_summary(summary)
        print(f"    - {len(summary_sentences)} sentences")

        if not summary_sentences:
            print("  [Skip] No valid summary sentences.")
            return {
                "coverage_score": 0.0,
                "total_key_points": len(key_points),
                "covered_count": 0,
                "partially_covered_count": 0,
                "missing_count": len(key_points),
                "contradicted_count": 0,
                "excluded_count": 0,
                "key_items": [{"key_point": kp, "final_label": "missing", "final_score": 0.0, "best_match": None, "all_matches": []} for kp in key_points]
            }

        print(f"  [Stage 4] Retrieving candidate pairs...")
        candidate_pairs = self.retrieve_candidates(key_points, summary_sentences)
        print(f"    - {len(candidate_pairs)} candidate pairs")

        if not candidate_pairs:
            print("  [Skip] No candidate pairs above similarity threshold.")
            return {
                "coverage_score": 0.0,
                "total_key_points": len(key_points),
                "covered_count": 0,
                "partially_covered_count": 0,
                "missing_count": len(key_points),
                "contradicted_count": 0,
                "excluded_count": 0,
                "key_items": [{"key_point": kp, "final_label": "missing", "final_score": 0.0, "best_match": None, "all_matches": []} for kp in key_points]
            }

        print(f"  [Stage 5] Running NLI verification...")
        nli_results = self.verify_pairs_nli(key_points, summary_sentences, candidate_pairs)
        print(f"    - NLI completed for {len(nli_results)} pairs")

        # Count neutrals for LLM refinement
        neutral_count = sum(1 for r in nli_results if r["nli_label"] == "neutral")
        print(f"  [Stage 6] Refining {neutral_count} neutral cases with LLM...")

        llm_refinements = {}
        for result in nli_results:
            if result["nli_label"] == "neutral":
                cache_key = f"{result['key_point']}|||{result['summary_sentence']}"
                if cache_key not in llm_refinements:
                    llm_refinement = self.refine_neutral_with_llm(result['key_point'], result['summary_sentence'])
                    llm_refinements[cache_key] = llm_refinement

        print(f"  [Stage 7] Aggregating results...")
        aggregated = self.aggregate_per_keypoint(key_points, nli_results, llm_refinements)

        print(f"  [Done] Coverage: {aggregated['coverage_score']:.2%} ({aggregated['covered_count']} covered, {aggregated['partially_covered_count']} partial, {aggregated['missing_count']} missing)")

        return {
            "coverage_score": aggregated["coverage_score"],
            "total_key_points": aggregated["total_key_points"],
            "covered_count": aggregated["covered_count"],
            "partially_covered_count": aggregated["partially_covered_count"],
            "missing_count": aggregated["missing_count"],
            "contradicted_count": aggregated["contradicted_count"],
            "excluded_count": aggregated["excluded_count"],
            "earned_score": aggregated["earned_score"],
            "max_score": aggregated["max_score"],
            "key_items": aggregated["per_key_point"]
        }


# ─── CLI Processing ────────────────────────────────────────────────────────

def read_csv_smart(csv_path):
    """Read summary CSVs while preserving quoted commas in text fields."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(65536)
        file.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","

    return pd.read_csv(
        csv_path,
        sep=delimiter,
        engine="python",
        dtype=str,
        on_bad_lines="warn",
    )

def process_csv(csv_path, evaluator, output_dir, keypoints_map, limit=None, filenames=None, sample=None, force=False):
    """Process a single CSV file."""
    if not os.path.exists(csv_path):
        print(f"[Skip] CSV not found: {csv_path}")
        return

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    summary_model = base_name
    for prefix in ["short_degraded_", "long_degraded_"]:
        if summary_model.startswith(prefix):
            summary_model = summary_model[len(prefix):]
            break

    output_path = os.path.join(output_dir, f"nli_hybrid_eval_{base_name}.json")

    # Load existing results
    results = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}
                print(f"Loaded {len(results)} existing results for {base_name}")
        except:
            pass

    # Load CSV
    try:
        df = read_csv_smart(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    # Filter by filenames if specified
    if filenames:
        df = df[df['filename'].isin(filenames)]
        if force:
            filenames_set = set(filenames)
            results = [r for r in results if r.get('filename') not in filenames_set]
            processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}

    # Filter unprocessed rows
    mask = df.apply(lambda r: f"{r['filename']}_{r.get('profile', '')}" not in processed_ids, axis=1)
    df_to_process = df[mask]

    # Apply sampling/limit
    if sample:
        df_to_process = df_to_process.sample(min(sample, len(df_to_process)))
    elif limit:
        df_to_process = df_to_process.head(limit)

    if len(df_to_process) == 0:
        print(f"[Skip] {base_name}: No new rows to process.")
        return

    print(f"\n{'='*60}")
    print(f"Processing {len(df_to_process)} rows from {base_name}...")
    print(f"{'='*60}")

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        pt_before = evaluator.total_prompt_tokens
        ct_before = evaluator.total_completion_tokens

        filename = row['filename']
        profile = row.get('profile', '')
        model = row.get('model', '')

        # Keypoints belong to the source transcript, not to a degraded summary
        # profile. Prefer metadata matches, but never require profile/model equality.
        key_points = keypoints_map.get((filename, profile, model))
        if key_points is None:
            filename_key = str(filename).strip()
            filename_matches = [
                value
                for (kp_filename, _, _), value in keypoints_map.items()
                if str(kp_filename).strip() == filename_key
            ]
            if filename_matches:
                key_points = filename_matches[0]

        res = evaluator.evaluate_row(
            str(row['transcript']),
            str(row['summary']),
            filename,
            profile=profile,
            model=model,
            key_points=key_points
        )

        res['filename'] = filename
        res['profile'] = profile
        res['summary_model'] = summary_model
        res['token_usage'] = {
            "prompt_tokens": evaluator.total_prompt_tokens - pt_before,
            "completion_tokens": evaluator.total_completion_tokens - ct_before,
            "total_tokens": (evaluator.total_prompt_tokens - pt_before) + (evaluator.total_completion_tokens - ct_before),
        }
        results.append(res)

        # Save after each row
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] Results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI-Hybrid Relevance Evaluation (Stages 3-7)")
    parser.add_argument("--csv", type=str, default="pilot_summaries.csv", help="CSV file path or substring")
    parser.add_argument("--csv-dir", type=str, help="Directory containing CSV files to evaluate")
    parser.add_argument("--output-dir", type=str, default="nli_hybrid_evaluation_results", help="Output directory")
    parser.add_argument("--keypoints-dir", type=str, default="entailment_keypoints_results", help="Directory with pre-extracted keypoints")
    parser.add_argument("--keypoints-file", type=str, help="Direct JSON file with pre-extracted keypoints")
    parser.add_argument("--limit", type=int, help="Limit rows per file")
    parser.add_argument("--sample", type=int, help="Random sample of rows per file")
    parser.add_argument("--filenames", type=str, nargs="+", help="Specific filenames to process")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieval candidates")
    parser.add_argument("--nli-model", type=str, default="joeddav/xlm-roberta-large-xnli", help="NLI model name")
    parser.add_argument("--entailment-threshold", type=float, default=0.7, help="Entailment confidence threshold")
    parser.add_argument("--contradiction-threshold", type=float, default=0.7, help="Contradiction confidence threshold")
    parser.add_argument("--model", type=str, default="meta-llama-3.1-8b-instruct", help="LLM model for refinement")
    parser.add_argument("--batch-size", type=int, default=16, help="NLI batch size")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = NLIHybridRelevanceEvaluator(
        model_name=args.model,
        nli_model_name=args.nli_model,
        top_k=args.top_k,
        entailment_threshold=args.entailment_threshold,
        contradiction_threshold=args.contradiction_threshold,
        batch_size=args.batch_size
    )

    # Print token usage at exit
    import atexit
    def print_usage():
        pt = evaluator.total_prompt_tokens
        ct = evaluator.total_completion_tokens
        print(f"\n{'='*60}")
        print(f"Token Usage")
        print(f"{'='*60}")
        print(f"  Prompt tokens:     {pt:>8}")
        print(f"  Completion tokens: {ct:>8}")
        print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    # Discover CSV files
    if args.csv_dir:
        files_to_process = sorted(
            os.path.join(args.csv_dir, filename)
            for filename in os.listdir(args.csv_dir)
            if filename.lower().endswith(".csv")
        ) if os.path.isdir(args.csv_dir) else []
        if not files_to_process:
            print(f"[Error] No CSV files found in directory: {args.csv_dir}")
            exit(1)
        print(f"Queue: {[os.path.basename(f) for f in files_to_process]}")
        csv_discovery_complete = True
    else:
        csv_discovery_complete = False

    strict_folders = [
        r"c:\Users\reyna\Downloads\MACode\short_summaries_strict",
        r"c:\Users\reyna\Downloads\MACode\summaries_strict",
        r"c:\Users\reyna\Downloads\MACode\summaries_strict3",
        r"c:\Users\reyna\Downloads\MACode\summaries_strict_4",
    ]

    all_csvs = []
    for folder in strict_folders:
        if os.path.exists(folder):
            all_csvs.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")])

    # Also check current directory and parent
    for check_path in [".", ".."]:
        pilot_path = os.path.join(check_path, "pilot_summaries.csv")
        if os.path.exists(pilot_path):
            all_csvs.append(os.path.abspath(pilot_path))

    # Filter by --csv argument unless --csv-dir already selected the queue.
    if not csv_discovery_complete and args.csv:
        files_to_process = []
        if os.path.exists(args.csv):
            files_to_process = [args.csv]
        else:
            matches = [f for f in all_csvs if args.csv.lower() in os.path.basename(f).lower()]
            if not matches:
                print(f"[Error] No file found matching pattern: {args.csv}")
                exit(1)
            files_to_process = matches
    elif not csv_discovery_complete:
        files_to_process = all_csvs

    if not files_to_process:
        print("[Error] No CSV files to process.")
        exit(1)

    print(f"Queue: {[os.path.basename(f) for f in files_to_process]}")

    # Load pre-extracted keypoints for all CSVs
    all_keypoints_map = {}
    keypoints_source = args.keypoints_file or args.keypoints_dir
    for csv_file in files_to_process:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        kp_map = evaluator.load_preextracted_keypoints(keypoints_source, base_name)
        all_keypoints_map.update(kp_map)

    # Process each CSV
    for csv_file in files_to_process:
        process_csv(
            csv_file,
            evaluator,
            args.output_dir,
            all_keypoints_map,
            limit=args.limit,
            filenames=args.filenames,
            sample=args.sample,
            force=args.force
        )
