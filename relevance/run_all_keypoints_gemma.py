import argparse
import glob
import hashlib
import json
import os
import re
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from extract_keypoints_entailment import EntailmentKeyPointExtractor


def transcript_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class GemmaEntailmentKeyPointExtractor(EntailmentKeyPointExtractor):
    """Full entailment key-point pipeline backed by a local Gemma model."""

    def __init__(
        self,
        model_name: str = "google/gemma-4-31B-it",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            self.resolved_model_name = model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            raise

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        print("Loading SentenceTransformer model...")
        try:
            self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        except Exception as exc:
            print(
                "  [Warning] Failed to load intfloat/multilingual-e5-large "
                f"({exc}); falling back to all-MiniLM-L6-v2."
            )
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # The inherited extractor uses ``self.tokenizer.encode(...)`` to
        # split long transcripts into bounded chunks.  Reuse Gemma's tokenizer
        # instead of leaving this as None.
        self.tokenizer = self.processor.tokenizer
        self.kp_cache = {}

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        """Adapt the base pipeline's chat calls to local Gemma generation."""
        prompt = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in messages
        )
        attempts = max_attempts
        for attempt in range(attempts):
            try:
                return self._chat(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                if attempt == attempts - 1:
                    raise
                wait = min(30, 2 ** attempt)
                print(f"    [Gemma Error] {type(exc).__name__}: {exc}; retrying in {wait}s...")
                time.sleep(wait)
        return None

    def _chat(self, prompt: str, max_tokens=None, temperature=0.0) -> str:
        """Generate one completion using local Gemma."""
        messages = [{"role": "user", "content": prompt}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens or self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        generated = outputs[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=False)
        return self._normalize_response(text)

    def _normalize_response(self, text: str) -> str:
        """Clean LLM output for parsing."""
        if not text:
            return ""

        text = text.strip()
        text = text.replace("<|start_header_id|>", "")
        text = text.replace("<|end_header_id|>", "")
        text = text.replace("<|eot_id|>", "")
        text = text.replace("<|im_end|>", "")
        return text.strip()

    def get_key_items_for_transcript(self, transcript, filename, short=False):
        return super().get_key_items_for_transcript(transcript, filename, short=short)

    def extract_key_points(self, transcript, filename=None, short=False):
        """Compatibility wrapper for callers using the older extractor API."""
        return self.get_key_items_for_transcript(
            transcript,
            filename or "<unknown>",
            short=short,
        )


def build_entry(key_points, filename, profile, model, thash, summary, word_count, short):
    return {
        "key_points": key_points,
        "filename": filename,
        "profile": profile,
        "model": model,
        "transcript_hash": thash,
        "variant": "long" if word_count > 2000 else "short",
        "short_mode": short,
        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
        "metadata": {
            "num_key_points": len(key_points),
            "word_count": word_count,
            "short_mode": short,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma-based key-point extraction, deduped per transcript")
    default_csv_dir = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
    default_output_dir = os.path.join(
        os.path.dirname(__file__),
        "entailment_keypoints_results_summaries_strict_9_gemma31b_wide",
    )

    parser.add_argument("--csv-dir", default=default_csv_dir)
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--output-file", default=None, help="Output JSON filename (default: keypoints_gemma.json)")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument(
        "--model",
        type=os.path.expanduser,
        default=os.path.expanduser("~/models/llama-70b"),
        help="Gemma model ID or local model path",
    )
    parser.add_argument(
        "--llm-only",
        default=True,
        action="store_true",
        help="Force the LLM pipeline for every transcript and disable the statistical fallback",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file or "keypoints_gemma.json")

    print(f"Initializing Gemma model: {args.model}")
    extractor = GemmaEntailmentKeyPointExtractor(model_name=args.model)

    if args.llm_only:
        extractor._assess_transcript_quality = lambda transcript: ("high", 0.0)
        print("LLM-only mode: forcing the entailment pipeline for every transcript")

    results = []
    emitted = set()
    cache = {}
    if not args.force and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        emitted = {r["transcript_hash"] for r in results if r.get("transcript_hash")}
        cache = {r["transcript_hash"]: r.get("key_points", []) for r in results if r.get("transcript_hash")}
        print(f"Resume: loaded {len(results)} entries ({len(emitted)} unique transcripts)")

    files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No CSVs found matching {os.path.join(args.csv_dir, args.pattern)}")

    new_entries = 0
    for csv_path in files:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
        print(f"\nScanning {len(df)} rows from {base_name}...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            profile = row.get('profile', '')
            model = row.get('model', '')
            transcript = str(row['transcript'])
            summary = str(row.get('summary', ''))
            thash = transcript_hash(transcript)

            if thash in emitted:
                continue

            word_count = len(transcript.split())
            short = word_count <= 2000

            if thash in cache:
                print(f"  [Cache] Reusing key points for transcript {filename} ({base_name})")
                key_points = cache[thash]
            else:
                print(f"\n{'='*60}")
                print(f"Extracting key points: {filename} ({base_name})")
                print(f"{'='*60}")
                key_points = extractor.get_key_items_for_transcript(transcript, filename, short=short)
                cache[thash] = key_points

            entry = build_entry(
                key_points, filename, profile, model, thash, summary, word_count, short
            )
            results.append(entry)
            emitted.add(thash)
            new_entries += 1

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {new_entries} new transcripts added. {len(results)} entries total ({len(emitted)} unique transcripts).")
    print(f"Results saved to: {output_file}")
