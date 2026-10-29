import argparse
import glob
import hashlib
import json
import os
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_keypoints_entailment import EntailmentKeyPointExtractor


def transcript_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class LlamaEntailmentKeyPointExtractor(EntailmentKeyPointExtractor):
    """Full entailment key-point pipeline backed by a local text-only Llama."""

    def __init__(
        self,
        model_name="/home/users/r/reynaldj/models/llama-70b",
        max_new_tokens=2048,
        temperature=0.0,
        device_map="auto",
        torch_dtype="auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            self.resolved_model_name = model_name
        except Exception as exc:
            print(f"Failed to load {model_name}: {exc}")
            raise

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        print("Loading SentenceTransformer model...")
        try:
            self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        except Exception as exc:
            print(f"  [Warning] Failed to load intfloat/multilingual-e5-large ({exc}); falling back to all-MiniLM-L6-v2.")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kp_cache = {}

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        prompt = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
        for attempt in range(max_attempts):
            try:
                return self._chat(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise
                wait = min(30, 2 ** attempt)
                print(f"    [Llama Error] {type(exc).__name__}: {exc}; retrying in {wait}s...")
                time.sleep(wait)

    def _chat(self, prompt, max_tokens=None, temperature=0.0):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens or self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][input_len:]
        return self._normalize_response(self.tokenizer.decode(generated, skip_special_tokens=False))

    @staticmethod
    def _normalize_response(text):
        if not text:
            return ""
        return text.strip().replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()


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
        "metadata": {"num_key_points": len(key_points), "word_count": word_count, "short_mode": short},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama-based key-point extraction, deduped per transcript")
    default_csv_dir = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
    default_output_dir = os.path.join(os.path.dirname(__file__), "entailment_keypoint_results_summaries_strict_9_llama70b")
    parser.add_argument("--csv-dir", default=default_csv_dir)
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--output-file", default=None, help="Output JSON filename (default: keypoints_llama.json)")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument(
            "--model",
            type=lambda p: os.path.expanduser(p),
            default=os.path.expanduser("~/models/llama-70b"),
            help="Gemma model ID or local model path",
        )
    parser.add_argument("--llm-only", action="store_true", default=True, help="Force the LLM pipeline for every transcript and disable the statistical fallback")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file or "keypoints_llama.json")
    print(f"Initializing Llama model: {args.model}")
    extractor = LlamaEntailmentKeyPointExtractor(model_name=args.model)
    if args.llm_only:
        extractor._assess_transcript_quality = lambda transcript: ("high", 0.0)
        print("LLM-only mode: forcing the entailment pipeline for every transcript")

    results, emitted, cache = [], set(), {}
    if not args.force and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            results = json.load(file)
        emitted = {entry["transcript_hash"] for entry in results if entry.get("transcript_hash")}
        cache = {entry["transcript_hash"]: entry.get("key_points", []) for entry in results if entry.get("transcript_hash")}
        print(f"Resume: loaded {len(results)} entries ({len(emitted)} unique transcripts)")

    files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No CSVs found matching {os.path.join(args.csv_dir, args.pattern)}")

    new_entries = 0
    for csv_path in files:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        dataframe = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
        print(f"\nScanning {len(dataframe)} rows from {base_name}...")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            transcript = str(row["transcript"])
            thash = transcript_hash(transcript)
            if thash in emitted:
                continue
            filename = row["filename"]
            profile = row.get("profile", "")
            model = row.get("model", "")
            summary = str(row.get("summary", ""))
            word_count = len(transcript.split())
            short = word_count <= 2000
            print(f"\n{'=' * 60}\nExtracting key points: {filename} ({base_name})\n{'=' * 60}")
            key_points = extractor.get_key_items_for_transcript(transcript, filename, short=short)
            cache[thash] = key_points
            results.append(build_entry(key_points, filename, profile, model, thash, summary, word_count, short))
            emitted.add(thash)
            new_entries += 1
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nDone. {new_entries} new transcripts added. {len(results)} entries total ({len(emitted)} unique transcripts).")
    print(f"Results saved to: {output_file}")
