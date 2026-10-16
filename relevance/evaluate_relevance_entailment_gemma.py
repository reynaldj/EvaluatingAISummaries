import os
import random
import sys
import time

import tiktoken
from openai import OpenAI  # Only used for the inherited evaluator type's interface.
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from relevance.evaluate_relevance_entailment import EntailmentRelevanceEvaluator
    from relevance.evaluate_relevance_entailment_vllm import (
        preload_keypoints,
        process_csv,
        read_csv_smart,
    )
except ModuleNotFoundError as exc:
    if exc.name != "relevance":
        raise
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from relevance.evaluate_relevance_entailment import EntailmentRelevanceEvaluator
    from relevance.evaluate_relevance_entailment_vllm import (
        preload_keypoints,
        process_csv,
        read_csv_smart,
    )


class GemmaEntailmentRelevanceEvaluator(EntailmentRelevanceEvaluator):
    """Relevance evaluator using a locally loaded Gemma model."""

    def __init__(
        self,
        model="google/gemma-4-31B-it",
        max_new_tokens=1024,
        temperature=0.0,
        device_map="auto",
        torch_dtype="auto",
        max_retries=5,
        retry_wait=2.0,
        no_embeddings=True,
    ):
        self.model_name = model
        self.resolved_model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.no_embeddings = no_embeddings
        self.fatal_error = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.kp_cache = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = None

        print(f"Loading Gemma model: {model}")
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def _programmatic_deduplication(self, items, sim_limit=0.85):
        if self.no_embeddings:
            return (items or [])[:50]
        return super()._programmatic_deduplication(items, sim_limit=sim_limit)

    def _union_and_dedup_key_items(self, run1, run2, emb_threshold=0.90):
        if self.no_embeddings:
            return (run1 + run2)[:50], {}
        return super()._union_and_dedup_key_items(run1, run2, emb_threshold=emb_threshold)

    def _cluster_key_items(self, items, sim_threshold=0.88):
        if self.no_embeddings:
            return items
        return super()._cluster_key_items(items, sim_threshold=sim_threshold)

    def check_binary_coverage_batch(self, key_points, summary, max_retries=2):
        """Keep one model judgment per keypoint when embeddings are disabled."""
        if self.no_embeddings:
            return [
                self.check_binary_coverage(key_point, summary, max_retries=max_retries)
                for key_point in key_points
            ]
        return super().check_binary_coverage_batch(
            key_points,
            summary,
            max_retries=max_retries,
        )

    def _normalize_response(self, text):
        if not text:
            return ""
        for marker in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_end|>"):
            text = text.replace(marker, "")
        return text.strip()

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        """Generate one response locally, retrying transient generation failures."""
        last_error = None
        for attempt in range(max_attempts or self.max_retries):
            try:
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
                    inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

                input_length = inputs["input_ids"].shape[-1]
                self.total_prompt_tokens += int(input_length)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
                generated = outputs[0][input_length:]
                self.total_completion_tokens += int(generated.shape[-1])
                response = self.processor.decode(generated, skip_special_tokens=False)
                response = self._normalize_response(response)
                if response:
                    return response
                last_error = "empty Gemma response"
            except Exception as exc:
                last_error = exc
                wait = self.retry_wait * (attempt + 1) + random.uniform(0, 0.5)
                print(
                    f"    [Gemma Error] {type(exc).__name__}: {exc} "
                    f"(retry in {wait:.0f}s, attempt {attempt + 1}/{max_attempts})"
                )
                if attempt + 1 < (max_attempts or self.max_retries):
                    time.sleep(wait)

        print(f"    [Gemma API FAILED] {last_error}")
        return None


def main():
    import argparse
    import atexit
    import os

    parser = argparse.ArgumentParser(
        description="Evaluate all summaries against local Gemma using entailment relevance scoring."
    )
    default_input_dir = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
    default_output_dir = os.path.join(os.path.dirname(__file__), "entailment_evaluation_results_gemma31b")
    default_keypoints_dir = os.path.join(
        os.path.dirname(__file__),
        "entailment_keypoints_results_summaries_strict_9_gemma",
    )

    parser.add_argument("--csv", default=None, help="One CSV path or filename substring.")
    parser.add_argument("--input-dir", default=default_input_dir, help="Directory containing summary CSVs.")
    parser.add_argument(
        "--keypoints-dir",
        nargs="+",
        default=[default_keypoints_dir],
        help="Directories containing pre-extracted keypoint JSON files.",
    )
    parser.add_argument(
        "--model",
        type=os.path.expanduser,
        default=os.path.expanduser("~/models/gemma-4-31B-it"),
        help="Gemma model ID or local model path.",
    )
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--filenames", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = GemmaEntailmentRelevanceEvaluator(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
    )
    preload_keypoints(evaluator, args.keypoints_dir)

    @atexit.register
    def print_usage():
        print("\n=== Token Usage ===")
        print(f"  Prompt tokens:     {evaluator.total_prompt_tokens:>8}")
        print(f"  Completion tokens: {evaluator.total_completion_tokens:>8}")
        print(f"  Total tokens:      {evaluator.total_prompt_tokens + evaluator.total_completion_tokens:>8}")

    if args.csv:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            matches = [
                os.path.join(args.input_dir, name)
                for name in os.listdir(args.input_dir)
                if args.csv.lower() in name.lower() and name.endswith(".csv")
            ]
            if len(matches) != 1:
                raise SystemExit(f"Expected one CSV match for {args.csv}, found: {matches}")
            csv_path = matches[0]
        files = [csv_path]
    else:
        files = [
            os.path.join(args.input_dir, name)
            for name in sorted(os.listdir(args.input_dir))
            if name.endswith(".csv")
        ]

    if args.filenames:
        requested = set(args.filenames)
        files = [
            path for path in files
            if requested.intersection(set(read_csv_smart(path)["filename"].astype(str)))
        ]

    print(f"Queue: {[os.path.basename(path) for path in files]}")
    for csv_path in files:
        process_csv(
            csv_path,
            evaluator,
            args.output_dir,
            limit=args.limit,
            filenames=args.filenames,
            sample=args.sample,
            force=args.force,
            retry_failed=args.retry_failed,
        )


if __name__ == "__main__":
    main()
