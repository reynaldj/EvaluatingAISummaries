"""Run hosted NLI-hybrid relevance evaluation with a Llama refinement model."""

import argparse
import atexit
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from evaluate_relevance_nli_hybrid import NLIHybridRelevanceEvaluator, process_csv
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluate_relevance_nli_hybrid import NLIHybridRelevanceEvaluator, process_csv


class LocalLlamaNLIHybridRelevanceEvaluator(NLIHybridRelevanceEvaluator):
    """NLI-hybrid evaluator whose neutral-case refinement runs on local Llama."""

    def __init__(self, model_name, max_new_tokens=1024, max_retries=5, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        print(f"Loading local Llama model: {model_name}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _normalize_response(text):
        if not text:
            return ""
        for marker in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>"):
            text = text.replace(marker, "")
        return text.strip()

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        last_error = None
        for attempt in range(max_attempts or self.max_retries):
            try:
                inputs = self.llm_tokenizer.apply_chat_template(
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

                input_length = inputs["input_ids"].shape[-1]
                self.total_prompt_tokens += int(input_length)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
                generated = outputs[0][input_length:]
                self.total_completion_tokens += int(generated.shape[-1])
                response = self._normalize_response(
                    self.llm_tokenizer.decode(generated, skip_special_tokens=False)
                )
                if response:
                    return response
                last_error = "empty Llama response"
            except Exception as exc:
                last_error = exc
                wait = 2 * (attempt + 1) + random.uniform(0, 0.5)
                print(f"    [Llama Error] {exc} (retry in {wait:.0f}s)")
                if attempt + 1 < (max_attempts or self.max_retries):
                    time.sleep(wait)
        raise SystemExit(f"\n[FATAL] Local Llama generation failed: {last_error}")


DEFAULT_MODEL = os.path.expanduser("~/models/llama-70b")
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "nli_hybrid_evaluation_results_llama70b")
DEFAULT_KEYPOINTS_FILE = os.path.join(
    os.path.dirname(__file__),
    "llama-70b",
    "entailment_keypoints_results_summaries_strict_9_llama70b_narrow",
    "keypoints_llama.json",
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summaries with local Llama LLM refinement and NLI."
    )
    parser.add_argument("--csv", help="One CSV path or filename substring.")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keypoints-file",
        default=DEFAULT_KEYPOINTS_FILE,
        help="JSON file containing the pre-extracted keypoints to evaluate.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--filenames", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nli-model", default="joeddav/xlm-roberta-large-xnli")
    parser.add_argument("--entailment-threshold", type=float, default=0.7)
    parser.add_argument("--contradiction-threshold", type=float, default=0.7)
    parser.add_argument(
        "--model",
        type=os.path.expanduser,
        default=DEFAULT_MODEL,
        help="Llama model ID or local model path.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not os.path.isfile(args.keypoints_file):
        raise SystemExit(f"Keypoint file not found: {args.keypoints_file}")

    if args.csv:
        if os.path.isfile(args.csv):
            files_to_process = [args.csv]
        else:
            files_to_process = sorted(
                os.path.join(args.csv_dir, name)
                for name in os.listdir(args.csv_dir)
                if name.lower().endswith(".csv") and args.csv.lower() in name.lower()
            )
    else:
        files_to_process = sorted(
            os.path.join(args.csv_dir, name)
            for name in os.listdir(args.csv_dir)
            if name.lower().endswith(".csv")
        )

    if not files_to_process:
        raise SystemExit(f"No CSV files found for evaluation in: {args.csv_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model: {args.model}")
    print(f"Keypoints: {args.keypoints_file}")
    print(f"Queue: {[os.path.basename(path) for path in files_to_process]}")

    evaluator = LocalLlamaNLIHybridRelevanceEvaluator(
        model_name=args.model,
        nli_model_name=args.nli_model,
        top_k=args.top_k,
        entailment_threshold=args.entailment_threshold,
        contradiction_threshold=args.contradiction_threshold,
        batch_size=args.batch_size,
    )

    @atexit.register
    def print_usage():
        prompt = evaluator.total_prompt_tokens
        completion = evaluator.total_completion_tokens
        print(f"\nToken usage: prompt={prompt}, completion={completion}, total={prompt + completion}")

    keypoints_map = {}
    for csv_file in files_to_process:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        keypoints_map.update(evaluator.load_preextracted_keypoints(args.keypoints_file, base_name))

    for csv_file in files_to_process:
        process_csv(
            csv_file,
            evaluator,
            args.output_dir,
            keypoints_map,
            limit=args.limit,
            filenames=args.filenames,
            sample=args.sample,
            force=args.force,
        )


if __name__ == "__main__":
    main()
