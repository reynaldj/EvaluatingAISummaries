import argparse
import glob
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma31b_factuality_evaluator import (
    Gemma31BFactualityEvaluator,
    run_file,
)


class Llama70BFactualityEvaluator(Gemma31BFactualityEvaluator):
    """Same factuality pipeline, with a local text-only Llama backend."""

    def __init__(
        self,
        model_name="/home/users/r/reynaldj/models/llama-70b",
        method="StSum_Text",
        chunk_size=3,
        max_text_length=1000,
        max_new_tokens=1024,
        temperature=0.0,
        device_map="auto",
        torch_dtype="auto",
    ):
        # Keep every factuality-stage setting identical to the Gemma evaluator.
        self.model_name = model_name
        self.resolved_model_name = model_name
        self.method = method
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        print(f"Loading Llama model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def _chat(self, prompt, system_prompt="Du bist ein hilfreicher Assistent für die Analyse von Texten."):
        """Llama replacement for Gemma's processor-based chat generation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
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
        self.total_prompt_tokens += int(input_len)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][input_len:]
        self.total_completion_tokens += int(generated.shape[-1])
        return self._normalize_response(
            self.tokenizer.decode(generated, skip_special_tokens=False)
        )

    def _normalize_response(self, text):
        if not text:
            return ""
        return (
            text.strip()
            .replace("<|start_header_id|>", "")
            .replace("<|end_header_id|>", "")
            .replace("<|eot_id|>", "")
            .replace("<|end_of_text|>", "")
            .replace("<|im_end|>", "")
            .strip()
        )


def parse_args():
    default_csv_dir = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
    default_output_dir = os.path.join(os.path.dirname(__file__), "results-factuality-evaluation-llama70b")
    parser = argparse.ArgumentParser(description="Llama 70B factuality evaluator mirroring the German SEval academic pipeline.")
    parser.add_argument("--csv", type=str, default=None, help="Single CSV with transcript and summary columns")
    parser.add_argument("--csv-dir", type=str, default=default_csv_dir, help="Directory containing CSVs to evaluate")
    parser.add_argument("--summary-column", type=str, default="summary", help="Summary column name")
    parser.add_argument("--transcript-column", type=str, default="transcript", help="Transcript column name")
    parser.add_argument("--model", type=os.path.expanduser, default=os.path.expanduser("~/models/llama-70b"), help="Llama model ID or local model path")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output folder")
    parser.add_argument("--resume", default=True, action="store_true", help="Skip rows already in the matching output file")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files when using --csv-dir")
    parser.add_argument("--max-row-retries", type=int, default=3, help="Retry count for failed row evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = Llama70BFactualityEvaluator(model_name=args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        run_file(evaluator, args.csv, args.output_dir, args.resume, args.max_row_retries, args.summary_column, args.transcript_column)
    else:
        if not os.path.isdir(args.csv_dir):
            raise FileNotFoundError(f"CSV directory not found: {args.csv_dir}")
        files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
        if not files:
            raise SystemExit(f"No CSVs found matching {os.path.join(args.csv_dir, args.pattern)}")
        for csv_path in files:
            run_file(evaluator, csv_path, args.output_dir, args.resume, args.max_row_retries, args.summary_column, args.transcript_column)

    print(f"\nAll files done. Results in {args.output_dir}")
