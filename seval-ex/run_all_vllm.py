import argparse
import glob
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from german_seval_vllm import GermanSEvalVLLMPipeline


class RetryVLLMPipeline(GermanSEvalVLLMPipeline):
    """Same logic, but retries on any transient API error (connection reset, 5xx, rate limit)."""
    def _get_completion(self, prompt, max_retries=6, base_delay=2):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Du bist ein hilfreicher Assistent für die Analyse von Texten."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                return response.choices[0].message.content
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                print(f"\nAPI Error: {e}. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
        return ""


def run_file(pipeline, csv_path, output_dir, resume, max_row_retries):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    model_slug = pipeline.model_name.replace("/", "_").replace(":", "_")
    output_file = os.path.join(output_dir, f"results_{base_name}_{model_slug}.json")

    results = []
    seen = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        seen = {(r.get("filename", ""), r.get("profile", "")) for r in results}
        print(f"Resume: loaded {len(results)} existing results for {base_name}")

    df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
    skipped = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc=base_name):
        filename = row.get("filename", f"row_{i}")
        profile = row.get("profile", "")
        if resume and (filename, profile) in seen:
            continue

        summary = str(row["summary"])
        transcript = str(row["transcript"])

        res = None
        for attempt in range(1, max_row_retries + 1):
            try:
                res = pipeline.evaluate_single(transcript, summary)
                break
            except Exception as e:
                print(f"\nRow {filename} attempt {attempt}/{max_row_retries} raised: {e}")
                if attempt < max_row_retries:
                    time.sleep(5 * attempt)

        if res is None:
            print(f"SKIPPING row {filename} after {max_row_retries} attempts")
            skipped += 1
            continue

        res["filename"] = filename
        res["profile"] = profile
        results.append(res)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Finished {base_name}: {len(results)} results, {skipped} skipped -> {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="summaries_strict_9")
    parser.add_argument("--output-dir", default="results-factuality-evaluation-vllm")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-row-retries", type=int, default=3)
    args = parser.parse_args()

    pipeline = RetryVLLMPipeline(model_name=args.model, base_url=args.url, api_key=args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No CSVs found matching {os.path.join(args.csv_dir, args.pattern)}")

    for csv_path in files:
        run_file(pipeline, csv_path, args.output_dir, args.resume, args.max_row_retries)

    print(f"\nAll files done. Results in {args.output_dir}")