import argparse
import glob
from html import parser
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor


class Gemma31BFactualityEvaluator:
    """
    Local Gemma 31B factuality evaluator inspired by the German SEval academic pipeline.

    It follows the same two-stage logic:
    1. Extract atomic statements from the summary.
    2. Verify each statement against the source transcript chunk-by-chunk.
    """

    def __init__(
        self,
        model_name: str = "gemma-4-31b-it",
        method: str = "StSum_Text",
        chunk_size: int = 3,
        max_text_length: int = 1000,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.method = method
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        resolved_model_name = model_name
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            alt_model = "google/gemma-4-31B-it"
            print(f"Model '{model_name}' not found locally; falling back to '{alt_model}'")
            self.processor = AutoProcessor.from_pretrained(alt_model)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                alt_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            resolved_model_name = alt_model

        self.model_name = model_name
        self.resolved_model_name = resolved_model_name

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _chat(self, prompt: str, system_prompt: str = "Du bist ein hilfreicher Assistent für die Analyse von Texten.") -> str:
        """Generate a completion using the local Gemma model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

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
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
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

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
        sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]

        chunks = []
        for i in range(0, len(sentences), self.chunk_size):
            chunk = ' '.join(sentences[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def extract_statements(self, text: str) -> List[str]:
        """Extract atomic statements from text."""
        prompt_template = """
        Extrahiere wichtige Aussagen aus dem folgenden Text. Jede Aussage sollte eine einzelne, in sich geschlossene Tatsache oder Behauptung sein. Füge keine Interpunktion oder zusätzliche Wörter hinzu. Jede Aussage sollte in einer neuen Zeile stehen.

        Beispiel:
        Text: "Albert Einstein wurde 1879 in Ulm geboren."
        Aussagen:
        "Albert Einstein wurde in Ulm geboren"
        "Albert Einstein wurde 1879 geboren"

        Text: {text}

        Extrahierte Aussagen:
        """

        if self.method == "Chunked" and len(text.split()) > self.max_text_length:
            chunks = self.split_into_chunks(text)
            all_statements = []
            for chunk in chunks:
                response = self._chat(prompt_template.format(text=chunk))
                if response:
                    statements = [s.strip().strip('"').strip('- ').strip('* ') for s in response.split('\n') if s.strip()]
                    all_statements.extend(statements)
            print(f"Extracted {len(all_statements)} statements from long summary.")
            return all_statements

        response = self._chat(prompt_template.format(text=text))
        if not response:
            return []
        statements = [s.strip().strip('"').strip('- ').strip('* ') for s in response.split('\n') if s.strip()]
        print(f"Extracted {len(statements)} statements from summary.")
        return statements

    def verify_statements(self, summary_statements: List[str], text: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """Verify a summary's statements against source text in chunks."""
        if not summary_statements:
            return {"TP": [], "FP": [], "FN": []}, []

        words = text.split()
        chunk_size = 1000
        overlap = 100

        text_chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            text_chunks.append(chunk)
            if i + chunk_size >= len(words):
                break

        labels = {"TP": [], "FP": [], "FN": []}
        assigned_statements = set()

        for chunk_idx, chunk in enumerate(text_chunks):
            remaining_statements = [s for s in summary_statements if s not in assigned_statements]
            if not remaining_statements:
                break

            batch_size = 20
            for b in range(0, len(remaining_statements), batch_size):
                current_batch = remaining_statements[b:b + batch_size]

                prompt_statements = []
                for s in current_batch:
                    global_idx = summary_statements.index(s)
                    prompt_statements.append(f"[{global_idx}] {s}")

                prompt = """
                Vergleiche die Aussagen der Zusammenfassung mit dem Originaltext-Abschnitt.
                Gib für jede Aussage an, ob sie durch den Text bestätigt, widerlegt oder nicht erwähnt wird.

                Kategorien:
                - TP (True Positive): Die Aussage wird durch diesen Abschnitt direkt gestützt.
                - CONTRADICTED: Der Abschnitt widerspricht dieser Aussage faktisch.
                  WICHTIG: Wenn eine Aussage widersprüchlich ist, gib kurz an, was die tatsächliche Information im Text ist.
                - NOT_MENTIONED: Die Information ist in diesem Abschnitt nicht enthalten.

                Format: [NUMMER] [Originale Aussage]. VERDICT: [VERDICT] (Falls CONTRADICTED: [Korrekte Info aus dem Text])

                Originaltext-Abschnitt:
                {text}

                Zu prüfende Aussagen (Batch {batch_idx}):
                {summary_statements}

                Ergebnisse:
                """.format(
                    summary_statements="\n".join(prompt_statements),
                    text=chunk,
                    batch_idx=(b // batch_size) + 1,
                )

                response = self._chat(prompt)

                if response:
                    print(f"  [Chunk {chunk_idx}] Processed batch {b // batch_size + 1}. Verdicts in this batch: {len(response.splitlines())}")

                if not response:
                    print(f"Warning: No response for Batch {b // batch_size + 1} in Chunk {chunk_idx}. Splitting further...")
                    smaller_batch_size = 5
                    for sub_b in range(0, len(current_batch), smaller_batch_size):
                        sub_batch = current_batch[sub_b:sub_b + smaller_batch_size]
                        sub_prompt_statements = [f"[{summary_statements.index(s)}] {s}" for s in sub_batch]
                        sub_prompt = prompt.replace("\n".join(prompt_statements), "\n".join(sub_prompt_statements))
                        sub_response = self._chat(sub_prompt)
                        if sub_response:
                            for line in sub_response.split('\n'):
                                self._process_verdict_line(line, summary_statements, sub_batch, assigned_statements, labels)
                    continue

                for line in response.split('\n'):
                    self._process_verdict_line(line, summary_statements, current_batch, assigned_statements, labels)

        remaining_unassigned = [s for s in summary_statements if s not in assigned_statements]
        for stmt in remaining_unassigned:
            labels["FP"].append(f"{stmt}. VERDICT: FP (Not found in any transcript chunk)")

        return labels, []

    def _process_verdict_line(self, line: str, summary_statements: List[str], current_batch: List[str], assigned_statements: set, labels: Dict):
        """Turn a single model output line into TP or FP labels."""
        line_upper = line.upper().strip()

        verdict = None
        if "VERDICT: TP" in line_upper:
            verdict = "TP"
        elif "VERDICT: CONTRADICTED" in line_upper:
            verdict = "FP"

        if verdict:
            match = re.search(r'\[(\d+)\]', line)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx < len(summary_statements):
                        stmt = summary_statements[idx]
                        if stmt not in assigned_statements:
                            clean_line = line.strip()
                            print(f"    -> Found {verdict}: {stmt[:60]}...")
                            labels[verdict].append(clean_line)
                            assigned_statements.add(stmt)
                        return
                except (ValueError, IndexError):
                    pass

            for stmt in current_batch:
                if stmt.lower().strip() in line.lower():
                    if stmt not in assigned_statements:
                        clean_line = line.strip()
                        print(f"    -> Found {verdict} (via text match): {stmt[:60]}...")
                        labels[verdict].append(clean_line)
                        assigned_statements.add(stmt)
                    break

    def calculate_metrics(self, labels: Dict[str, List[str]], unevaluated_count: int = 0) -> Dict[str, float]:
        tp = len(labels["TP"])
        fp = len(labels["FP"]) + unevaluated_count
        fn = len(labels.get("FN", []))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_count": tp,
            "fp_count": fp,
        }

    def evaluate_single(self, text: str, summary: str, extract_source_statements: bool = False) -> Dict:
        summary_statements = self.extract_statements(summary)
        print("Summary Statements extracted:")
        for idx, stmt in enumerate(summary_statements):
            print(f"  [{idx}] {stmt}")

        source_statements = []
        if extract_source_statements:
            source_statements = self.extract_statements(text)
            labels, unevaluated = self.verify_statements(summary_statements, "\n".join(source_statements))
        else:
            labels, unevaluated = self.verify_statements(summary_statements, text)

        metrics = self.calculate_metrics(labels, len(unevaluated))
        return {
            "summary_statements": summary_statements,
            "verdict_reasoning": labels,
            "unevaluated_statements": unevaluated,
            "metrics": metrics,
            "token_usage": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            },
            "model": self.model_name,
        }



def run_file(pipeline, csv_path, output_dir, resume, max_row_retries, summary_column, transcript_column):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    model_slug = pipeline.model_name.replace("/", "_").replace(":", "_")
    output_file = os.path.join(output_dir, f"results_{base_name}_{model_slug}.json")

    # Resume should also work if the model name differs slightly from the file that was saved earlier.
    candidate_files = sorted(glob.glob(os.path.join(output_dir, f"results_{base_name}_*.json")))
    if resume and not os.path.exists(output_file) and candidate_files:
        output_file = candidate_files[0]
        print(f"Resume: using existing result file {os.path.basename(output_file)} for {base_name}")

    results = []
    seen = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        seen = {
            (str(r.get("filename", "")).strip(), str(r.get("profile", "")).strip())
            for r in results
        }
        print(f"Resume: loaded {len(results)} existing results for {base_name} from {os.path.basename(output_file)}")

    df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
    skipped = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc=base_name):
        filename = str(row.get("filename", f"row_{i}")).strip()
        profile = str(row.get("profile", "")).strip()
        if resume and (filename, profile) in seen:
            continue

        summary = str(row[summary_column])
        transcript = str(row[transcript_column])

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


def parse_args():
    default_csv_dir = os.path.join(os.path.dirname(__file__), "..", "summaries_strict_9")
    default_output_dir = os.path.join(os.path.dirname(__file__), "results-factuality-evaluation-academic")

    parser = argparse.ArgumentParser(description="Gemma 31B factuality evaluator built to mirror the German SEval academic pipeline.")
    parser.add_argument("--csv", type=str, default=None, help="Single CSV with transcript and summary columns")
    parser.add_argument("--csv-dir", type=str, default=default_csv_dir, help="Directory containing CSVs to evaluate")
    parser.add_argument("--summary-column", type=str, default="summary", help="Summary column name")
    parser.add_argument("--transcript-column", type=str, default="transcript", help="Transcript column name")
    parser.add_argument(
        "--model",
        type=lambda p: os.path.expanduser(p),
        default=os.path.expanduser("~/models/gemma-4-31B-it"),
        help="Gemma model ID or local model path",
    )
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output folder")
    parser.add_argument("--resume", default=True, action="store_true", help="Skip rows already in the matching output file")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files when using --csv-dir")
    parser.add_argument("--max-row-retries", type=int, default=3, help="Retry count for failed row evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = Gemma31BFactualityEvaluator(model_name=args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        run_file(
            evaluator,
            args.csv,
            args.output_dir,
            args.resume,
            args.max_row_retries,
            args.summary_column,
            args.transcript_column,
        )
    else:
        csv_dir = args.csv_dir
        if not os.path.isdir(csv_dir):
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        files = sorted(glob.glob(os.path.join(csv_dir, args.pattern)))
        if not files:
            raise SystemExit(f"No CSVs found matching {os.path.join(csv_dir, args.pattern)}")

        for csv_path in files:
            run_file(
                evaluator,
                csv_path,
                args.output_dir,
                args.resume,
                args.max_row_retries,
                args.summary_column,
                args.transcript_column,
            )

    print(f"\nAll files done. Results in {args.output_dir}")
