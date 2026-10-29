import argparse
import json
from typing import Dict, List, Tuple
from openai import OpenAI
from tqdm import tqdm
import re
import os
import time
import sys

class GermanSEvalVLLMPipeline:
    """
    German SEval-Ex pipeline using a self-hosted vLLM server.
    Identical logic to the academic SAIA version, only the endpoint differs.
    
    This class implements:
    1. Statement Extraction - Decomposing German text into atomic statements
    2. Verdict Reasoning - Matching and classifying statements in German
    """
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 api_key: str = "EMPTY",
                 base_url: str = "http://localhost:8000/v1",
                 method: str = "StSum_Text", 
                 chunk_size: int = 3,
                 max_text_length: int = 1000):
        
        self.model_name = model_name
        self.method = method
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.temperature = 0.0
        
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # Initialize OpenAI-compatible client for the vLLM server
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

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

    def _get_completion(self, prompt: str) -> str:
        """Helper to get chat completion from API with exponential backoff."""
        max_retries = 5
        base_delay = 2  # Start with 2 seconds
        
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
                
                # Update token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                
                return response.choices[0].message.content
            except Exception as e:
                # Catch rate limits or safety filter blocks (which often return empty/none)
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "context_length_exceeded" in error_msg:
                    delay = base_delay * (2 ** attempt)
                    print(f"\nAPI Error/Rate Limit: {e}. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                
                print(f"Non-retryable API Error: {e}")
                return ""
        return ""

    def extract_statements(self, text: str) -> List[str]:
        """Extract atomic statements from the vLLM API."""
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
                response = self._get_completion(prompt_template.format(text=chunk))
                if response:
                    statements = [s.strip().strip('"').strip('- ').strip('* ') for s in response.split('\n') if s.strip()]
                    all_statements.extend(statements)
            print(f"Extracted {len(all_statements)} statements from long summary.")
            return all_statements
        else:
            response = self._get_completion(prompt_template.format(text=text))
            if not response:
                return []
            statements = [s.strip().strip('"').strip('- ').strip('* ') for s in response.split('\n') if s.strip()]
            print(f"Extracted {len(statements)} statements from summary.")
            return statements

    def verify_statements(self, summary_statements: List[str], text: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """Verify summary statements against source text, with chunking and overlap."""
        if not summary_statements:
            return {"TP": [], "FP": [], "FN": []}, []

        # Split long source text into chunks with 10% overlap
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
        
        # Iterate through text chunks
        for chunk_idx, chunk in enumerate(text_chunks):
            # Only verify statements that haven't been proven (TP) or found as CONTRADICTED (FP) yet
            remaining_statements = [s for s in summary_statements if s not in assigned_statements]
            
            if not remaining_statements:
                break
            
            # Batch statements to max 20 per request to stay within response limits
            batch_size = 50
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
                    batch_idx=(b//batch_size)+1
                )

                response = self._get_completion(prompt)
                
                if response:
                    print(f"  [Chunk {chunk_idx}] Processed batch {b//batch_size + 1}. Verdicts in this batch: {len(response.splitlines())}")
                
                if not response:
                    print(f"Warning: No response for Batch {b//batch_size + 1} in Chunk {chunk_idx}. Splitting further...")
                    # Micro-retry logic preserved for robustness
                    smaller_batch_size = 5
                    for sub_b in range(0, len(current_batch), smaller_batch_size):
                        sub_batch = current_batch[sub_b:sub_b + smaller_batch_size]
                        sub_prompt_statements = [f"[{summary_statements.index(s)}] {s}" for s in sub_batch]
                        sub_prompt = prompt.replace("\n".join(prompt_statements), "\n".join(sub_prompt_statements))
                        sub_response = self._get_completion(sub_prompt)
                        if sub_response:
                            for line in sub_response.split('\n'):
                                self._process_verdict_line(line, summary_statements, sub_batch, assigned_statements, labels)
                    continue

                for line in response.split('\n'):
                    self._process_verdict_line(line, summary_statements, current_batch, assigned_statements, labels)
        
        # Final processing: statements that were never TP or CONTRADICTED after all chunks
        remaining_unassigned = [s for s in summary_statements if s not in assigned_statements]
        for stmt in remaining_unassigned:
            labels["FP"].append(f"{stmt}. VERDICT: FP (Not found in any transcript chunk)")

        return labels, []

    def _process_verdict_line(self, line: str, summary_statements: List[str], current_batch: List[str], assigned_statements: set, labels: Dict):
        """Helper to process a single verdict line from the vLLM API."""
        line_upper = line.upper().strip()
        
        # Check for TP or CONTRADICTED
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
                            # Clean up the line to ensure the corrected info is preserved in the output
                            clean_line = line.strip()
                            print(f"    -> Found {verdict}: {stmt[:60]}...")
                            labels[verdict].append(clean_line)
                            assigned_statements.add(stmt)
                        return
                except (ValueError, IndexError):
                    pass

            # Fallback to text matching
            for stmt in current_batch:
                if stmt.lower().strip() in line.lower():
                    if stmt not in assigned_statements:
                        clean_line = line.strip()
                        print(f"    -> Found {verdict} (via text match): {stmt[:60]}...")
                        labels[verdict].append(clean_line)
                        assigned_statements.add(stmt)
                    break

    def calculate_metrics(self, labels: Dict[str, List[str]], unevaluated_count: int = 0) -> Dict[str, float]:
        # unevaluated_count will now be 0 as we move missing statements to FP explicitly
        tp = len(labels["TP"])
        fp = len(labels["FP"]) + unevaluated_count
        # FN is only used if we are doing reference summary comparison (StSum_StRef)
        fn = len(labels.get("FN", []))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1, "tp_count": tp, "fp_count": fp}

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
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
            },
            "model": self.model_name
        }

if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="pilot_summaries.csv",help="Input CSV file with summaries and transcripts")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="Base URL of the vLLM OpenAI-compatible server")
    parser.add_argument("--api-key", default="EMPTY", help="API key (vLLM usually ignores this)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="results-factuality-evaluation", help="Output directory for results")
    parser.add_argument("--summary-column", default="summary", help="Column name containing the summary text")
    parser.add_argument("--transcript-column", default="transcript", help="Column name containing the source transcript")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output file")
    args = parser.parse_args()

    pipeline = GermanSEvalVLLMPipeline(model_name=args.model, api_key=args.api_key, base_url=args.url)

    try:
        df = pd.read_csv(args.csv, sep=None, engine='python', on_bad_lines='warn')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    base_csv_name = os.path.splitext(os.path.basename(args.csv))[0]
    model_slug = args.model.replace("/", "_").replace(":", "_")
    
    os.makedirs(args.output_dir, exist_ok=True)
        
    output_file = os.path.join(args.output_dir, f"results_{base_csv_name}_{model_slug}.json")
    
    results = []
    seen_keys = set()
    
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        seen_keys = {(r.get("filename", ""), r.get("profile", "")) for r in results}
        print(f"Resume: loaded {len(results)} existing results")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = row.get('filename', f'row_{i}')
        profile = row.get('profile', '')
        
        if args.resume and (filename, profile) in seen_keys:
            continue

        summary = str(row[args.summary_column])
        transcript = str(row[args.transcript_column])

        res = pipeline.evaluate_single(transcript, summary)
        res['filename'] = filename
        res['profile'] = profile
        
        metrics = res['metrics']
        usage = res['token_usage']
        print(f"\n--- Row {i+1}: {filename} ({profile}) ---")
        print(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")
        print(f"TP: {metrics['tp_count']}  FP: {metrics['fp_count']}")
        print(f"Token Usage: {usage['total_tokens']} (P: {usage['prompt_tokens']}, C: {usage['completion_tokens']})")
        
        results.append(res)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
                    
    print(f"\nDone. Results saved to {output_file}")