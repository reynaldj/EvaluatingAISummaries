import os
import json
import time
import random
import argparse

import pandas as pd
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from evaluate_relevance_entailment import EntailmentRelevanceEvaluator


class VllmEntailmentRelevanceEvaluator(EntailmentRelevanceEvaluator):
    """Evaluation pipeline served by a local vLLM OpenAI-compatible endpoint.

    Reuses all the prompt/logic/checkpoint code from
    EntailmentRelevanceEvaluator and only overrides the client setup
    (``__init__``) and the request/retry handling (``_safe_api_call``) so it
    works against a self-hosted vLLM server instead of the cloud endpoint.
    """

    def __init__(self, base_url=None, model=None, api_key="EMPTY", timeout=120, max_retries=3, retry_wait=2.0, no_embeddings=False):
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.model_name = model or os.environ.get("VLLM_MODEL", os.environ.get("MODEL_NAME", "meta-llama-3.1-8b-instruct"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.no_embeddings = no_embeddings
        self.fatal_error = None

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        print(f"Connecting to vLLM at {self.base_url} (model={self.model_name}, timeout={timeout}s)")

        if not self._ping():
            print("  [Warning] Could not reach the vLLM endpoint. Requests will retry and may fail.")

        if self.no_embeddings:
            print("Embeddings disabled (--no-embeddings): skipping SentenceTransformer load.")
            self.embedding_model = None
        else:
            print("Loading SentenceTransformer model...")
            try:
                self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            except Exception as exc:
                print(f"  [Warning] Failed to load intfloat/multilingual-e5-large ({exc}); falling back to all-MiniLM-L6-v2.")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.kp_cache = {}

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


    def _ping(self):
        try:
            self.client.models.list()
            return True
        except Exception as exc:
            print(f"  [Info] Endpoint check failed: {str(exc)[:120]}")
            return False

    def log_api_failure(self, error, context=None):
        """Log an API failure (with its error message) to the console only."""
        ctx = context or {}
        ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        suffix = f" ({ctx_str})" if ctx_str else ""
        print(f"    [API FAIL LOG] {type(error).__name__ if error else 'None'}: {str(error)}{suffix}")

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        """Single request with explicit timeout; retry transient errors.

        Local vLLM is not rate limited like the cloud API, so 429 is treated as
        a normal transient error instead of a hard skip. A request that does not
        return within ``self.timeout`` is retried rather than hanging forever.
        """
        timeout = timeout if timeout is not None else self.timeout
        max_attempts = max_attempts if max_attempts is not None else self.max_retries
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
                last_error = "empty response"
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if any(code in error_msg for code in ("404", "model_not_found")):
                    print(f"    [Fatal] Model '{self.model_name}' not found on the vLLM server. "
                          f"Check VLLM_MODEL / --model. (attempt {attempt+1}/{max_attempts})")
                    self.fatal_error = last_error
                    self.log_api_failure(last_error, {"phase": "model_not_found"})
                    break
                wait = self.retry_wait * (attempt + 1) + random.uniform(0, 0.5)
                print(f"    [API Error] {type(e).__name__}: {str(e)[:120]}  (retry in {wait:.0f}s, attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)

        print(f"    [API FAILED] ({self.model_name}): {last_error}")
        self.log_api_failure(last_error, {"phase": "api_call"})
        return None


def read_csv_smart(csv_path):
    """Read CSV auto-detecting the delimiter (supports both ';' and ',' files)."""
    for sep in (',', ';'):
        try:
            df = pd.read_csv(csv_path, sep=sep, dtype=str, on_bad_lines='warn')
            if 'summary' in df.columns and 'transcript' in df.columns:
                return df
        except Exception:
            continue
    return pd.read_csv(csv_path, sep=None, engine='python', dtype=str, on_bad_lines='warn')


def _row_has_failed_items(result):
    """A row is retryable if any key item was marked api_failed."""
    for item in result.get("key_items", []) or []:
        if item.get("api_failed"):
            return True
        reason = str(item.get("reason", "") or "")
        if reason.startswith("[API FAILED]") or reason.startswith("[PARSE FAILED]") or reason.startswith("[FAILED]"):
            return True
    return False


def process_csv(csv_path, evaluator, output_dir, limit=None, filenames=None, sample=None, force=False, retry_failed=False):
    if not os.path.exists(csv_path):
        return
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    summary_model = base_name
    for prefix in ["short_degraded_", "long_degraded_"]:
        if summary_model.startswith(prefix):
            summary_model = summary_model[len(prefix):]
            break
    output_path = os.path.join(output_dir, f"entailment_eval_{base_name}.json")

    results = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}
                print(f"Loaded {len(results)} existing results for {base_name}")
        except Exception:
            pass

    if retry_failed:
        failed_ids = {
            f"{r['filename']}_{r.get('profile', '')}"
            for r in results
            if _row_has_failed_items(r)
        }
        n_kept = len(results) - len(failed_ids)
        results = [r for r in results if not _row_has_failed_items(r)]
        processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}
        if failed_ids:
            print(f"  [Retry] Will re-evaluate {len(failed_ids)} row(s) with API failures; keeping {n_kept} clean result(s).")
        else:
            print(f"  [Retry] No rows with API failures found.")

    df = read_csv_smart(csv_path)
    if df is None or df.empty:
        print(f"Skipping {base_name}: could not read CSV.")
        return

    if filenames:
        df = df[df['filename'].isin(filenames)]
        if force:
            filenames_set = set(filenames)
            results = [r for r in results if r.get('filename') not in filenames_set]
            processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}

    mask = df.apply(lambda r: f"{r['filename']}_{r.get('profile', '')}" not in processed_ids, axis=1)
    df_to_process = df[mask]

    if sample:
        df_to_process = df_to_process.sample(min(sample, len(df_to_process)))
    elif limit:
        df_to_process = df_to_process.head(limit)

    if len(df_to_process) == 0:
        print(f"Skipping {base_name}: No new rows.")
        return

    print(f"Processing {len(df_to_process)} rows in {base_name}...")

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        attempt = 0
        while True:
            attempt += 1
            pt_before = evaluator.total_prompt_tokens
            ct_before = evaluator.total_completion_tokens
            res = evaluator.evaluate_row(
                str(row['transcript']),
                str(row['summary']),
                row['filename'],
                profile=row.get('profile', ''),
            )
            if not _row_has_failed_items(res):
                break

            failed_items = [
                item for item in (res.get("key_items", []) or [])
                if item.get("api_failed") or str(item.get("reason", "") or "").startswith(("[API FAILED]", "[PARSE FAILED]", "[FAILED]"))
            ]
            for item in failed_items:
                evaluator.log_api_failure(
                    item.get("reason", "unknown"),
                    {
                        "phase": "row_retry",
                        "filename": row['filename'],
                        "profile": row.get('profile', ''),
                        "item": item.get("item"),
                        "attempt": attempt,
                    },
                )
            if evaluator.fatal_error:
                raise SystemExit(
                    f"[FATAL] Model '{evaluator.model_name}' was not found on the vLLM server. "
                    f"Endless retry aborted. Check VLLM_MODEL / --model. Last error: {evaluator.fatal_error}"
                )
            if attempt == 1 or attempt % 5 == 0:
                print(f"    [Endless Retry] {row['filename']} (profile={row.get('profile','')}) "
                      f"attempt {attempt} still has {len(failed_items)} failed item(s); retrying...")
            time.sleep(1.0)

        res['filename'] = row['filename']
        res['profile'] = row.get('profile', '')
        res['summary_model'] = summary_model
        res['token_usage'] = {
            "prompt_tokens": evaluator.total_prompt_tokens - pt_before,
            "completion_tokens": evaluator.total_completion_tokens - ct_before,
            "total_tokens": (evaluator.total_prompt_tokens - pt_before) + (evaluator.total_completion_tokens - ct_before),
        }
        results.append(res)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def preload_keypoints(evaluator, keypoints_dirs):
    """Load pre-extracted key points into the evaluator's kp_cache.

    Each JSON must contain entries with 'filename' and 'key_points'.
    Since the transcript is identical for all rows of a filename, keying
    by filename is sufficient (matches get_key_items_for_transcript).
    """
    loaded = 0
    for kp_dir in keypoints_dirs:
        if not kp_dir or not os.path.isdir(kp_dir):
            print(f"  [Keypoints] Directory not found, skipping: {kp_dir}")
            continue
        for kp_file in sorted(os.listdir(kp_dir)):
            if not kp_file.endswith(".json"):
                continue
            kp_path = os.path.join(kp_dir, kp_file)
            try:
                with open(kp_path, 'r', encoding='utf-8') as f:
                    kp_data = json.load(f)
                for entry in kp_data:
                    fname = entry.get("filename")
                    key_points = entry.get("key_points")
                    if fname and key_points:
                        evaluator.kp_cache[fname] = key_points
                        loaded += 1
                print(f"  [Keypoints] Loaded from {kp_file}: {len(kp_data)} entries")
            except Exception as exc:
                print(f"  [Keypoints] Failed to load {kp_file}: {exc}")
    print(f"  [Keypoints] Total {loaded} cached keypoint sets across {len(keypoints_dirs)} dir(s)")
    return loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summary completeness against a local vLLM-served LLM.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a specific CSV file or a unique substring of the filename to process. If omitted, processes all CSVs in --input-dir.")
    parser.add_argument("--filter", type=str, help="Only process files containing this substring.")
    parser.add_argument("--limit", type=int, help="Limit rows per file.")
    parser.add_argument("--sample", type=int, help="Number of random rows to sample from each file.")
    parser.add_argument("--filenames", type=str, nargs="+", help="Specific transcript filenames to process.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of specified filenames (overwrite existing results).")
    parser.add_argument("--base-url", type=str, default=None, help="vLLM OpenAI-compatible base URL (default http://localhost:8000/v1).")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Served model name (default VLLM_MODEL env or meta-llama-3.1-8b-instruct).")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds (default 120).")
    parser.add_argument("--max-retries", type=int, default=3, help="API call retries (default 3).")
    parser.add_argument("--output-dir", type=str, default="entailment_evaluation_results_vllm", help="Output directory for results JSON.")
    parser.add_argument("--input-dir", type=str, default="summaries_strict_9", help="Directory containing the summary CSVs to evaluate.")
    parser.add_argument(
        "--keypoints-dir",
        type=str,
        nargs="+",
        default=["relevance/entailment_keypoints_results_summaries_strict_9_1"],
        help="One or more directories with pre-extracted key point JSON files.",
    )
    parser.add_argument("--no-embeddings", default=True, action="store_true", help="Skip loading the SentenceTransformer model (disables dedup/clustering).")
    parser.add_argument("--retry-failed", action="store_true", help="Re-evaluate only rows whose saved results contain API-failed key items.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = VllmEntailmentRelevanceEvaluator(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        no_embeddings=args.no_embeddings,
    )

    preload_keypoints(evaluator, args.keypoints_dir)

    import atexit
    def print_usage():
        pt = evaluator.total_prompt_tokens
        ct = evaluator.total_completion_tokens
        print(f"\n=== Token Usage ===")
        print(f"  Prompt tokens:     {pt:>8}")
        print(f"  Completion tokens: {ct:>8}")
        print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    strict_folders = [args.input_dir]

    all_csvs = []
    for folder in strict_folders:
        if os.path.exists(folder):
            all_csvs.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")])
        else:
            print(f"[Warning] Input folder not found: {folder}")

    files_to_process = []
    if args.csv:
        if os.path.exists(args.csv):
            files_to_process = [args.csv]
        else:
            matches = [f for f in all_csvs if args.csv.lower() in os.path.basename(f).lower()]
            if not matches:
                print(f"No file found matching pattern: {args.csv}")
            files_to_process = matches
    else:
        files_to_process = all_csvs

    if args.filter:
        files_to_process = [f for f in files_to_process if args.filter.lower() in os.path.basename(f).lower()]

    if args.filenames:
        requested_set = set(args.filenames)
        filtered_csvs = []
        for csvf in files_to_process:
            try:
                df_check = read_csv_smart(csvf)
                names_in_csv = set(df_check['filename'].astype(str).values)
                if names_in_csv & requested_set:
                    filtered_csvs.append(csvf)
            except Exception as e:
                print(f"  [Warning] Could not inspect {os.path.basename(csvf)}: {e}")
        files_to_process = filtered_csvs
        if not files_to_process:
            print("No discovered CSVs contain the requested filenames. Exiting.")
            exit(0)

    if not files_to_process:
        print("No files discovered/matched. Exiting.")
    else:
        print(f"Queue: {[os.path.basename(f) for f in files_to_process]}")

    for csv_file in files_to_process:
        process_csv(csv_file, evaluator, args.output_dir, limit=args.limit, filenames=args.filenames, sample=args.sample, force=args.force, retry_failed=args.retry_failed)