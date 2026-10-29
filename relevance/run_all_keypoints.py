import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from extract_keypoints_entailment import EntailmentKeyPointExtractor


class TeeStream:
    """Write process output to both the original stream and the run log."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, message):
        self.original.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()

    @property
    def encoding(self):
        return self.original.encoding


def enable_file_logging(log_path):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    print(f"\n{'=' * 72}\nRun started: {datetime.now().astimezone().isoformat()}\nLog file: {os.path.abspath(log_path)}\n{'=' * 72}")
    return log_file


def transcript_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_entry(key_points, filename, profile, model, thash, summary, word_count, short,
                ent_pt_delta, ent_ct_delta, stat_pt_delta, stat_ct_delta, pt_now, ct_now, stat_pt_now, stat_ct_now):
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
        "token_usage": {
            "prompt_tokens": ent_pt_delta + stat_pt_delta,
            "completion_tokens": ent_ct_delta + stat_ct_delta,
            "total_tokens": ent_pt_delta + ent_ct_delta + stat_pt_delta + stat_ct_delta,
            "entailment_prompt_tokens": ent_pt_delta,
            "entailment_completion_tokens": ent_ct_delta,
            "statistical_prompt_tokens": stat_pt_delta,
            "statistical_completion_tokens": stat_ct_delta,
            "cumulative_prompt_tokens": pt_now + stat_pt_now,
            "cumulative_completion_tokens": ct_now + stat_ct_now,
            "cumulative_total_tokens": pt_now + ct_now + stat_pt_now + stat_ct_now,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM entailment key-point extraction, deduped per transcript, across a directory")
    parser.add_argument("--csv-dir", default="summaries_strict_8")
    parser.add_argument("--output-dir", default="entailment_keypoints_results_summaries_strict_9_llama-3.18b_narrowNew")
    parser.add_argument("--output-file", default=None, help="Output JSON filename (default: entailment_kp_dedup.json)")
    parser.add_argument("--log-file", default=None,
                        help="Log path (default: <output-dir>/run_all_keypoints.log)")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument("--model", type=str, default="meta-llama-3.1-8b-instruct", help="LLM model name")
    parser.add_argument("--llm-only",default=True, action="store_true",
                        help="Force the LLM entailment pipeline for ALL transcripts (disable the statistical fallback for low-quality ones)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file or "keypoints_gemma.json")
    log_path = args.log_file or os.path.join(args.output_dir, "run_all_keypoints.log")
    log_file = enable_file_logging(log_path)

    import atexit
    atexit.register(log_file.flush)

    extractor = EntailmentKeyPointExtractor(model_name=args.model)

    if args.llm_only:
        extractor._assess_transcript_quality = lambda t: ("high", 0.0)
        print("LLM-only mode: forcing the entailment pipeline for every transcript (statistical fallback disabled)")

    def print_usage():
        pt = extractor.total_prompt_tokens
        ct = extractor.total_completion_tokens
        stat = getattr(extractor, '_stat_extractor', None)
        spt = stat.total_prompt_tokens if stat is not None else 0
        sct = stat.total_completion_tokens if stat is not None else 0
        if pt + ct + spt + sct > 0:
            print(f"\n=== Token Usage ===")
            print(f"  Prompt tokens:     {pt + spt:>8}  (entailment {pt}, statistical {spt})")
            print(f"  Completion tokens: {ct + sct:>8}  (entailment {ct}, statistical {sct})")
            print(f"  Total tokens:      {pt + ct + spt + sct:>8}")
    atexit.register(print_usage)

    results = []
    emitted = set()
    cache = {}
    if not args.force and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        emitted = {r["transcript_hash"] for r in results if r.get("transcript_hash")}
        cache = {r["transcript_hash"]: r.get("key_points", []) for r in results if r.get("transcript_hash")}
        extractor.kp_cache.update(cache)
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
            short = word_count > 2000

            pt_before = extractor.total_prompt_tokens
            ct_before = extractor.total_completion_tokens
            stat = getattr(extractor, '_stat_extractor', None)
            stat_pt_before = stat.total_prompt_tokens if stat is not None else 0
            stat_ct_before = stat.total_completion_tokens if stat is not None else 0

            if thash in cache:
                print(f"  [Cache] Reusing key points for transcript {filename} ({base_name})")
                key_points = cache[thash]
            else:
                print(f"\n{'='*60}")
                print(f"Extracting key points (LLM): {filename} ({base_name})")
                print(f"{'='*60}")
                extractor.kp_cache.pop(filename, None)
                key_points = extractor.get_key_items_for_transcript(transcript, filename, short=short)
                cache[thash] = key_points

            pt_now = extractor.total_prompt_tokens
            ct_now = extractor.total_completion_tokens
            stat = getattr(extractor, '_stat_extractor', None)
            stat_pt_now = stat.total_prompt_tokens if stat is not None else stat_pt_before
            stat_ct_now = stat.total_completion_tokens if stat is not None else stat_ct_before

            entry = build_entry(
                key_points, filename, profile, model, thash, summary, word_count, short,
                pt_now - pt_before, ct_now - ct_before,
                stat_pt_now - stat_pt_before, stat_ct_now - stat_ct_before,
                pt_now, ct_now, stat_pt_now, stat_ct_now,
            )
            results.append(entry)
            emitted.add(thash)
            new_entries += 1

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {new_entries} new transcripts added. {len(results)} entries total ({len(emitted)} unique transcripts).")
    print(f"Results saved to: {output_file}")
