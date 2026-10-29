import argparse
import glob
import hashlib
import json
import os

import pandas as pd
from tqdm import tqdm

from extract_keypoints_statistical import StatisticalKeyPointExtractor


def transcript_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_entry(extraction, filename, profile, model, thash, summary, word_count, pt_delta, ct_delta, pt_now, ct_now):
    entry = dict(extraction)
    entry.update(
        filename=filename,
        profile=profile,
        model=model,
        transcript_hash=thash,
        variant="long" if word_count > 2000 else "short",
        summary=summary[:200] + "..." if len(summary) > 200 else summary,
    )
    entry["token_usage"] = {
        "prompt_tokens": pt_delta,
        "completion_tokens": ct_delta,
        "total_tokens": pt_delta + ct_delta,
        "cumulative_prompt_tokens": pt_now,
        "cumulative_completion_tokens": ct_now,
        "cumulative_total_tokens": pt_now + ct_now,
    }
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical key-point extraction, deduped per transcript, across a directory")
    parser.add_argument("--csv-dir", default="summaries_strict_9")
    parser.add_argument("--output-dir", default="statistical_keypoints_results_summaries_strict_9")
    parser.add_argument("--output-file", default=None, help="Output JSON filename (default: statistical_kp_dedup.json)")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM compression step")
    parser.add_argument("--llm-curation-mode", choices=["none", "curate", "curate_informative"], default=None)
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--target-pct", type=float, default=0.05)
    parser.add_argument("--seg-threshold", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="meta-llama-3.1-8b-instruct", help="LLM model name (compression/curation)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file or "statistical_kp_dedup.json")

    extractor = StatisticalKeyPointExtractor(
        llm_model_name=args.model,
        embedding_model_name=args.embedding_model,
        use_llm_compression=not args.no_llm,
        llm_curation_mode=args.llm_curation_mode,
        mmr_lambda=args.mmr_lambda,
        target_percentage=args.target_pct,
        seg_threshold=args.seg_threshold,
    )

    import atexit
    def print_usage():
        pt = extractor.total_prompt_tokens
        ct = extractor.total_completion_tokens
        if pt + ct > 0:
            print(f"\n=== Token Usage ===")
            print(f"  Prompt tokens:     {pt:>8}")
            print(f"  Completion tokens: {ct:>8}")
            print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    results = []
    emitted = set()
    cache = {}
    if not args.force and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        emitted = {r["transcript_hash"] for r in results if r.get("transcript_hash")}
        cache = {
            r["transcript_hash"]: {"key_points": r.get("key_points", []), "metadata": r.get("metadata", {})}
            for r in results if r.get("transcript_hash")
        }
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

            pt_before = extractor.total_prompt_tokens
            ct_before = extractor.total_completion_tokens

            if thash in cache:
                print(f"  [Cache] Reusing key points for transcript {filename} ({base_name})")
                extraction = cache[thash]
            else:
                print(f"\n{'='*60}")
                print(f"Extracting key points (statistical): {filename} ({base_name})")
                print(f"{'='*60}")
                extraction = extractor.extract_key_points(transcript, filename)
                cache[thash] = extraction

            pt_now = extractor.total_prompt_tokens
            ct_now = extractor.total_completion_tokens

            word_count = extraction.get('metadata', {}).get('word_count', len(transcript.split()))
            entry = build_entry(
                extraction, filename, profile, model, thash, summary, word_count,
                pt_now - pt_before, ct_now - ct_before, pt_now, ct_now,
            )
            results.append(entry)
            emitted.add(thash)
            new_entries += 1

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {new_entries} new transcripts added. {len(results)} entries total ({len(emitted)} unique transcripts).")
    print(f"Results saved to: {output_file}")