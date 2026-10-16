import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PROMPT_FILES = [
    "flu_rate_explain_auto_cot.txt",
    "flu_rate_explain.txt",
]

RATINGS = [1, 2, 3, 4, 5]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, summary: str) -> str:
    return template.replace("{{Summary}}", summary)


def token_to_rating(token: str) -> Optional[int]:
    cleaned = token.strip()
    cleaned = cleaned.replace("*", "").replace("`", "")
    m = re.search(r"[1-5]", cleaned)
    if not m:
        return None
    return int(m.group(0))


def aggregate_rating_probs(first_token_info: Any) -> Dict[int, float]:
    rating_probs: Dict[int, float] = {r: 0.0 for r in RATINGS}

    candidates: List[Tuple[str, float]] = []
    chosen_token = getattr(first_token_info, "token", "")
    chosen_lp = getattr(first_token_info, "logprob", None)
    if chosen_lp is not None:
        candidates.append((chosen_token, float(chosen_lp)))

    for alt in getattr(first_token_info, "top_logprobs", None) or []:
        tok = getattr(alt, "token", "")
        lp = getattr(alt, "logprob", None)
        if lp is None:
            continue
        candidates.append((tok, float(lp)))

    for tok, lp in candidates:
        rating = token_to_rating(tok)
        if rating is None:
            continue
        rating_probs[rating] += math.exp(lp)

    total_mass = sum(rating_probs.values())
    if total_mass > 0:
        for r in RATINGS:
            rating_probs[r] /= total_mass

    return rating_probs


def extract_rating_probs_from_logprobs(content: Any) -> Dict[int, float]:
    for token_info in content or []:
        probs = aggregate_rating_probs(token_info)
        if sum(probs.values()) > 0:
            return probs
    return {r: 0.0 for r in RATINGS}


def expected_score(rating_probs: Dict[int, float]) -> float:
    return sum(float(r) * rating_probs.get(r, 0.0) for r in RATINGS)


def argmax_rating(rating_probs: Dict[int, float]) -> Optional[int]:
    best_rating = None
    best_prob = -1.0
    for r in RATINGS:
        p = rating_probs.get(r, 0.0)
        if p > best_prob:
            best_prob = p
            best_rating = r
    if best_prob <= 0:
        return None
    return best_rating


def fallback_one_hot_from_text(text: Optional[str]) -> Dict[int, float]:
    probs = {r: 0.0 for r in RATINGS}
    if not text:
        return probs
    m = re.search(r"[1-5]", text)
    if not m:
        return probs
    probs[int(m.group(0))] = 1.0
    return probs


def parse_rating_and_explanation(raw_text: Optional[str]) -> Tuple[Optional[int], str]:
    if not raw_text:
        return None, ""

    text = str(raw_text).strip()
    if not text:
        return None, ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ""

    rating = token_to_rating(lines[0])

    if len(lines) <= 1:
        return rating, ""

    explanation = " ".join(lines[1:]).strip()
    explanation = re.sub(r"^Begruendung\s*:\s*", "", explanation, flags=re.IGNORECASE)
    explanation = re.sub(r"^Begründung\s*:\s*", "", explanation, flags=re.IGNORECASE)
    return rating, explanation


def should_retry(error_text: str) -> bool:
    e = error_text.lower()
    return (
        "429" in e
        or "rate limit" in e
        or "timeout" in e
        or "temporarily unavailable" in e
        or "503" in e
        or "502" in e
    )


def judge_once(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
    top_logprobs: int,
    max_tokens: int,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Du bist ein praeziser Bewerter von Zusammenfassungen."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
            )

            choice = resp.choices[0]
            raw_text = getattr(choice.message, "content", None)

            rating_probs = {r: 0.0 for r in RATINGS}
            used_fallback = False

            logprobs_obj = getattr(choice, "logprobs", None)
            content = getattr(logprobs_obj, "content", None) if logprobs_obj is not None else None
            if content and len(content) > 0:
                rating_probs = extract_rating_probs_from_logprobs(content)

            if sum(rating_probs.values()) == 0:
                rating_probs = fallback_one_hot_from_text(raw_text)
                used_fallback = True

            exp_score = expected_score(rating_probs)
            best = argmax_rating(rating_probs)

            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

            return {
                "ok": True,
                "raw_output": raw_text,
                "rating_probs": rating_probs,
                "expected_rating": exp_score,
                "argmax_rating": best,
                "used_text_fallback": used_fallback,
                "usage": usage_dict,
                "error": None,
            }

        except Exception as exc:
            err = str(exc)
            if attempt < max_retries - 1 and should_retry(err):
                wait_s = 2 ** attempt
                time.sleep(wait_s)
                continue
            return {
                "ok": False,
                "raw_output": None,
                "rating_probs": {r: 0.0 for r in RATINGS},
                "expected_rating": None,
                "argmax_rating": None,
                "used_text_fallback": False,
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "error": err,
            }

    return {
        "ok": False,
        "raw_output": None,
        "rating_probs": {r: 0.0 for r in RATINGS},
        "expected_rating": None,
        "argmax_rating": None,
        "used_text_fallback": False,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "error": "Unknown failure",
    }


def read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=None, engine="python", dtype=str, on_bad_lines="warn")


def extract_model_name(csv_name: str) -> str:
    name = csv_name.replace(".csv", "")
    for prefix in ["short_degraded_", "long_degraded_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge fluency evaluation for German summaries"
    )
    parser.add_argument("--input-dir", default="summaries_strict_9")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional CSV filenames or relative paths inside input-dir to evaluate",
    )
    parser.add_argument(
        "--prompts-dir",
        default="A-Closer-Look-To-LLM-Evaluation/prompts/topical_chat_german"
    )
    parser.add_argument("--output-dir", default="fluency/fluency_evaluation_results_llama-3.1-8b")
    parser.add_argument("--model", default="meta-llama-3.1-8b-instruct")
    parser.add_argument("--api-key", default="ce64487341616c6d2455a926db59e9c4")
    parser.add_argument("--base-url", default="https://chat-ai.academiccloud.de/v1")
    parser.add_argument("--summary-column", default="summary")
    parser.add_argument("--limit", type=int, default=0, help="Max rows per CSV (0 = all)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip summaries already in output files")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Set ACADEMIC_CLOUD_API_KEY or pass --api-key.")
        return 2

    input_dir = Path(args.input_dir)
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_records: List[Dict[str, Any]] = []
    seen_keys: set = set()

    detailed_jsonl_path = output_dir / "fluency_evaluation_detailed.jsonl"

    if args.resume and detailed_jsonl_path.exists():
        with detailed_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                detailed_records.append(rec)
                seen_keys.add(
                    (
                        rec.get("summary_model", ""),
                        rec.get("filename", ""),
                        rec.get("profile", ""),
                        rec.get("prompt_name", ""),
                    )
                )
        print(f"Resume: loaded {len(detailed_records)} existing detailed records")

    csv_files = sorted(input_dir.glob("*.csv"))
    if args.files:
        requested_files = [Path(f) for f in args.files]
        requested_names = {f.name for f in requested_files}
        csv_files = [c for c in csv_files if c.name in requested_names or c in requested_files]

        missing = [str(f) for f in requested_files if not any(c.name == f.name for c in csv_files)]
        if missing:
            print(f"Warning: requested files not found: {', '.join(missing)}")

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return 1

    prompt_templates: Dict[str, str] = {}
    for prompt_name in PROMPT_FILES:
        p = prompts_dir / prompt_name
        if not p.exists():
            print(f"Missing prompt file: {p}")
            return 1
        prompt_templates[prompt_name] = read_text(p)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        model_name = extract_model_name(csv_path.name)
        try:
            df = read_csv_flexible(csv_path)
        except Exception as exc:
            print(f"  Failed to read CSV: {exc}")
            continue

        if args.summary_column not in df.columns:
            print(f"  Skipping file: missing column '{args.summary_column}'")
            continue

        if args.limit > 0:
            df = df.head(args.limit)

        total_jobs = len(df) * len(PROMPT_FILES)
        pbar = tqdm(total=total_jobs, desc=f"Judging {csv_path.name}")

        for row_idx, row in df.iterrows():
            summary = str(row.get(args.summary_column, "") or "")
            filename = str(row.get("filename", f"row_{row_idx}"))
            profile = str(row.get("profile", ""))
            degraded = str(row.get("degraded", ""))
            new_records: List[Dict[str, Any]] = []

            for prompt_name in PROMPT_FILES:
                prompt_key = (model_name, filename, profile, prompt_name)
                if args.resume and prompt_key in seen_keys:
                    pbar.update(1)
                    continue

                template = prompt_templates[prompt_name]
                judge_prompt = fill_prompt(template, summary)

                result = judge_once(
                    client=client,
                    model=args.model,
                    prompt=judge_prompt,
                    max_retries=args.max_retries,
                    top_logprobs=args.top_logprobs,
                    max_tokens=args.max_tokens,
                )

                usage = result.get("usage", {})
                if usage.get("prompt_tokens") is not None:
                    total_prompt_tokens += usage["prompt_tokens"]
                if usage.get("completion_tokens") is not None:
                    total_completion_tokens += usage["completion_tokens"]

                parsed_rating, parsed_explanation = parse_rating_and_explanation(result["raw_output"])

                # Classify fluency degradation type from the 'degraded' column
                has_fluency_mild = "fluency_mild" in degraded
                has_fluency_full = "fluency_full" in degraded
                if has_fluency_full:
                    fluency_label = "fluency_full"
                elif has_fluency_mild:
                    fluency_label = "fluency_mild"
                else:
                    fluency_label = "none"

                record = {
                    "filename": filename,
                    "profile": profile,
                    "degraded": degraded,
                    "fluency_label": fluency_label,
                    "summary_model": model_name,
                    "prompt_name": prompt_name,
                    "expected_rating": result["expected_rating"],
                    "parsed_rating": parsed_rating,
                    "rating_1_prob": result["rating_probs"][1],
                    "rating_2_prob": result["rating_probs"][2],
                    "rating_3_prob": result["rating_probs"][3],
                    "rating_4_prob": result["rating_probs"][4],
                    "rating_5_prob": result["rating_probs"][5],
                    "explanation": parsed_explanation,
                    "judge_status": "ok" if result["ok"] else "error",
                    "judge_error": result["error"],
                    "token_usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                }
                detailed_records.append(record)
                new_records.append(record)
                seen_keys.add(prompt_key)
                pbar.update(1)

            if new_records:
                with detailed_jsonl_path.open("a", encoding="utf-8") as f:
                    for rec in new_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            detailed_df = pd.DataFrame(detailed_records)
            detailed_csv_path = output_dir / "fluency_evaluation_detailed.csv"
            detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8")

        pbar.close()

    if not detailed_records:
        print("No results produced.")
        return 1

    detailed_csv = output_dir / "fluency_evaluation_detailed.csv"
    summary_csv = output_dir / "fluency_evaluation_summary.csv"
    detailed_df = pd.DataFrame(detailed_records)

    if not detailed_df.empty:
        summary_df = (
            detailed_df.groupby(["summary_model", "prompt_name", "fluency_label"], dropna=False)
            .agg(
                n=("filename", "count"),
                mean_expected_rating=("expected_rating", "mean"),
                mean_rating_1_prob=("rating_1_prob", "mean"),
                mean_rating_2_prob=("rating_2_prob", "mean"),
                mean_rating_3_prob=("rating_3_prob", "mean"),
                mean_rating_4_prob=("rating_4_prob", "mean"),
                mean_rating_5_prob=("rating_5_prob", "mean"),
            )
            .reset_index()
        )
    else:
        summary_df = pd.DataFrame(columns=[
            "summary_model", "prompt_name", "fluency_label", "n",
            "mean_expected_rating", "mean_rating_1_prob", "mean_rating_2_prob",
            "mean_rating_3_prob", "mean_rating_4_prob", "mean_rating_5_prob",
        ])

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    print("\n=== Token Usage ===")
    print(f"  Prompt tokens:     {total_prompt_tokens:>8}")
    print(f"  Completion tokens: {total_completion_tokens:>8}")
    print(f"  Total tokens:      {total_prompt_tokens + total_completion_tokens:>8}")

    print(f"\nDone.")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Detailed JSONL: {detailed_jsonl_path}")
    print(f"Summary CSV: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
