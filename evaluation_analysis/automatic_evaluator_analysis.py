"""Analyse automatic evaluation across generation and evaluator models.

The script builds a score panel from the paired summaries in summaries_strict_9
and available factuality, coverage, coherence, and fluency evaluations.  It
then writes descriptive, paired-model, evaluator-agreement, and degradation
sensitivity results.  Run from repository root:

    python annotation_analysis/automatic_evaluator_analysis.py
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import kendalltau, spearmanr


GENERATION_MODELS = ("gpt-oss-120b", "mistral-large-3-675b-instruct-2512", "qwen3.5-122b-a10b")
TARGET_DIMENSIONS = ("factuality", "coverage", "coherence", "fluency")
AUTO_COT = {
    "coherence": "coh_rate_explain_auto_cot.txt",
    "fluency": "flu_rate_explain_auto_cot.txt",
}


def display_dimension(value):
    return value


def display_quality_group(value):
    return value


def presentation_labels(frame):
    frame = frame.copy()
    for column in ("dimension", "metric_dimension", "degradation_dimension"):
        if column in frame:
            frame[column] = frame[column].map(display_dimension)
    for column in ("profile_quality_group", "profile_a", "profile_b"):
        if column in frame:
            frame[column] = frame[column].map(display_quality_group)
    return frame


def normalized_filename(value):
    """Repair common UTF-8-as-Latin-1 damage for metadata joins only."""
    text = str(value).strip()
    if "Ã" in text or "â" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    return text


def parse_literalish_mapping(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return value
    cleaned = re.sub(r"(?i)\bnan\b", "None", text)
    cleaned = re.sub(r"(?i)\bnull\b", "None", cleaned)
    cleaned = re.sub(r"(?i)\btrue\b", "True", cleaned)
    cleaned = re.sub(r"(?i)\bfalse\b", "False", cleaned)
    try:
        parsed = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return value
    return parsed


def scalar_from_mapping(value, filename=""):
    """Normalize dict-like metadata values to a scalar string when the source emits Python dict literals."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    parsed = parse_literalish_mapping(value)
    if isinstance(parsed, dict):
        value = parsed
    if isinstance(value, dict):
        if filename:
            normalized_filename = key(filename)
            for key_name, key_value in value.items():
                if key(key_name) == normalized_filename:
                    return scalar_from_mapping(key_value, filename)
        if len(value) == 1:
            return scalar_from_mapping(next(iter(value.values())), filename)
        return ""
    return value


def key(value):
    value = scalar_from_mapping(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        if len(value) == 1:
            return key(next(iter(value.values())))
        return ""
    if isinstance(value, str):
        parsed = parse_literalish_mapping(value)
        if isinstance(parsed, dict):
            if len(parsed) == 1:
                return key(next(iter(parsed.values())))
            return ""
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    normalized = re.sub(r"(?<![a-z])relevance(?![a-z])", "coverage", normalized)
    return "" if normalized in {"", "nan", "none", "null"} else normalized


def source_from_path(path: Path):
    text = path.name.lower()
    return "long" if "long" in text else "short" if "short" in text else ""


def generator_from_path(path: Path):
    name = path.stem.lower()
    match = re.search(r"degraded_(.+?)(?:__|_meta-|_gemma-|$)", name)
    return match.group(1) if match else ""


def entry_value(value, filename):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        normalized_filename = key(filename)
        for key_name, key_value in value.items():
            if key(key_name) == normalized_filename:
                return scalar_from_mapping(key_value, filename)
        if len(value) == 1:
            return scalar_from_mapping(next(iter(value.values())), filename)
        return ""
    parsed = parse_literalish_mapping(value)
    if isinstance(parsed, dict):
        return entry_value(parsed, filename)
    return scalar_from_mapping(value, filename)


def keypoint_type_from_path(path: Path):
    match = re.search(r"(?:narrow|wide|statistical)_keypoints", path.as_posix().lower())
    return match.group(0).removesuffix("_keypoints") if match else "not_applicable"


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    ordered = np.argsort(values[valid])
    ranked = values[valid][ordered]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    indices = np.flatnonzero(valid)[ordered]
    output[indices] = np.minimum(adjusted, 1.0)
    return output


def bootstrap_mean_ci(values, n_resamples=1000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(n_resamples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_likert_mappings(path):
    mapping = pd.read_csv(path)
    thresholds = {}
    for dimension, group in mapping.groupby("dimension"):
        group = group.sort_values("bin")
        thresholds[dimension] = group["upper_score"].iloc[:-1].to_numpy(dtype=float)
    required = {"factuality", "relevance"}
    missing = required - set(thresholds)
    if missing:
        raise ValueError(f"Missing Likert mappings for: {', '.join(sorted(missing))}")
    return thresholds


def apply_likert_mappings(scores, thresholds):
    scores = scores.copy()
    scores["automatic_score_raw"] = pd.to_numeric(scores["automatic_score"], errors="coerce")
    scores["score_scale"] = "raw"
    for dimension, mapping_dimension in (("factuality", "factuality"), ("coverage", "relevance")):
        mask = scores["dimension"].eq(dimension)
        scores.loc[mask, "automatic_score"] = (
            np.digitize(scores.loc[mask, "automatic_score_raw"], thresholds[mapping_dimension], right=False) + 1
        )
        scores.loc[mask, "score_scale"] = "mapped_likert_1_5"
    return scores


def score_span_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"score_min": np.nan, "score_max": np.nan, "score_range": np.nan, "score_p05": np.nan, "score_p95": np.nan}
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return {
        "score_min": float(np.min(values)),
        "score_max": float(np.max(values)),
        "score_range": float(np.max(values) - np.min(values)),
        "score_p05": float(q05),
        "score_p95": float(q95),
    }


def score_bucket_summary(values, dimension):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if dimension in {"coverage", "coherence", "fluency"}:
        values = values[(values >= 1.0) & (values <= 5.0)]
        breaks = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = [
            ("score_bucket_1_2", "1_2"),
            ("score_bucket_2_3", "2_3"),
            ("score_bucket_3_4", "3_4"),
            ("score_bucket_4_5", "4_5"),
        ]
    else:
        values = values[(values >= 0.0) & (values <= 1.0)]
        breaks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        labels = [
            ("score_bucket_0_25", "0_25"),
            ("score_bucket_25_50", "25_50"),
            ("score_bucket_50_75", "50_75"),
            ("score_bucket_75_1", "75_1"),
        ]
    if len(values) == 0:
        return {"score_bucket_1_2": np.nan, "score_bucket_2_3": np.nan, "score_bucket_3_4": np.nan, "score_bucket_4_5": np.nan,
            "score_bucket_0_25": np.nan, "score_bucket_25_50": np.nan, "score_bucket_50_75": np.nan,
            "score_bucket_75_1": np.nan}
    output = {bucket_key: 0.0 for bucket_key, _ in labels}
    for i in range(len(breaks) - 1):
        left, right = breaks[i], breaks[i + 1]
        bucket_key, _ = labels[i]
        in_bucket = (values >= left) & (values < right)
        if i == len(breaks) - 2:
            in_bucket |= values == right
        output[bucket_key] = float(np.sum(in_bucket) / len(values))
    return output


def svg_to_png(svg_path, scale=3):
    """Rasterize the plot SVG elements into a high-resolution PNG."""
    svg_path = Path(svg_path)
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    width, height = int(float(root.attrib["width"])), int(float(root.attrib["height"]))
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)
    style = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("style"))
    base_size_match = re.search(r"text\s*\{[^}]*font-size:\s*([\d.]+)px", style)
    base_size = int(float(base_size_match.group(1))) if base_size_match else 14
    sizes = {name: int(float(value)) for name, value in re.findall(r"\.([\w-]+)\s*\{[^}]*font-size:\s*([\d.]+)px", style)}
    bold = {name for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", style) if "font-weight:bold" in body}
    font_cache = {}

    def get_font(size, is_bold=False):
        key = (size, is_bold)
        if key not in font_cache:
            font_path = Path("C:/Windows/Fonts/arialbd.ttf" if is_bold else "C:/Windows/Fonts/arial.ttf")
            font_cache[key] = ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
        return font_cache[key]

    def length(value, reference):
        return reference * float(value[:-1]) / 100 if value.endswith("%") else float(value)

    for node in root.iter():
        tag, attrs = node.tag.rsplit("}", 1)[-1], node.attrib
        if tag == "rect":
            x, y = float(attrs.get("x", 0)) * scale, float(attrs.get("y", 0)) * scale
            w, h = length(attrs.get("width", "0"), width) * scale, length(attrs.get("height", "0"), height) * scale
            fill = attrs.get("fill")
            if fill and fill != "none":
                draw.rectangle((x, y, x + w, y + h), fill=fill)
            stroke = attrs.get("stroke")
            if stroke and stroke != "none":
                draw.rectangle((x, y, x + w, y + h), outline=stroke, width=max(1, int(float(attrs.get("stroke-width", 1)) * scale)))
        elif tag == "line":
            xy = tuple(float(attrs[key]) * scale for key in ("x1", "y1", "x2", "y2"))
            draw.line(xy, fill=attrs.get("stroke", "black"), width=max(1, int(float(attrs.get("stroke-width", 1)) * scale)))
        elif tag == "circle":
            cx, cy, radius = (float(attrs[key]) * scale for key in ("cx", "cy", "r"))
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=attrs.get("fill"), outline=attrs.get("stroke"))
        elif tag == "text":
            classes = attrs.get("class", "").split()
            size = max((sizes.get(name, 0) for name in classes), default=base_size) * scale
            text = "".join(node.itertext())
            anchor = {"middle": "ms", "end": "rs"}.get(attrs.get("text-anchor"), "ls")
            x, y = float(attrs.get("x", 0)) * scale, float(attrs.get("y", 0)) * scale
            rotation = re.search(r"rotate\((-?[\d.]+)", attrs.get("transform", ""))
            selected_font = get_font(size, any(name in bold for name in classes))
            if rotation:
                layer = Image.new("RGBA", (max(1, width * scale), max(1, size * 2)), (255, 255, 255, 0))
                ImageDraw.Draw(layer).text((layer.width / 2, layer.height / 2), text, fill=attrs.get("fill", "black"), font=selected_font, anchor="mm")
                layer = layer.rotate(float(rotation.group(1)), expand=True)
                image.paste(layer, (int(x-layer.width/2), int(y-layer.height/2)), layer)
                draw = ImageDraw.Draw(image)
            else:
                draw.text((x, y), text, fill=attrs.get("fill", "black"), font=selected_font, anchor=anchor)
    image.save(svg_path.with_suffix(".png"), format="PNG", dpi=(300, 300))


def load_summary_metadata(summary_dir):
    rows = []
    for path in summary_dir.glob("*_degraded_*.csv"):
        source = source_from_path(path)
        model = generator_from_path(path)
        data = pd.read_csv(path, dtype=str)
        for _, row in data.iterrows():
            rows.append({
                "source": source, "filename": normalized_filename(row["filename"]),
                "profile": key(row["profile"]), "degraded": key(row.get("degraded", "")),
                "generation_model": model,
                "keypoint_type": "not_applicable",
            })
    metadata = pd.DataFrame(rows).drop_duplicates()
    if metadata.empty:
        raise FileNotFoundError(f"No summary CSV files found in {summary_dir}")
    return metadata


def factuality_rows(root):
    configs = {
        "llama-3.1-8b": root / "results-factuality-evaluation-llama8b",
        "llama-70b": root / "results-factuality-evaluation-llama70b-correct",
        "gemma-12b": root / "results-factuality-evaluation-gemma-12b",
        "gemma-31b": root / "results-factuality-evaluation-gemma-31b",
    }
    rows = []
    for evaluator, directory in configs.items():
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source, generator = source_from_path(path), generator_from_path(path)
            for entry in data:
                filename = normalized_filename(entry.get("filename", ""))
                rows.append({"dimension": "factuality", "evaluator_model": evaluator,
                             "filename": filename,
                             "source": source, "profile": key(entry_value(entry.get("profile", ""), filename)),
                             "generation_model": generator,
                             "keypoint_type": "not_applicable",
                             "automatic_score": entry.get("metrics", {}).get("precision"),
                             "metric_file": str(path),
                             "degraded_from_evaluator": key(entry_value(entry.get("degraded", ""), filename))})
    return rows


def coverage_rows(root):
    rows = []
    directories = [path for path in root.glob("*/entailment_evaluation_results_*_log_reconsider_*_keypoints")
                   if path.is_dir()]
    for directory in directories:
        evaluator = {"gemma12b": "gemma-12b", "gemma31b": "gemma-31b"}.get(
            directory.parent.name, directory.parent.name
        )
        keypoint_type = keypoint_type_from_path(directory)
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source = source_from_path(path)
            for entry in data:
                rows.append({"dimension": "coverage", "evaluator_model": evaluator,
                             "filename": normalized_filename(entry.get("filename", "")), "source": source,
                             "profile": key(entry.get("profile", "")),
                             "generation_model": key(entry.get("summary_model", "")),
                             "keypoint_type": keypoint_type,
                             "automatic_score": entry.get("completeness"), "metric_file": str(path)})
    return rows


def normalize_evaluator_name(name):
    text = str(name).strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"-nli$", "", text)
    mapping = {
        "gemma12b": "gemma-12b",
        "gemma31b": "gemma-31b",
        "llama70b": "llama-70b",
        "llama3.1-8b": "llama-3.1-8b",
        "llama3.1-8b-instruct": "llama-3.1-8b",
        "llama-3.1-8b-instruct": "llama-3.1-8b",
    }
    return mapping.get(text, text)


def nli_keypoint_type_from_path(path: Path):
    text = path.name.lower()
    match = re.search(r"(narrow|wide|statistical)", text)
    return match.group(1) if match else "narrow"


def nli_relevance_rows(root):
    """Load the NLI-hybrid relevance outputs as coverage scores with an explicit NLI approach tag."""
    rows = []
    directories = [path for path in root.glob("*-nli/nli_hybrid_evaluation_results_*") if path.is_dir()]
    for directory in directories:
        evaluator = normalize_evaluator_name(directory.parent.name)
        keypoint_type = nli_keypoint_type_from_path(directory)
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, list):
                continue
            source = source_from_path(path)
            for entry in data:
                rows.append({
                    "dimension": "coverage",
                    "evaluator_model": evaluator,
                    "filename": normalized_filename(entry.get("filename", "")),
                    "source": source,
                    "profile": key(entry.get("profile", "")),
                    "generation_model": key(entry.get("summary_model", "")),
                    "keypoint_type": keypoint_type,
                    "evaluation_approach": "nli",
                    "degraded_from_evaluator": key(entry.get("degraded", "")),
                    "automatic_score": entry.get("coverage_score"),
                    "metric_file": str(path),
                })
    return rows


def rating_rows(root, dimension, directories):
    rows = []
    for evaluator, path in directories.items():
        if not path.exists():
            continue
        data = pd.read_csv(path, dtype=str)
        data = data[data.get("prompt_name", "").eq(AUTO_COT[dimension])].copy()
        data["automatic_score"] = pd.to_numeric(data["expected_rating"], errors="coerce")
        data = data.dropna(subset=["automatic_score"])
        if dimension in {"coherence", "fluency"}:
            data = data[data["automatic_score"].between(1.0, 5.0)]
        for _, entry in data.iterrows():
            rows.append({"dimension": dimension, "evaluator_model": evaluator,
                         "filename": normalized_filename(entry["filename"]), "source": "",
                         "profile": key(entry["profile"]), "generation_model": key(entry.get("summary_model", "")),
                         "keypoint_type": "not_applicable",
                         "degraded_from_evaluator": key(entry.get("degraded", "")),
                         "automatic_score": entry["automatic_score"], "metric_file": str(path)})
    return rows


def rating_approach_rows(root, dimension, directories):
    """Load both the standard and chain-of-thought prompt variants."""
    rows = []
    standard_prompt = {"coherence": "coh_rate_explain.txt", "fluency": "flu_rate_explain.txt"}[dimension]
    prompt_names = {
        standard_prompt: "without_cot",
        AUTO_COT[dimension]: "with_cot",
    }
    for evaluator, path in directories.items():
        if not path.exists():
            continue
        data = pd.read_csv(path, dtype=str)
        data = data[data.get("prompt_name", "").isin(prompt_names)].copy()
        data["automatic_score"] = pd.to_numeric(data["expected_rating"], errors="coerce")
        data = data.dropna(subset=["automatic_score"])
        data = data[data["automatic_score"].between(1.0, 5.0)]
        for _, entry in data.iterrows():
            rows.append({"dimension": dimension, "evaluator_model": evaluator,
                         "filename": normalized_filename(entry["filename"]), "source": "",
                         "profile": key(entry["profile"]), "generation_model": key(entry.get("summary_model", "")),
                         "keypoint_type": "not_applicable",
                         "evaluation_approach": prompt_names[entry["prompt_name"]],
                         "degraded_from_evaluator": key(entry.get("degraded", "")),
                         "automatic_score": entry["automatic_score"], "metric_file": str(path)})
    return rows


def rating_output_availability(dimension, directories):
    """Record which evaluator output files contain usable numeric ratings."""
    rows = []
    for evaluator, path in directories.items():
        if not path.exists():
            rows.append({"dimension": dimension, "evaluator_model": evaluator, "file_exists": False,
                         "selected_prompt_rows": 0, "numeric_score_rows": 0, "status": "file_missing"})
            continue
        data = pd.read_csv(path, dtype=str)
        selected = data[data.get("prompt_name", "").eq(AUTO_COT[dimension])]
        numeric = pd.to_numeric(selected.get("expected_rating"), errors="coerce").notna().sum()
        rows.append({"dimension": dimension, "evaluator_model": evaluator, "file_exists": True,
                     "selected_prompt_rows": len(selected), "numeric_score_rows": numeric,
                     "status": "usable" if numeric else "no_numeric_expected_rating"})
    return rows


def attach_metadata(scores, metadata):
    scores = scores.copy()
    scores["automatic_score"] = pd.to_numeric(scores["automatic_score"], errors="coerce")
    # Some coherence files omit generation model and source, so use metadata
    # to resolve them from filename/profile/degradation where possible.
    known_model = scores[scores["generation_model"].notna() & scores["generation_model"].ne("")]
    known_model = known_model.merge(metadata, on=["filename", "profile", "generation_model"], how="left",
                                    suffixes=("", "_metadata"))
    missing_model = scores[~(scores["generation_model"].notna() & scores["generation_model"].ne(""))].copy()
    # Coherence outputs omit the summary model.  Match them only where a
    # filename/profile pair maps uniquely to a generated summary; ambiguity is
    # intentionally dropped rather than assigned to an arbitrary model.
    missing_model["degraded_join"] = missing_model.get("degraded_from_evaluator", "").map(key)
    metadata_for_missing = metadata.copy()
    metadata_for_missing["degraded_join"] = metadata_for_missing["degraded"].map(key)
    # The degraded label disambiguates profiles assigned to different
    # generators for the same transcript.  Retain only unique matches.
    unique_metadata = metadata_for_missing.groupby(["filename", "profile", "degraded_join"]).filter(
        lambda g: len(g) == 1
    )
    missing_model = missing_model.drop(columns=["generation_model"]).merge(
        unique_metadata, on=["filename", "profile", "degraded_join"], how="left", suffixes=("", "_metadata")
    )
    resolved = pd.concat([known_model, missing_model], ignore_index=True, sort=False)
    resolved["source"] = resolved["source"].where(resolved["source"].ne(""), resolved.get("source_metadata"))
    resolved["source"] = resolved["source"].fillna("unknown")
    resolved["source"] = resolved["source"].replace({"": "unknown", "nan": "unknown", "NaN": "unknown"})
    resolved["degraded"] = resolved.get("degraded", pd.Series(index=resolved.index, dtype=str))
    resolved["degraded"] = resolved["degraded"].fillna(resolved.get("degraded_from_evaluator", "")).fillna("")
    resolved = resolved.drop(columns=[column for column in ("source_metadata", "degraded_join") if column in resolved])
    # Do not use duplicated prompt results or unresolved/ambiguous score rows.
    # Some raw evaluation files do not populate source in metadata, especially in
    # the non-COT coherence/fluency ablations, so keep those rows under a neutral
    # placeholder instead of silently dropping them as if they were invalid.
    dedupe = ["dimension", "evaluator_model", "filename", "profile", "generation_model", "metric_file"]
    if "evaluation_approach" in resolved.columns:
        dedupe.append("evaluation_approach")
    return resolved.dropna(subset=["automatic_score"]).drop_duplicates(dedupe)


def descriptives(scores):
    group_columns = ["dimension", "evaluator_model", "keypoint_type", "source", "generation_model"]
    rows = []
    for keys, group in scores.groupby(group_columns):
        values = group["automatic_score"].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        span = score_span_summary(values)
        bucket = score_bucket_summary(values, keys[0])
        row = dict(zip(group_columns, keys)) | {
            "n": len(values), "mean": np.mean(values), "sd": np.std(values, ddof=1),
            "median": np.median(values),
            "min": np.min(values), "max": np.max(values), "score_range": np.max(values) - np.min(values),
            "score_min": span["score_min"], "score_max": span["score_max"], "score_range": span["score_range"],
            "score_p05": span["score_p05"], "score_p95": span["score_p95"],
            "ci_95_low": low, "ci_95_high": high,
            "tie_rate": 1 - group["automatic_score"].nunique() / len(group),
        }
        row["dimension"] = display_dimension(row["dimension"])
        row.update(bucket)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluator_model_descriptives(scores):
    """Summarize scores by evaluator, generation model, and dimension.

    Source type and coverage keypoint type are intentionally pooled here so
        the main evaluator comparison stays compact.  Their detailed summaries
    are written separately.
    """
    group_columns = ["dimension", "evaluator_model", "generation_model"]
    rows = []
    for keys, group in scores.groupby(group_columns):
        values = group["automatic_score"].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        span = score_span_summary(values)
        bucket = score_bucket_summary(values, keys[0])
        row = dict(zip(group_columns, keys)) | {
            "n": len(values), "mean": np.mean(values), "sd": np.std(values, ddof=1),
            "median": np.median(values),
            "score_min": span["score_min"], "score_max": span["score_max"],
            "score_range": span["score_range"], "score_p05": span["score_p05"],
            "score_p95": span["score_p95"], "ci_95_low": low, "ci_95_high": high,
        }
        row["dimension"] = display_dimension(row["dimension"])
        row.update(bucket)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_keypoint_descriptives(scores):
    """Keep coverage keypoint-type comparisons separate from the main table."""
    coverage = scores[scores["dimension"].eq("coverage")].copy()
    if coverage.empty:
        return pd.DataFrame()
    group_columns = ["dimension", "evaluator_model", "generation_model", "keypoint_type"]
    rows = []
    for keys, group in coverage.groupby(group_columns):
        values = group["automatic_score"].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        span = score_span_summary(values)
        bucket = score_bucket_summary(values, "coverage")
        row = dict(zip(group_columns, keys)) | {
            "n": len(values), "mean": np.mean(values), "sd": np.std(values, ddof=1),
            "median": np.median(values), "score_min": span["score_min"],
            "score_max": span["score_max"], "score_range": span["score_range"],
            "score_p05": span["score_p05"], "score_p95": span["score_p95"],
            "ci_95_low": low, "ci_95_high": high,
        }
        row["dimension"] = display_dimension(row["dimension"])
        row["keypoint_type"] = row["keypoint_type"]
        row.update(bucket)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluator_generation_comparison_svg(scores, output):
    """Plot generation-model means by evaluator with bootstrap confidence bars."""
    summary = evaluator_model_descriptives(scores).copy()
    summary["dimension"] = summary["dimension"].map(display_dimension)
    dimensions = [display_dimension(dimension) for dimension in TARGET_DIMENSIONS if display_dimension(dimension) in summary["dimension"].unique()]
    generation_models = [model for model in GENERATION_MODELS if model in summary["generation_model"].unique()]
    evaluators = sorted(summary["evaluator_model"].unique())
    colours = {evaluator: colour for evaluator, colour in zip(
        evaluators, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    )}
    evaluator_labels = {"gemma-12b": "G12B", "gemma-31b": "G31B", "llama-3.1-8b": "L3.1-8B", "llama-70b": "L70B"}
    width, panel_width, panel_height = 1160, 540, 410
    height = 110 + panel_height * ((len(dimensions) + 1) // 2)
    margin_left, margin_right, top = 105, 36, 86
    plot_width, plot_height = panel_width - margin_left - margin_right, panel_height - 135
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:27px}.title{font-size:36px;font-weight:bold}.panel{font-size:29px;font-weight:bold}.axis{font-size:30px}.subtitle{font-size:20px}.legend{font-size:27px}</style>',
        '<text x="20" y="30" class="title">Automatic scores by generation model and evaluator</text>',
        '<text x="20" y="66" class="subtitle">Points show means; vertical bars show bootstrap 95% confidence intervals. Source type and coverage keypoint type are pooled.</text>',
    ]
    for panel_number, dimension in enumerate(dimensions):
        panel = summary[summary["dimension"].eq(dimension)]
        panel_x = (panel_number % 2) * panel_width
        panel_y = top + (panel_number // 2) * panel_height
        x0, x1 = panel_x + margin_left, panel_x + panel_width - margin_right
        y0, y1 = panel_y + 58, panel_y + 58 + plot_height
        score_low, score_high = (1.0, 5.0) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0.0, 1.0)
        elements.append(f'<text x="{panel_x + 20}" y="{panel_y + 4}" class="panel">{html.escape(display_dimension(dimension).title())}</text>')
        for tick in np.linspace(score_low, score_high, 5 if score_high == 5 else 5):
            y = y1 - (tick - score_low) / (score_high - score_low) * plot_height
            elements.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dddddd"/>')
            elements.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:g}</text>')
        x_positions = np.linspace(x0 + 58, x1 - 58, len(generation_models))
        for position, generation_model in enumerate(generation_models):
            x = x_positions[position]
            short_name = {"gpt-oss-120b": "GPT-OSS", "mistral-large-3-675b-instruct-2512": "Mistral", "qwen3.5-122b-a10b": "Qwen"}.get(generation_model, generation_model)
            elements.append(f'<text x="{x:.1f}" y="{y1 + 24}" text-anchor="middle" class="axis">{short_name}</text>')
        for evaluator_index, evaluator in enumerate(evaluators):
            evaluator_offset = (evaluator_index - (len(evaluators) - 1) / 2) * 18
            values = []
            for generation_model in generation_models:
                row = panel[(panel["evaluator_model"] == evaluator) & (panel["generation_model"] == generation_model)]
                if row.empty:
                    values.append(None)
                else:
                    item = row.iloc[0]
                    values.append((float(item["mean"]), float(item["ci_95_low"]), float(item["ci_95_high"])))
            for position, value in enumerate(values):
                if value is None:
                    continue
                mean, ci_low, ci_high = value
                x = x_positions[position] + evaluator_offset
                y = y1 - (mean - score_low) / (score_high - score_low) * plot_height
                low_y = y1 - (ci_low - score_low) / (score_high - score_low) * plot_height
                high_y = y1 - (ci_high - score_low) / (score_high - score_low) * plot_height
                elements += [
                    f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" y2="{high_y:.1f}" stroke="{colours[evaluator]}" stroke-width="2.5"/>',
                    f'<line x1="{x - 5:.1f}" y1="{low_y:.1f}" x2="{x + 5:.1f}" y2="{low_y:.1f}" stroke="{colours[evaluator]}" stroke-width="2"/>',
                    f'<line x1="{x - 5:.1f}" y1="{high_y:.1f}" x2="{x + 5:.1f}" y2="{high_y:.1f}" stroke="{colours[evaluator]}" stroke-width="2"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colours[evaluator]}"/>',
                ]
        elements.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#555555" stroke-width="2"/>')
        elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#555555" stroke-width="2"/>')
        if panel_number == 0:
            for position, evaluator in enumerate(evaluators):
                legend_x = x0 + position * 118
                elements += [f'<circle cx="{legend_x}" cy="{panel_y + panel_height - 28}" r="8" fill="{colours[evaluator]}"/>',
                             f'<text x="{legend_x + 14}" y="{panel_y + panel_height - 20}" class="legend">{html.escape(evaluator_labels.get(evaluator, evaluator))}</text>']
    elements.append('</svg>')
    output.write_text("\n".join(elements), encoding="utf-8")


def evaluator_generation_comparison_png(scores, output, *, evaluation_approach=None):
    """Write the thesis comparison directly as an opaque PNG."""
    if evaluation_approach is not None and "evaluation_approach" in scores.columns:
        scores = scores[scores["evaluation_approach"].eq(evaluation_approach)].copy()
    if scores.empty:
        image = Image.new("RGB", (2200, 1500), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((80, 120), "Automatic scores by generation model and evaluator", fill="black", font=font)
        draw.text((80, 180), f"No rows available for evaluation_approach={evaluation_approach}.", fill="#444444", font=font)
        image.save(str(output), format="PNG", dpi=(300, 300))
        return
    summary = evaluator_model_descriptives(scores).copy()
    summary["dimension"] = summary["dimension"].map(display_dimension)
    dimensions = [display_dimension(dimension) for dimension in TARGET_DIMENSIONS if display_dimension(dimension) in summary["dimension"].unique()]
    generation_models = [model for model in GENERATION_MODELS if model in summary["generation_model"].unique()]
    evaluators = sorted(summary["evaluator_model"].unique())
    colours = dict(zip(evaluators, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    labels = {"gpt-oss-120b": "GPT-OSS", "mistral-large-3-675b-instruct-2512": "Mistral", "qwen3.5-122b-a10b": "Qwen"}
    image = Image.new("RGB", (2200, 1500), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    regular = ImageFont.truetype(str(font_path), 56) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 52) if font_path.exists() else ImageFont.load_default()
    x_label_font = ImageFont.truetype(str(font_path), 60) if font_path.exists() else small
    panel_font = ImageFont.truetype(str(bold_font_path), 54) if bold_font_path.exists() else regular
    tick_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    legend_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    title = f"Automatic scores by generation model and evaluator ({evaluation_approach})" if evaluation_approach else "Automatic scores by generation model and evaluator"
    draw.text((45, 28), title, fill="black", font=regular)
    draw.text((45, 72), "Points: means; bars: bootstrap 95% confidence intervals", fill="#444444", font=small)
    for panel_number, dimension in enumerate(dimensions):
        panel = summary[summary["dimension"].eq(dimension)]
        panel_x = 90 + (panel_number % 2) * 1060
        panel_y = 150 + (panel_number // 2) * 630
        plot_left, plot_right = panel_x + 115, panel_x + 990
        plot_top, plot_bottom = panel_y + 82, panel_y + 500
        score_low, score_high = (1, 5) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0, 1)
        draw.text((panel_x, panel_y - 12), display_dimension(dimension).title(), fill="black", font=panel_font)
        for tick in np.linspace(score_low, score_high, 5):
            y = plot_bottom - (tick - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill="#dddddd", width=5)
            draw.text((plot_left - 50, y - 12), f"{tick:g}", fill="#444444", font=tick_font)
        positions = np.linspace(plot_left + 70, plot_right - 70, len(generation_models))
        for position, model in zip(positions, generation_models):
            draw.text((position - 52, plot_bottom + 26), labels.get(model, model), fill="#444444", font=x_label_font)
        for evaluator_index, evaluator in enumerate(evaluators):
            evaluator_offset = (evaluator_index - (len(evaluators) - 1) / 2) * 28
            values = panel[panel["evaluator_model"].eq(evaluator)].set_index("generation_model").reindex(generation_models)
            for x, (_, row) in zip(positions, values.iterrows()):
                x = x + evaluator_offset
                mean, low, high = row["mean"], row["ci_95_low"], row["ci_95_high"]
                if not np.isfinite([mean, low, high]).all():
                    continue
                y = plot_bottom - (mean - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                low_y = plot_bottom - (low - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                high_y = plot_bottom - (high - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                colour = colours[evaluator]
                draw.line((x, high_y, x, low_y), fill=colour, width=5)
                draw.line((x - 9, high_y, x + 9, high_y), fill=colour, width=5)
                draw.line((x - 9, low_y, x + 9, low_y), fill=colour, width=5)
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colour)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#555555", width=5)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#555555", width=5)
    legend_x, legend_y = 100, 1430
    for evaluator in evaluators:
        draw.ellipse((legend_x, legend_y, legend_x + 30, legend_y + 30), fill=colours[evaluator])
        draw.text((legend_x + 42, legend_y - 12), evaluator, fill="#333333", font=legend_font)
        legend_x += 340
    image.save(str(output), format="PNG", dpi=(300, 300))


def coverage_keypoint_comparison_png(scores, output, *, evaluation_approach=None):
    """Plot coverage scores separately for narrow, wide, and statistical keypoints."""
    if evaluation_approach is not None and "evaluation_approach" in scores.columns:
        scores = scores[scores["evaluation_approach"].eq(evaluation_approach)].copy()
    summary = coverage_keypoint_descriptives(scores)
    if summary.empty:
        image = Image.new("RGB", (2200, 1500), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        title = "Coverage scores by keypoint type" if evaluation_approach is None else f"Coverage scores by keypoint type ({evaluation_approach})"
        draw.text((80, 120), title, fill="black", font=font)
        draw.text((80, 180), "No rows available for this subset.", fill="#444444", font=font)
        image.save(str(output), format="PNG", dpi=(300, 300))
        return
    keypoint_types = ["narrow", "wide", "statistical"]
    generation_models = [model for model in GENERATION_MODELS if model in summary["generation_model"].unique()]
    evaluators = sorted(summary["evaluator_model"].unique())
    colours = dict(zip(evaluators, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    labels = {"gpt-oss-120b": "GPT-OSS", "mistral-large-3-675b-instruct-2512": "Mistral", "qwen3.5-122b-a10b": "Qwen"}
    image = Image.new("RGB", (2200, 1500), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    regular = ImageFont.truetype(str(font_path), 56) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 52) if font_path.exists() else ImageFont.load_default()
    x_label_font = ImageFont.truetype(str(font_path), 60) if font_path.exists() else small
    panel_font = ImageFont.truetype(str(bold_font_path), 54) if bold_font_path.exists() else regular
    tick_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    legend_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    title = "Coverage scores by keypoint type" if evaluation_approach is None else f"Coverage scores by keypoint type ({evaluation_approach})"
    draw.text((45, 28), title, fill="black", font=regular)
    draw.text((45, 72), "Points: means; bars: bootstrap 95% confidence intervals", fill="#444444", font=small)
    for panel_number, keypoint_type in enumerate(keypoint_types):
        panel = summary[summary["keypoint_type"].eq(keypoint_type)]
        panel_evaluators = sorted(panel["evaluator_model"].dropna().unique())
        panel_x = 90 + (panel_number % 2) * 1060
        panel_y = 150 + (panel_number // 2) * 630
        plot_left, plot_right = panel_x + 115, panel_x + 990
        plot_top, plot_bottom = panel_y + 82, panel_y + 500
        draw.text((panel_x, panel_y - 12), keypoint_type.title(), fill="black", font=panel_font)
        score_low, score_high = 1.0, 5.0
        for tick in np.linspace(score_low, score_high, 5):
            y = plot_bottom - (tick - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill="#dddddd", width=2)
            draw.text((plot_left - 50, y - 12), f"{tick:.0f}", fill="#444444", font=tick_font)
        positions = np.linspace(plot_left + 70, plot_right - 70, len(generation_models))
        for position, model in zip(positions, generation_models):
            draw.text((position - 52, plot_bottom + 26), labels.get(model, model), fill="#444444", font=x_label_font)
        for evaluator_index, evaluator in enumerate(panel_evaluators):
            evaluator_offset = (evaluator_index - (len(panel_evaluators) - 1) / 2) * 28
            values = panel[panel["evaluator_model"].eq(evaluator)].set_index("generation_model").reindex(generation_models)
            points = []
            for x, (_, row) in zip(positions, values.iterrows()):
                x = x + evaluator_offset
                mean, low, high = row["mean"], row["ci_95_low"], row["ci_95_high"]
                if not np.isfinite([mean, low, high]).all():
                    continue
                y = plot_bottom - (mean - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                low_y = plot_bottom - (low - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                high_y = plot_bottom - (high - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                colour = colours[evaluator]
                draw.line((x, high_y, x, low_y), fill=colour, width=5)
                draw.line((x - 9, high_y, x + 9, high_y), fill=colour, width=5)
                draw.line((x - 9, low_y, x + 9, low_y), fill=colour, width=5)
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colour)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#555555", width=5)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#555555", width=5)
    legend_x, legend_y = 100, 1430
    for evaluator in evaluators:
        draw.ellipse((legend_x, legend_y, legend_x + 30, legend_y + 30), fill=colours[evaluator])
        draw.text((legend_x + 42, legend_y - 12), evaluator, fill="#333333", font=legend_font)
        legend_x += 370
    image.save(output, format="PNG", dpi=(300, 300))


def cot_approach_comparison_png(scores, dimension, output):
    """Plot standard and COT score distributions for one rating dimension.

    Only matched evaluator/model pairs with both prompt variants are shown. This
    avoids misleading figures when one approach is absent for a subset of the raw
    evaluation files.
    """
    panel = scores[scores["dimension"].eq(dimension)].copy()
    if panel.empty:
        return
    group_columns = ["evaluation_approach", "evaluator_model", "generation_model"]
    summary = (panel.groupby(group_columns, dropna=False)["automatic_score"]
               .agg(mean="mean", n="size", sd="std").reset_index())
    intervals = []
    for keys, group in panel.groupby(group_columns, dropna=False):
        ci_low, ci_high = bootstrap_mean_ci(group["automatic_score"])
        intervals.append(dict(zip(group_columns, keys)) | {"ci_95_low": ci_low, "ci_95_high": ci_high})
    summary = summary.merge(pd.DataFrame(intervals), on=group_columns, how="left")
    matched = summary.pivot_table(index=["evaluator_model", "generation_model"], columns="evaluation_approach", values="mean").reset_index()
    matched = matched.dropna(subset=["without_cot", "with_cot"]).copy() if {"without_cot", "with_cot"}.issubset(set(summary["evaluation_approach"].unique())) else matched
    approaches = [approach for approach in ("without_cot", "with_cot") if approach in summary["evaluation_approach"].unique()]
    generation_models = [model for model in GENERATION_MODELS if model in matched["generation_model"].unique()] if not matched.empty else []
    evaluators = sorted(matched["evaluator_model"].dropna().unique()) if not matched.empty else []
    if not evaluators or not generation_models:
        image = Image.new("RGB", (2200, 1500), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((80, 120), f"{dimension.title()} scores by reasoning approach", fill="black", font=font)
        draw.text((80, 180), "No matched evaluator/model pairs are available for both prompt variants.", fill="#444444", font=font)
        image.save(str(output), format="PNG", dpi=(300, 300))
        return
    colours = dict(zip(evaluators, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    labels = {"gpt-oss-120b": "GPT-OSS", "mistral-large-3-675b-instruct-2512": "Mistral", "qwen3.5-122b-a10b": "Qwen"}
    approach_labels = {"without_cot": "Without COT", "with_cot": "With COT"}
    image = Image.new("RGB", (2200, 1500), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    regular = ImageFont.truetype(str(font_path), 56) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 52) if font_path.exists() else ImageFont.load_default()
    x_label_font = ImageFont.truetype(str(font_path), 60) if font_path.exists() else small
    panel_font = ImageFont.truetype(str(bold_font_path), 54) if bold_font_path.exists() else regular
    tick_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    legend_font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
    draw.text((45, 28), f"{dimension.title()} scores by reasoning approach", fill="black", font=regular)
    draw.text((45, 72), "Points show matched means; bars show bootstrap 95% confidence intervals.", fill="#444444", font=small)
    score_low, score_high = 1, 5
    for panel_number, approach in enumerate(approaches):
        panel_x = 90 + (panel_number % 2) * 1060
        panel_y = 150 + (panel_number // 2) * 630
        plot_left, plot_right = panel_x + 115, panel_x + 990
        plot_top, plot_bottom = panel_y + 82, panel_y + 500
        draw.text((panel_x, panel_y - 12), approach_labels[approach], fill="black", font=panel_font)
        for tick in np.linspace(score_low, score_high, 5):
            y = plot_bottom - (tick - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill="#dddddd", width=5)
            draw.text((plot_left - 50, y - 12), f"{tick:g}", fill="#444444", font=tick_font)
        positions = np.linspace(plot_left + 70, plot_right - 70, len(generation_models))
        for position, model in zip(positions, generation_models):
            draw.text((position - 52, plot_bottom + 26), labels.get(model, model), fill="#444444", font=x_label_font)
        for evaluator_index, evaluator in enumerate(evaluators):
            subset = matched[(matched["evaluator_model"] == evaluator)].set_index("generation_model").reindex(generation_models)
            evaluator_offset = (evaluator_index - (len(evaluators) - 1) / 2) * 28
            for x, model in zip(positions, generation_models):
                row = subset.loc[model]
                if pd.isna(row.get(approach)):
                    continue
                mean = float(row[approach])
                interval = summary[
                    summary["evaluation_approach"].eq(approach)
                    & summary["evaluator_model"].eq(evaluator)
                    & summary["generation_model"].eq(model)
                ].iloc[0]
                ci_low, ci_high = float(interval["ci_95_low"]), float(interval["ci_95_high"])
                x += evaluator_offset
                y = plot_bottom - (mean - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                low_y = plot_bottom - (ci_low - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                high_y = plot_bottom - (ci_high - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                colour = colours[evaluator]
                draw.line((x, high_y, x, low_y), fill=colour, width=5)
                draw.line((x - 9, high_y, x + 9, high_y), fill=colour, width=5)
                draw.line((x - 9, low_y, x + 9, low_y), fill=colour, width=5)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colours[evaluator])
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#555555", width=5)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#555555", width=5)
    legend_x, legend_y = 100, 1430
    for evaluator in evaluators:
        draw.ellipse((legend_x, legend_y, legend_x + 30, legend_y + 30), fill=colours[evaluator])
        draw.text((legend_x + 42, legend_y - 12), evaluator, fill="#333333", font=legend_font)
        legend_x += 340
    image.save(str(output), format="PNG", dpi=(300, 300))


def adjusted_generation_model_comparison(scores):
    """Compare generators while controlling for profile and source type.

    The three generators summarize the same transcripts, but their degradation
    profiles are not generally identical for a transcript.  A direct paired
    test would therefore confound generator and degradation.  This OLS model
    includes profile and source fixed effects and clusters inference by source
    transcript.
    """
    rows = []
    reference = GENERATION_MODELS[0]
    for keys, group in scores.groupby(["dimension", "evaluator_model", "keypoint_type"]):
        group = group.dropna(subset=["automatic_score", "generation_model", "profile", "source", "filename"])
        if group["generation_model"].nunique() < 2 or len(group) < 20:
            continue
        try:
            model = smf.ols(
                f"automatic_score ~ C(generation_model, Treatment(reference='{reference}')) + C(profile) + C(source)",
                data=group,
            ).fit(cov_type="cluster", cov_kwds={"groups": group["filename"]}) # tells that when calculating uncertainty, allow observations within same cluster to be correlated, with cluster being "filenamey"
        except (ValueError, np.linalg.LinAlgError):
            continue
        for term in model.params.index:
            if "generation_model" not in term:
                continue
            comparison = re.search(r"\[T\.(.+?)\]", term)
            rows.append({
                "dimension": keys[0], "evaluator_model": keys[1], "keypoint_type": keys[2], "reference_generation_model": reference,
                "generation_model": comparison.group(1) if comparison else term,
                "n_scores": len(group), "n_transcript_clusters": group["filename"].nunique(),
                "adjusted_mean_difference": model.params[term], "robust_se": model.bse[term],
                "ci_95_low": model.conf_int().loc[term, 0], "ci_95_high": model.conf_int().loc[term, 1],
                "z_value": model.tvalues[term], "p_value": model.pvalues[term],
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bh"] = bh_adjust(result["p_value"])
    return result


def evaluator_agreement(scores):
    unit = ["source", "filename", "profile", "degraded", "generation_model", "keypoint_type"]
    rows = []
    for dimension, group in scores.groupby("dimension"):
        matrix = group.pivot_table(index=unit, columns="evaluator_model", values="automatic_score", aggfunc="mean")
        for left, right in combinations(matrix.columns, 2):
            paired = matrix[[left, right]].dropna()
            if len(paired) < 3:
                continue
            rho, rho_p = spearmanr(paired[left], paired[right])
            tau, tau_p = kendalltau(paired[left], paired[right])
            difference = paired[left] - paired[right]
            rows.append({"dimension": dimension, "evaluator_a": left, "evaluator_b": right, "n": len(paired),
                         "spearman_rho": rho, "spearman_p": rho_p, "kendall_tau_b": tau, "kendall_p": tau_p,
                         "mean_absolute_pairwise_difference": np.mean(np.abs(difference)),
                         "bias_a_minus_b": np.mean(difference)})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["spearman_p_bh"] = bh_adjust(result["spearman_p"])
        result["kendall_p_bh"] = bh_adjust(result["kendall_p"])
    return result


def has_degradation(row, target):
    profile, degraded = key(row["profile"]), key(row["degraded"])
    return target in degraded or f"low_{target}" in profile or f"{target}_" in profile


def degradation_sensitivity(scores):
    rows = []
    for keys, group in scores.groupby(["dimension", "evaluator_model", "keypoint_type"]):
        for target in TARGET_DIMENSIONS:
            flags = group.apply(lambda row: has_degradation(row, target), axis=1)
            present, absent = group.loc[flags, "automatic_score"], group.loc[~flags, "automatic_score"]
            if len(present) < 2 or len(absent) < 2:
                continue
            pooled_sd = np.sqrt((present.var(ddof=1) + absent.var(ddof=1)) / 2)
            rows.append({"metric_dimension": keys[0], "evaluator_model": keys[1], "keypoint_type": keys[2], "degradation_dimension": target,
                         "n_present": len(present), "n_absent": len(absent), "mean_present": present.mean(),
                         "mean_absent": absent.mean(), "mean_difference_present_minus_absent": present.mean() - absent.mean(),
                         "standardized_difference": (present.mean() - absent.mean()) / pooled_sd if pooled_sd else np.nan})
    return pd.DataFrame(rows)


PROFILE_GROUP_ORDER = (
    "high_quality",
    "balanced_mid_quality",
    "low_factuality",
    "low_coverage",
    "low_coherence",
    "low_fluency",
)


def profile_quality_groups(row, score_dimension=None):
    """Return clean controls and every profile degraded on the score dimension."""
    profile = key(row.get("profile", ""))
    degraded = key(row.get("degraded", ""))
    groups = []
    if profile in {"high_quality", "balanced_mid_quality"} and not degraded:
        groups.append(profile)
    if score_dimension in TARGET_DIMENSIONS:
        low_group = f"low_{score_dimension}"
        profile_is_degraded = score_dimension in profile or profile == "all_degraded"
        degraded_is_degraded = score_dimension in degraded
        if profile_is_degraded or degraded_is_degraded:
            groups.append(low_group)
    return groups


def profile_quality_descriptives(scores):
    """Score distributions for each dimension/evaluator/combined quality group."""
    rows = []
    expanded = []
    for _, row in scores.iterrows():
        for group in profile_quality_groups(row, row.get("dimension")):
            item = row.to_dict()
            item["profile_quality_group"] = group
            expanded.append(item)
    if not expanded:
        return pd.DataFrame(columns=["dimension", "evaluator_model", "profile_quality_group", "n", "mean", "sd", "median",
                                     "score_min", "score_max", "score_range", "score_p05", "score_p95",
                                     "ci_95_low", "ci_95_high", "score_bucket_1_2", "score_bucket_2_3",
                                     "score_bucket_3_4", "score_bucket_4_5", "score_bucket_0_25",
                                     "score_bucket_25_50", "score_bucket_50_75", "score_bucket_75_1"])
    expanded = pd.DataFrame(expanded)
    grouped = expanded.groupby(["dimension", "evaluator_model", "profile_quality_group"], dropna=False)
    for values, group in grouped:
        score = group["automatic_score"].to_numpy(float)
        if len(score) < 2:
            continue
        low, high = bootstrap_mean_ci(score)
        span = score_span_summary(score)
        bucket = score_bucket_summary(score, values[0])
        row = dict(zip(["dimension", "evaluator_model", "profile_quality_group"], values)) | {
            "n": len(score), "mean": np.mean(score), "sd": np.std(score, ddof=1) if len(score) > 1 else np.nan,
            "median": np.median(score),
            "score_min": span["score_min"], "score_max": span["score_max"], "score_range": span["score_range"],
            "score_p05": span["score_p05"], "score_p95": span["score_p95"],
            "ci_95_low": low, "ci_95_high": high,
        }
        row["dimension"] = display_dimension(row["dimension"])
        row["profile_quality_group"] = display_quality_group(row["profile_quality_group"])
        row.update(bucket)
        rows.append(row)
    return pd.DataFrame(rows)


def profile_quality_adjusted_comparisons(scores):
    """Pairwise comparisons between the six effective quality groups pooled across rows."""
    rows = []
    expanded = []
    for _, row in scores.iterrows():
        for group in profile_quality_groups(row, row.get("dimension")):
            item = row.to_dict()
            item["profile_quality_group"] = group
            expanded.append(item)
    if not expanded:
        return pd.DataFrame(columns=["dimension", "profile_a", "profile_b", "n_scores", "n_transcript_clusters",
                                     "adjusted_mean_difference_b_minus_a", "robust_se", "ci_95_low", "ci_95_high", "z_value", "p_value", "p_bh"])
    expanded = pd.DataFrame(expanded)
    for dimension, dimension_group in expanded.groupby("dimension"):
        dimension_group = dimension_group.dropna(subset=["automatic_score", "profile_quality_group"])
        groups = sorted(dimension_group["profile_quality_group"].unique())
        for left, right in combinations(groups, 2):
            pair = dimension_group[dimension_group["profile_quality_group"].isin([left, right])].copy()
            if len(pair) < 20 or pair["profile_quality_group"].nunique() < 2:
                continue
            pair["profile_quality_group_is_right"] = (pair["profile_quality_group"] == right).astype(int)
            try:
                model = smf.ols(
                    "automatic_score ~ profile_quality_group_is_right + C(generation_model) + C(source)",
                    data=pair,
                ).fit(cov_type="cluster", cov_kwds={"groups": pair["filename"]})
            except (ValueError, np.linalg.LinAlgError):
                continue
            term = "profile_quality_group_is_right"
            rows.append({
                "dimension": dimension,
                "profile_a": display_quality_group(left), "profile_b": display_quality_group(right),
                "n_scores": len(pair), "n_transcript_clusters": pair["filename"].nunique(),
                "adjusted_mean_difference_b_minus_a": model.params[term], "robust_se": model.bse[term],
                "ci_95_low": model.conf_int().loc[term, 0], "ci_95_high": model.conf_int().loc[term, 1],
                "z_value": model.tvalues[term], "p_value": model.pvalues[term],
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bh"] = bh_adjust(result["p_value"])
    return result


def profile_quality_svg(scores, output):
    """Write a readable 2x2 PNG/SVG grid of scores by quality group."""
    expanded = []
    for _, row in scores.iterrows():
        for group in profile_quality_groups(row, row.get("dimension")):
            item = row.to_dict()
            item["profile_quality_group"] = group
            expanded.append(item)
    grouped = pd.DataFrame(expanded)
    dimensions = [dimension for dimension in TARGET_DIMENSIONS if dimension in set(grouped.get("dimension", []))]
    if not dimensions:
        return
    summary = profile_quality_descriptives(scores)
    panels = [(dimension, grouped[grouped["dimension"] == dimension]) for dimension in dimensions]
    panel_width, panel_height = 860, 560
    width, height = panel_width * 2, panel_height * 2
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<style>text{font-family:Arial,sans-serif;font-size:28px}.title{font-size:34px;font-weight:bold}.small{font-size:25px}.axis{font-size:29px}</style>',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="24" y="36" class="title">Automatic scores by effective quality group</text>',
                '<text x="24" y="90" class="small">Points: means; bars: bootstrap 95% confidence intervals</text>']
    evaluator_colours = dict(zip(sorted(grouped["evaluator_model"].dropna().unique()), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    evaluator_labels = {"gemma-12b": "G12B", "gemma-31b": "G31B", "llama-3.1-8b": "L3.1-8B", "llama-70b": "L70B"}
    quality_labels = {"high_quality": "High quality", "balanced_mid_quality": "Balanced mid-quality", "low_factuality": "Low factuality", "low_coverage": "Low coverage", "low_coherence": "Low coherence", "low_fluency": "Low fluency"}
    for panel_number, (dimension, panel) in enumerate(panels[:4]):
        panel_x = (panel_number % 2) * panel_width
        panel_y = (panel_number // 2) * panel_height
        score_low, score_high = (1.0, 5.0) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0.0, 1.0)
        candidate_groups = ("high_quality", "balanced_mid_quality", f"low_{dimension}")
        panel_groups = tuple(group for group in candidate_groups if not panel.loc[panel["profile_quality_group"] == group].empty)
        x0, x1 = panel_x + 100, panel_x + panel_width - 30
        y0, y1 = panel_y + 175, panel_y + 425
        evaluators = sorted(panel["evaluator_model"].dropna().unique())
        elements.append(f'<text x="{panel_x + 24}" y="{panel_y + 120}" class="title">{html.escape(display_dimension(dimension).title())}</text>')
        for tick in range(int(score_low), int(score_high) + 1):
            y = y1 - (tick - score_low) / (score_high - score_low) * (y1 - y0)
            elements += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dddddd"/>',
                         f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick}</text>']
        category_spacing = (x1 - x0) / max(1, len(panel_groups))
        for category_index, quality_group in enumerate(panel_groups):
            category_x = x0 + category_spacing * (category_index + 0.5)
            elements.append(f'<text x="{category_x:.1f}" y="{y1 + 28}" text-anchor="middle" class="small">{html.escape(quality_labels.get(quality_group, quality_group))}</text>')
            for evaluator_index, evaluator in enumerate(evaluators):
                colour = evaluator_colours[evaluator]
                x = category_x + (evaluator_index - (len(evaluators) - 1) / 2) * 13
                stat = summary.loc[summary["dimension"].eq(dimension) & summary["evaluator_model"].eq(evaluator) & summary["profile_quality_group"].eq(quality_group)]
                if stat.empty:
                    continue
                row = stat.iloc[0]
                mean_y = y1 - (row["mean"] - score_low) / (score_high - score_low) * (y1 - y0)
                low_y = y1 - (row["ci_95_low"] - score_low) / (score_high - score_low) * (y1 - y0)
                high_y = y1 - (row["ci_95_high"] - score_low) / (score_high - score_low) * (y1 - y0)
                elements += [f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" y2="{high_y:.1f}" stroke="{colour}" stroke-width="2.5"/>',
                             f'<line x1="{x-5:.1f}" y1="{low_y:.1f}" x2="{x+5:.1f}" y2="{low_y:.1f}" stroke="{colour}" stroke-width="2"/>',
                             f'<line x1="{x-5:.1f}" y1="{high_y:.1f}" x2="{x+5:.1f}" y2="{high_y:.1f}" stroke="{colour}" stroke-width="2"/>',
                             f'<circle cx="{x:.1f}" cy="{mean_y:.1f}" r="5" fill="{colour}" stroke="{colour}" stroke-width="2.5"/>']
        elements.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#555555" stroke-width="1.5"/>')
        if panel_number == 0:
            for evaluator_index, evaluator in enumerate(evaluators):
                legend_x = x0 + evaluator_index * 125
                elements += [f'<circle cx="{legend_x}" cy="{panel_y + 515}" r="8" fill="{evaluator_colours[evaluator]}"/>',
                             f'<text x="{legend_x + 14}" y="{panel_y + 530}" class="small">{html.escape(evaluator_labels.get(evaluator, evaluator))}</text>']
    elements.append('</svg>')
    output.write_text("\n".join(elements), encoding="utf-8")
    svg_to_png(output)


def profile_quality_png(scores, output):
    """Write the profile-quality score distributions directly as a 2x2 PNG."""
    expanded = []
    for _, row in scores.iterrows():
        for group in profile_quality_groups(row, row.get("dimension")):
            item = row.to_dict()
            item["profile_quality_group"] = group
            expanded.append(item)
    grouped = pd.DataFrame(expanded)
    dimensions = [dimension for dimension in TARGET_DIMENSIONS if dimension in set(grouped.get("dimension", []))]
    if not dimensions:
        return

    summary = profile_quality_descriptives(scores)
    image = Image.new("RGB", (2200, 1500), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    regular = ImageFont.truetype(str(font_path), 50) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 38) if font_path.exists() else ImageFont.load_default()
    category_font = ImageFont.truetype(str(font_path), 46) if font_path.exists() else small
    panel_font = ImageFont.truetype(str(bold_font_path), 54) if bold_font_path.exists() else regular
    tick_font = ImageFont.truetype(str(font_path), 38) if font_path.exists() else ImageFont.load_default()
    colours = {"high_quality": "#238b45", "balanced_mid_quality": "#74c476", "low_factuality": "#fdae6b", "low_coverage": "#fd8d3c", "low_coherence": "#9e9ac8", "low_fluency": "#cb181d"}
    evaluator_labels = {"gemma-12b": "G12B", "gemma-31b": "G31B", "llama-3.1-8b": "L3.1-8B", "llama-70b": "L70B"}
    evaluator_colours = dict(zip(sorted(grouped["evaluator_model"].dropna().unique()), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    quality_labels = {"high_quality": "High quality", "balanced_mid_quality": "Balanced mid-quality", "low_factuality": "Low factuality", "low_coverage": "Low coverage", "low_coherence": "Low coherence", "low_fluency": "Low fluency"}
    draw.text((45, 24), "Automatic scores by effective quality group", fill="black", font=regular)
    draw.text((45, 90), "Points: means; bars: bootstrap 95% confidence intervals", fill="#444444", font=small)

    for panel_number, dimension in enumerate(dimensions[:4]):
        panel = grouped[grouped["dimension"].eq(dimension)]
        panel_x = 75 + (panel_number % 2) * 1060
        panel_y = 180 + (panel_number // 2) * 650
        plot_left, plot_right = panel_x + 110, panel_x + 1015
        plot_top, plot_bottom = panel_y + 88, panel_y + 485
        score_low, score_high = (1, 5) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0, 1)
        candidate_groups = ("high_quality", "balanced_mid_quality", f"low_{dimension}")
        panel_groups = tuple(group for group in candidate_groups if not panel.loc[panel["profile_quality_group"] == group].empty)
        draw.text((panel_x, panel_y - 12), display_dimension(dimension).title(), fill="black", font=panel_font)
        for tick in range(score_low, score_high + 1):
            y = plot_bottom - (tick - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill="#dddddd", width=2)
            draw.text((plot_left - 30, y - 10), f"{tick:g}", fill="#444444", font=tick_font)

        evaluators = sorted(panel["evaluator_model"].dropna().unique())
        category_spacing = (plot_right - plot_left) / max(1, len(panel_groups))
        for category_index, quality_group in enumerate(panel_groups):
            category_x = plot_left + category_spacing * (category_index + 0.5)
            category_label = {
                "high_quality": "High quality",
                "balanced_mid_quality": "Balanced mid-\nquality",
                "low_factuality": "Low\nfactuality",
                "low_coverage": "Low\ncoverage",
                "low_coherence": "Low\ncoherence",
                "low_fluency": "Low\nfluency",
            }.get(quality_group, quality_labels.get(quality_group, quality_group))
            for line_number, label_line in enumerate(category_label.split("\n")):
                draw.text((category_x, plot_bottom + 15 + line_number * 48), label_line,
                          fill="#333333", font=category_font, anchor="mt")
            for evaluator_index, evaluator in enumerate(evaluators):
                offset = (evaluator_index - (len(evaluators) - 1) / 2) * 25
                x_center = category_x + offset
                colour = evaluator_colours[evaluator]
                stat = summary.loc[
                    summary["dimension"].eq(dimension)
                    & summary["evaluator_model"].eq(evaluator)
                    & summary["profile_quality_group"].eq(quality_group)
                ]
                if stat.empty:
                    continue
                row = stat.iloc[0]
                mean_y = plot_bottom - (row["mean"] - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                low_y = plot_bottom - (row["ci_95_low"] - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                high_y = plot_bottom - (row["ci_95_high"] - score_low) / (score_high - score_low) * (plot_bottom - plot_top)
                draw.line((x_center, high_y, x_center, low_y), fill=colour, width=4)
                draw.line((x_center - 7, high_y, x_center + 7, high_y), fill=colour, width=4)
                draw.line((x_center - 7, low_y, x_center + 7, low_y), fill=colour, width=4)
                draw.ellipse((x_center - 8, mean_y - 8, x_center + 8, mean_y + 8), fill=colour, outline=colour, width=4)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#555555", width=3)
    legend_x, legend_y = 100, 1430
    for evaluator in sorted(grouped["evaluator_model"].dropna().unique()):
        draw.ellipse((legend_x, legend_y, legend_x + 28, legend_y + 28), fill=evaluator_colours[evaluator])
        draw.text((legend_x + 38, legend_y - 8), evaluator_labels.get(evaluator, evaluator), fill="#333333", font=small)
        legend_x += 190

    image.save(output, format="PNG", dpi=(300, 300))


def heatmap_svg(sensitivity, output):
    """Dependency-free SVG heatmap of mean standardized degradation effects."""
    matrix = sensitivity.pivot_table(index=["metric_dimension", "evaluator_model"], columns="degradation_dimension",
                                     values="standardized_difference", aggfunc="mean").reindex(columns=TARGET_DIMENSIONS)
    cell_w, cell_h, left, top = 175, 58, 290, 150
    width, height = left + cell_w * len(TARGET_DIMENSIONS) + 25, top + cell_h * len(matrix) + 55
    max_abs = max(1.0, np.nanmax(np.abs(matrix.to_numpy(float)))) if not matrix.empty else 1.0
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,sans-serif;font-size:29px}.title{font-size:32px;font-weight:bold}</style>',
                '<text x="10" y="40" class="title">Standardized score change when a degradation is present</text>',
                '<text x="10" y="88">Negative values (blue) indicate lower automatic scores.</text>']
    for col, target in enumerate(TARGET_DIMENSIONS):
        elements.append(f'<text x="{left + col * cell_w + 8}" y="{top - 10}">{html.escape(target)}</text>')
    for row, (index, values) in enumerate(matrix.iterrows()):
        y = top + row * cell_h
        elements.append(f'<text x="8" y="{y + 27}">{html.escape(" / ".join(display_dimension(part) for part in index))}</text>')
        for col, value in enumerate(values):
            if np.isnan(value):
                colour, label = "#eeeeee", "NA"
            else:
                intensity = min(abs(value) / max_abs, 1)
                colour = f"rgb({int(255*(1-intensity))},{int(255*(1-intensity))},255)" if value < 0 else f"rgb(255,{int(255*(1-intensity))},{int(255*(1-intensity))})"
                label = f"{value:.2f}"
            x = left + col * cell_w
            elements.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" fill="{colour}" stroke="#aaaaaa"/>')
            elements.append(f'<text x="{x+48}" y="{y+27}">{label}</text>')
    elements.append('</svg>')
    output.write_text("\n".join(elements), encoding="utf-8")
    svg_to_png(output)


def score_distribution_svg(scores, output):
    """Jittered dot plots showing which scores each evaluator usually assigns."""
    panels = list(scores.groupby("dimension"))
    panel_width, left, right = 1000, 310, 45
    panel_heights = [150 + 40 * group["evaluator_model"].nunique() for _, group in panels]
    width, height = panel_width, 150 + sum(panel_heights)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,sans-serif;font-size:28px}.title{font-size:32px;font-weight:bold}.small{font-size:29px}.axis{font-size:29px;font-weight:bold}</style>',
                '<text x="10" y="40" class="title">Automatic evaluator score distributions</text>',
                '<text x="10" y="84">Each dot is one evaluated summary; dense areas show frequently assigned scores.</text>']
    rng = np.random.default_rng(42)
    y0 = 150
    for panel_number, (dimension, group) in enumerate(panels):
        low, high = (1.0, 5.0) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0.0, 1.0)
        x0, x1 = left, width - right
        evaluators = sorted(group["evaluator_model"].unique())
        panel_height = panel_heights[panel_number]
        plot_top, plot_bottom = y0 + 42, y0 + 55 + 36 * len(evaluators)
        elements.append(f'<text x="10" y="{y0+17}" class="title">{html.escape(dimension)}</text>')
        for tick in np.linspace(low, high, 5 if high == 5 else 6):
            x = x0 + (tick - low) / (high - low) * (x1 - x0)
            elements.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#dddddd"/>')
            elements.append(f'<text x="{x-9:.1f}" y="{plot_bottom+18}" class="small">{tick:.1f}</text>')
        for position, evaluator in enumerate(evaluators):
            baseline = y0 + 66 + position * 36
            values = group.loc[group["evaluator_model"] == evaluator, "automatic_score"].dropna().to_numpy(float)
            if position % 2 == 0:
                elements.append(f'<rect x="{x0}" y="{baseline-10}" width="{x1-x0}" height="20" fill="#f7f7f7"/>')
            elements.append(f'<text x="{x0-12}" y="{baseline+4}" text-anchor="end" class="small">{html.escape(evaluator)} (n={len(values)})</text>')
            for value in values:
                x = x0 + (np.clip(value, low, high) - low) / (high - low) * (x1 - x0)
                y = baseline + rng.uniform(-5, 5)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#2c7fb8" fill-opacity="0.45"/>')
        elements.append(f'<text x="{(x0+x1)/2:.1f}" y="{plot_bottom+64}" text-anchor="middle" class="axis">Automatic rating / score</text>')
        y0 += panel_height
    elements.append('</svg>')
    output.write_text("\n".join(elements), encoding="utf-8")
    svg_to_png(output)


def evaluator_scatter_svgs(scores, output_dir):
    """One direct evaluator-agreement scatter plot per dimension/evaluator pair."""
    unit = ["source", "filename", "profile", "degraded", "generation_model", "keypoint_type"]
    rng = np.random.default_rng(42)
    for dimension, group in scores.groupby("dimension"):
        matrix = group.pivot_table(index=unit, columns="evaluator_model", values="automatic_score", aggfunc="mean")
        low, high = (1.0, 5.0) if dimension in {"factuality", "coverage", "coherence", "fluency"} else (0.0, 1.0)
        for left, right in combinations(matrix.columns, 2):
            paired = matrix[[left, right]].dropna()
            if len(paired) < 3:
                continue
            width, height, margin = 760, 800, 108
            scale = (width - 2 * margin) / (high - low)
            rho = spearmanr(paired[left], paired[right]).statistic
            elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                        '<rect width="100%" height="100%" fill="white"/>',
                        '<style>text{font-family:Arial,sans-serif;font-size:29px}.title{font-size:32px;font-weight:bold}</style>',
                        f'<text x="10" y="45" class="title">{html.escape(dimension)}: {html.escape(left)} vs {html.escape(right)}</text>',
                        f'<text x="10" y="82">n = {len(paired)}; Spearman ρ = {rho:.2f}</text>']
            for tick in np.linspace(low, high, 5 if high == 5 else 6):
                x = margin + (tick - low) * scale
                y = height - margin - (tick - low) * scale
                elements += [f'<line x1="{x:.1f}" y1="{margin}" x2="{x:.1f}" y2="{height-margin}" stroke="#eeeeee"/>',
                             f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#eeeeee"/>',
                             f'<text x="{x-12:.1f}" y="{height-margin+27}">{tick:.1f}</text>',
                             f'<text x="{margin-48}" y="{y+6:.1f}">{tick:.1f}</text>']
            elements.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#555555" stroke-dasharray="5,4"/>')
            for x_value, y_value in paired[[left, right]].to_numpy(float):
                x = margin + (x_value - low) * scale + rng.uniform(-1.8, 1.8)
                y = height - margin - (y_value - low) * scale + rng.uniform(-1.8, 1.8)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#d95f0e" fill-opacity="0.45"/>')
            elements += [f'<text x="{width/2:.1f}" y="{height-12}" text-anchor="middle">{html.escape(left)} score</text>',
                         f'<text x="16" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 16 {height/2:.1f})">{html.escape(right)} score</text>', '</svg>']
            safe_name = re.sub(r"[^a-z0-9]+", "_", f"{dimension}_{left}_vs_{right}".lower()).strip("_")
            svg_path = output_dir / f"evaluator_scatter_{safe_name}.svg"
            svg_path.write_text("\n".join(elements), encoding="utf-8")
            svg_to_png(svg_path)


def metric_correlation_matrix(scores):
    unit = ["source", "filename", "profile", "degraded", "generation_model"]
    selected = scores.sort_values("evaluator_model").groupby(["dimension", *unit], as_index=False).first()
    matrix = selected.pivot_table(index=unit, columns="dimension", values="automatic_score", aggfunc="mean")
    return matrix.corr(method="spearman"), matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=Path("summaries_strict_9"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_analysis/output"))
    parser.add_argument("--mapping-file", type=Path,
                        default=Path("annotation_analysis/analysis_output/calibration_mapping/learned_integer_mappings.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_likert_mappings(args.mapping_file)
    metadata = load_summary_metadata(args.summary_dir)
    rows = factuality_rows(Path("seval-ex")) + coverage_rows(Path("relevance")) + nli_relevance_rows(Path("relevance"))
    coherence_files = {
        "llama-3.1-8b": Path("coherence/coherence_geval_ablation_llama3.1-8b-instruct/coherence_ablation_detailed.csv"),
        "llama-70b": Path("coherence/coherence_geval_ablation_llama70b/coherence_ablation_detailed.csv"),
        "gemma-12b": Path("coherence/coherence_geval_ablation_gemma12b/coherence_ablation_detailed.csv"),
        "gemma-31b": Path("coherence/coherence_geval_ablation_gemma31b/coherence_ablation_detailed.csv"),
    }
    fluency_files = {
        "llama-3.1-8b": Path("fluency/fluency_evaluation_results_llama-3.1-8b/fluency_evaluation_detailed.csv"),
        "llama-70b": Path("fluency/fluency_evaluation_results_llama70b/fluency_evaluation_detailed.csv"),
        "gemma-12b": Path("fluency/fluency_evaluation_results_gemma12b/fluency_evaluation_detailed.csv"),
        "gemma-31b": Path("fluency/fluency_evaluation_results_gemma_31b/fluency_evaluation_detailed.csv"),
    }
    rows += rating_rows(Path("coherence"), "coherence", coherence_files)
    rows += rating_rows(Path("fluency"), "fluency", fluency_files)
    scores = attach_metadata(pd.DataFrame(rows), metadata)
    scores = apply_likert_mappings(scores, thresholds)
    nli_scores = scores[scores.get("evaluation_approach", "").eq("nli")].copy() if "evaluation_approach" in scores.columns else pd.DataFrame(columns=scores.columns)
    scores.to_csv(args.output_dir / "automatic_score_panel.csv", index=False)
    nli_scores.to_csv(args.output_dir / "nli_coverage_score_panel.csv", index=False)
    approach_rows = rating_approach_rows(Path("coherence"), "coherence", coherence_files)
    approach_rows += rating_approach_rows(Path("fluency"), "fluency", fluency_files)
    approach_scores = attach_metadata(pd.DataFrame(approach_rows), metadata)
    for approach_dimension in ("coherence", "fluency"):
        approach_scores[approach_scores["dimension"].eq(approach_dimension)].to_csv(
            args.output_dir / f"{approach_dimension}_scores_by_approach.csv", index=False
        )
    pd.DataFrame(rating_output_availability("coherence", coherence_files) +
                 rating_output_availability("fluency", fluency_files)).to_csv(
                     args.output_dir / "rating_evaluator_output_availability.csv", index=False)
    descriptives(scores).to_csv(args.output_dir / "automatic_scores_by_generation_model.csv", index=False)
    evaluator_summary = evaluator_model_descriptives(scores)
    evaluator_summary.assign(
        dimension=evaluator_summary["dimension"].map(display_dimension)
    ).to_csv(args.output_dir / "automatic_scores_by_evaluator_model.csv", index=False)
    coverage_keypoint_descriptives(scores).to_csv(
        args.output_dir / "coverage_scores_by_keypoint_type.csv", index=False
    )
    presentation_labels(adjusted_generation_model_comparison(scores)).to_csv(
        args.output_dir / "generation_model_adjusted_comparisons.csv", index=False
    )
    presentation_labels(evaluator_agreement(scores)).to_csv(
        args.output_dir / "evaluator_model_agreement.csv", index=False
    )
    sensitivity = degradation_sensitivity(scores)
    presentation_labels(sensitivity).to_csv(args.output_dir / "degradation_sensitivity.csv", index=False)
    heatmap_svg(sensitivity, args.output_dir / "degradation_sensitivity_heatmap.svg")
    profile_quality_descriptives(scores).to_csv(args.output_dir / "profile_quality_descriptives.csv", index=False)
    profile_quality_adjusted_comparisons(scores).to_csv(args.output_dir / "profile_quality_adjusted_comparisons.csv", index=False)
    evaluator_generation_comparison_svg(
        scores, args.output_dir / "evaluator_generation_comparison.svg"
    )
    evaluator_generation_comparison_png(
        scores, args.output_dir / "evaluator_generation_comparison.png"
    )
    evaluator_generation_comparison_png(
        nli_scores, args.output_dir / "evaluator_generation_comparison_nli.png", evaluation_approach="nli"
    )
    coverage_keypoint_comparison_png(
        scores, args.output_dir / "coverage_scores_by_keypoint_type.png"
    )
    coverage_keypoint_comparison_png(
        nli_scores, args.output_dir / "coverage_scores_by_keypoint_type_nli.png", evaluation_approach="nli"
    )
    cot_approach_comparison_png(
        approach_scores, "coherence", args.output_dir / "coherence_scores_by_approach.png"
    )
    cot_approach_comparison_png(
        approach_scores, "fluency", args.output_dir / "fluency_scores_by_approach.png"
    )
    profile_quality_svg(scores, args.output_dir / "profile_quality_score_distributions.svg")
    profile_quality_png(scores, args.output_dir / "profile_quality_score_distributions.png")
    score_distribution_svg(scores, args.output_dir / "evaluator_score_distributions.svg")
    evaluator_scatter_svgs(scores, args.output_dir)
    correlation, selected = metric_correlation_matrix(scores)
    correlation.to_csv(args.output_dir / "automatic_metric_spearman_correlation.csv")
    selected.to_csv(args.output_dir / "automatic_metric_correlation_input.csv", index=False)
    print(f"Wrote {len(scores)} automatic scores to {args.output_dir}")
    print(scores.groupby(["dimension", "evaluator_model"]).size().to_string())


if __name__ == "__main__":
    main()
