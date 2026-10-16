"""Build thesis tables for completed human annotations, including the pilot.

Run: python annotation_analysis/data_inventory.py
Other inputs resolve metadata only; input files are never modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev

csv.field_size_limit(10_000_000)
SLOTS = "abc"
DIMENSIONS = ("coherence", "factuality", "fluency", "relevance")


def read_csv(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def digest(text):
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def transcript_payload(text):
    """Remove a serialization wrapper, preserving the actual transcript text."""
    if text.lstrip().startswith("{"):
        try:
            value = ast.literal_eval(text)
            if isinstance(value, dict) and len(value) == 1 and isinstance(next(iter(value.values())), str):
                return next(iter(value.values()))
        except (ValueError, SyntaxError):
            pass
    return text


def transcript_key(row):
    return row["source"], row["filename"].replace("\\", "/").strip()


def summary_key(row, text):
    return (*transcript_key(row), digest(text))


def rating_present(value):
    try:
        return any(isinstance(x, dict) and isinstance(x.get("rating"), (int, float))
                   for x in json.loads(value or "[]"))
    except (ValueError, TypeError):
        return False


def markdown_table(rows):
    columns = list(rows[0])
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(escape(row[c]) for c in columns) + " |" for row in rows),
    ])


def split_counts(keys):
    keys = list(keys)
    return dict(short=sum(k[0] == "short" for k in keys),
                long=sum(k[0] == "long" for k in keys), total=len(keys))


def length_statistics(values):
    return dict(mean_words=round(mean(values), 1), sd_words=round(stdev(values), 1) if len(values) > 1 else 0,
                median_words=median(values), min_words=min(values), max_words=max(values))


def classify_interaction(text, attributed_speakers):
    """Classify the recorded excerpt, independently of its source/length group.

    Require recurrent speaker turns for dialogue; an unlabelled excerpt is a
    speech only when its exact text has single-speaker attribution in the corpus.
    Unknown structures must be reviewed instead of assigned a length-based label.
    """
    markers = re.findall(r"(?m)^\s*(?:\[Speaker\s+(\d+)\]\s*:|Sprecher\s+(\d+)\s*:?)", text)
    labels = [a or b for a, b in markers]
    turns = [speaker for i, speaker in enumerate(labels) if i == 0 or speaker != labels[i-1]]
    if len(set(turns)) > 1 and len(turns) > len(set(turns)):
        return dict(interaction_type="dialogue", recorded_speakers=len(set(turns)), recorded_turns=len(turns),
                    classification_basis="Multiple labelled speakers; speaker returns after another speaker's turn")
    if not turns and len(attributed_speakers) == 1:
        return dict(interaction_type="speech", recorded_speakers=1, recorded_turns=1,
                    classification_basis="Exact text matches a single-speaker corpus contribution; no labelled turn changes")
    raise ValueError("Interaction type requires review: insufficient speaker-turn evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output_dir or root / "annotation_analysis/data_inventory"
    output.mkdir(parents=True, exist_ok=True)

    exports = []
    labels_by_id = {}
    for path in sorted((root / "main_annotation/Results").glob("*.csv")):
        for row in read_csv(path):
            row.update(export=path.name, phase="pilot" if path.stem.lower().startswith("pilot") else "main")
            assert row["annotator"], f"Missing annotator ID: {path}"
            assert all(rating_present(row.get(f"{s}_{d}")) for s in SLOTS for d in DIMENSIONS), f"Incomplete form: {path}, {row['annotation_id']}"
            if row["phase"] == "main":
                previous = labels_by_id.setdefault(row["annotator"], path.stem)
                assert previous == path.stem, "Inconsistent annotator label"
            exports.append(row)
    assert exports, "No returned human annotations"
    for row in exports:
        labels_by_id.setdefault(row["annotator"], f"Annotator {row['annotator']}")

    transcripts = {}
    summaries = {}
    people = defaultdict(set)
    phases = defaultdict(set)
    annotation_instances = set()
    for row in exports:
        key = transcript_key(row)
        payload = transcript_payload(row["transcript"])
        if key in transcripts:
            assert digest(transcripts[key]) == digest(payload), f"Different transcript contents for {key}"
        transcripts[key] = payload
        label = labels_by_id[row["annotator"]]
        assert (key, label) not in annotation_instances, f"Repeated transcript/annotator export: {key}, {label}"
        annotation_instances.add((key, label))
        people[key].add(label)
        phases[key].add(row["phase"])
        current = {summary_key(row, row[f"summary_{s}"]) for s in SLOTS}
        assert len(current) == 3, f"Expected three distinct summaries: {key}"
        for slot in SLOTS:
            skey = summary_key(row, row[f"summary_{slot}"])
            summaries.setdefault(skey, dict(text=row[f"summary_{slot}"], annotators=set()))["annotators"].add(label)
    for key in transcripts:
        group = [k for k in summaries if k[:2] == key]
        assert len(group) == 3, f"Summary versions differ between annotators: {key}"
        assert all(summaries[k]["annotators"] == people[key] for k in group), f"Incomplete summary overlap: {key}"

    # Metadata sources never add unannotated transcripts to the analysis sample.
    metadata = defaultdict(set)
    def add_metadata(row, text, model, profile):
        key = summary_key(row, text)
        if key in summaries:
            metadata[key].add((model, profile))
    for path in sorted((root / "summaries_strict_8").glob("*.csv")):
        source, model = path.stem.split("_degraded_")
        for row in read_csv(path, ";"):
            row["source"] = source
            add_metadata(row, row["summary"], model, row.get("profile", ""))
    for path in sorted((root / "main_annotation/annotator_datasets").glob("*_annotations.json")):
        for row in json.loads(path.read_text(encoding="utf-8-sig")):
            row = row.get("data", row)
            for slot in SLOTS:
                add_metadata(row, row[f"summary_{slot}"], row[f"model_{slot}"], row.get(f"profile_{slot}", ""))
    for key in summaries:
        assert len(metadata[key]) == 1, f"Missing or conflicting summary metadata: {key}"
        summaries[key]["model"], summaries[key]["profile"] = next(iter(metadata[key]))
    models = sorted({r["model"] for r in summaries.values()})
    assert len(models) == 3
    for key in transcripts:
        assert {r["model"] for k,r in summaries.items() if k[:2] == key} == set(models)

    origins = defaultdict(set)
    speaker_metadata = defaultdict(set)
    for row in read_csv(root / "resources/summarization_corpus_20230815.csv", ";"):
        origins[("short", row["speaker_id"])].add("summarization_corpus_20230815")
        if row.get("speaker", "").strip():
            speaker_metadata[(row["speaker_id"], digest(row["text"]))].add(row["speaker"])
    for folder in ["lhd_2025_06_29_2026_04_29/lhd_cleaned_long",
                   "lhd_2025_06_29_2026_04_29/lhd_medium", "transcripts_cleaned"]:
        directory = root / "resources" / folder
        for path in directory.rglob("*.txt"):
            origins[("long", path.relative_to(directory).as_posix())].add(directory.name)
    assert all(len(origins[k]) == 1 for k in transcripts), "Missing or ambiguous transcript source"
    interactions = {key: classify_interaction(text, speaker_metadata.get((key[1], digest(text)), set()))
                    for key, text in transcripts.items()}

    tables = {}
    tables["overview"] = [dict(measure="Human-annotated transcripts", **split_counts(transcripts))]
    for number in sorted({len(v) for v in people.values()}):
        tables["overview"].append(dict(measure=f"Transcripts annotated by {number} {'person' if number == 1 else 'people'}",
            **split_counts(k for k,v in people.items() if len(v) == number)))

    tables["annotator_distribution"] = []
    for label in sorted(set(labels_by_id.values())):
        keys = {k for k,v in people.items() if label in v}
        tables["annotator_distribution"].append(dict(annotator=label, **split_counts(keys),
            annotated_only_by_this_person=sum(len(people[k]) == 1 for k in keys),
            also_annotated_by_another_person=sum(len(people[k]) > 1 for k in keys),
            exclusive_short=sum(k[0] == "short" and len(people[k]) == 1 for k in keys),
            exclusive_long=sum(k[0] == "long" and len(people[k]) == 1 for k in keys),
            shared_short=sum(k[0] == "short" and len(people[k]) > 1 for k in keys),
            shared_long=sum(k[0] == "long" and len(people[k]) > 1 for k in keys)))
    for row in tables["annotator_distribution"]:
        for source in ("short", "long"):
            assert row[f"exclusive_{source}"] + row[f"shared_{source}"] == row[source]
    tables["pairwise_overlap"] = []
    for a,b in combinations(sorted(set(labels_by_id.values())), 2):
        tables["pairwise_overlap"].append(dict(annotator_1=a, annotator_2=b,
            **split_counts(k for k,v in people.items() if {a,b} <= v)))
    tables["source_distribution"] = []
    for origin in sorted({next(iter(origins[k])) for k in transcripts}):
        keys = {k for k in transcripts if origin in origins[k]}
        tables["source_distribution"].append(dict(source_corpus=origin, **split_counts(keys),
            percent_of_transcripts=round(100*len(keys)/len(transcripts), 1)))
    tables["transcript_lengths"] = []
    for source in ("short", "long", "all"):
        values = [len(v.split()) for k,v in transcripts.items() if source == "all" or k[0] == source]
        tables["transcript_lengths"].append(dict(group=source, **length_statistics(values)))
    tables["summary_lengths"] = []
    for model in models:
        for source in ("short", "long"):
            values = [len(v["text"].split()) for k,v in summaries.items() if k[0] == source and v["model"] == model]
            tables["summary_lengths"].append(dict(model=model, transcript_group=source, **length_statistics(values)))

    tables["transcript_inventory"] = [dict(source=k[0], filename=k[1],
        source_corpus=next(iter(origins[k])), phase=";".join(sorted(phases[k])),
        annotators=";".join(sorted(people[k])), annotator_count=len(people[k]), word_count=len(transcripts[k].split()),
        **interactions[k])
        for k in sorted(transcripts)]
    tables["speaker_turns_per_transcript"] = [dict(source=k[0], filename=k[1],
        word_count=len(transcripts[k].split()), speaker_labels=interactions[k]["recorded_speakers"],
        speaker_turns=interactions[k]["recorded_turns"])
        for k in sorted(transcripts)]
    tables["speaker_turn_statistics"] = []
    for source in ("short", "long", "all"):
        turns = [v["recorded_turns"] for k, v in interactions.items() if source == "all" or k[0] == source]
        tables["speaker_turn_statistics"].append(dict(group=source, transcripts=len(turns),
            total_turns=sum(turns), mean_turns=round(mean(turns), 1), median_turns=median(turns),
            min_turns=min(turns), max_turns=max(turns)))
    tables["summary_metadata"] = [dict(source=k[0], filename=k[1], summary_hash=k[2], model=v["model"],
        profile=v["profile"], word_count=len(v["text"].split()), annotators=";".join(sorted(v["annotators"])))
        for k,v in sorted(summaries.items())]

    counts = split_counts(transcripts)
    overlap = split_counts(k for k,v in people.items() if len(v) > 1)
    pilot = split_counts(k for k,v in phases.items() if "pilot" in v)
    assert sum(r["total"] for r in tables["annotator_distribution"]) == len(annotation_instances)
    assert sum(r["total"] for r in tables["source_distribution"]) == len(transcripts)
    assert sum(r["total"] for r in tables["overview"][1:]) == len(transcripts)
    assert len(summaries) == 3*len(transcripts)
    for label, aggregate in (("Total speaker turns", sum), ("Average speaker turns per transcript", lambda values: round(mean(values), 1))):
        tables["overview"].append(dict(measure=label,
            **{column: aggregate([value["recorded_turns"] for key, value in interactions.items()
                                  if column == "total" or key[0] == column])
               for column in ("short", "long", "total")}))
    tables["overview"].append(dict(measure="Average transcript word count",
        **{column: round(mean(len(text.split()) for key, text in transcripts.items()
                             if column == "total" or key[0] == column), 1)
           for column in ("short", "long", "total")}))
    headline = (f"The human-annotated dataset, including the pilot, contains {counts['total']} unique transcripts "
                f"({counts['short']} short and {counts['long']} long). "
                f"{overlap['total']} transcripts ({100*overlap['total']/counts['total']:.1f}%) were annotated by more than one person.")
    captions = {
        "overview": (f"Counts refer to unique transcripts, including {pilot['total']} pilot transcripts "
            f"({pilot['short']} short, {pilot['long']} long). Each transcript has three summaries, one per generation model "
            f"({', '.join(models)}), giving {len(summaries)} unique summaries. Repeated annotations do not increase the unique transcript count. "
            "A speaker turn is one uninterrupted contribution: consecutive labelled segments by the same speaker are merged. Single-speaker corpus excerpts count as one turn. Turns are not counts of complete conversational exchanges. "
            "Average word counts use whitespace-delimited tokens in each unique displayed transcript, including speaker labels where present, with dictionary wrappers removed. The overall mean is weighted by the number of transcripts."),
        "annotator_distribution": (f"Pilot and main-study contributions are combined by participant ID. Overlap transcripts appear in each "
            f"participating annotator's row, so these rows sum to {len(annotation_instances)} transcript annotations on {len(transcripts)} unique transcripts. "
            "Exclusive and shared counts partition each annotator's contributions, with separate short/long breakdowns. Only annotators with returned annotations are included."),
        "pairwise_overlap": "Counts refer to transcripts actually annotated by both participants, including the pilot. Zero-overlap pairs are retained to show the annotation design represented in the returned data.",
        "source_distribution": "Only human-annotated transcripts are counted. Source names identify the supplied corpus/file collections; lhd_medium is classified as long under the study's original labels.",
        "speaker_turn_statistics": "Speaker turns are counted once per unique transcript, including the pilot. A change of speaker begins a new turn; consecutive segments with the same speaker label are merged. Single-speaker excerpts count as one turn. These are recorded speaker turns, not topic segments or complete conversations. Individual counts for all transcripts are in speaker_turns_per_transcript.csv. Means and medians are calculated across transcripts without weighting by annotation count.",
        "transcript_lengths": "Lengths are whitespace-delimited word counts of the transcript text displayed to annotators, with serialized dictionary wrappers removed. Each transcript is counted once. SD is the sample standard deviation. Short/long are the original study groups, not thresholds inferred from these lengths.",
        "summary_lengths": "Lengths are whitespace-delimited word counts of the summaries actually annotated. Each unique summary is counted once, regardless of annotator overlap. Each model contributes one summary per transcript; sample sizes therefore follow the overview.",
    }
    notes = [
        "The analysis sample consists only of completed human-annotation CSV exports in main_annotation/Results, including both pilot exports. Assigned but unannotated transcripts are excluded.",
        "Transcript identity uses source group and full relative filename/speaker ID. The script checks that repeated annotations of an ID contain the same transcript text after whitespace normalization and removal of dictionary wrappers.",
        "Interaction type is independent of length: dialogue requires multiple labelled speakers and a speaker returning after another speaker's turn. Speech requires an exact-text match to a single-speaker corpus contribution with no labelled turn changes. This classifies the recorded excerpt, not the wider meeting: a speech may address earlier remarks outside the excerpt. Dialogue includes multi-party discussions. Per-transcript evidence is in transcript_inventory.csv; speaker labels are not independently verified against audio.",
        "The overview reports speaker turns within transcripts, rather than numbers of transcripts classified as speeches/dialogues. Speaker-turn counts are derived from the recorded speaker labels, with single-speaker source metadata used for unlabelled speech excerpts. No speaker changes are inferred from quotations or mentions of other people.",
        "Summary identities use actual text, not shuffled A/B/C slots. The script verifies the same three summaries across annotators, one per generation model, and all four ratings (coherence, factuality, fluency, relevance) for every summary.",
        "Assignment JSON and summaries_strict_8 are used only to resolve model/profile metadata. Human-annotated material outside strict_8 remains in the dataset.",
        "L1/L2 have no returned exports and therefore contribute no observations to these tables. Missing exports do not establish whether work was performed elsewhere.",
        "Overlap measures replicated annotation, not annotator agreement. Agreement and rating distributions should be reported separately in the evaluation analysis.",
        "Counts describe the exports currently available. CSV files use UTF-8 with BOM for spreadsheet compatibility. Input files are never changed.",
    ]
    recommendations = [
        "Include the overview and annotator distribution in the thesis; the overlap table makes clear which annotators support inter-annotator agreement estimates.",
        "Use the source distribution and transcript-length table to describe the evaluated sample. The summary-length table adds useful context when comparing generation models.",
        "Describe the selection criteria, language/domain, and the rationale for the short/long groups in the dataset section. Explain why the pilot is included and whether its annotation instructions and rating scales were comparable to the main study; this cannot be established from export counts alone.",
        "If controlled quality/degradation profiles are part of the experimental design, report their distribution by model for the annotated subset. Exact profile metadata is provided in summary_metadata.csv; define any merged categories before counting them.",
        "Report human rating distributions and inter-annotator agreement in a separate annotation-quality/results table. State the overlapping sample size and participating pair; do not interpret replicated ratings as independent transcripts.",
    ]
    tables["methodology"] = [dict(note=n) for n in notes]
    tables["table_captions"] = [dict(table=k, caption=v) for k,v in captions.items()]
    for name, rows in tables.items():
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = "# Human-annotated dataset statistics\n\n" + headline + "\n\n"
    for name, caption in captions.items():
        report += f"## {name.replace('_', ' ').title()}\n\n{markdown_table(tables[name])}\n\n{caption}\n\n"
    report += "## Counting rules\n\n" + "\n\n".join(f"- {n}" for n in notes)
    report += "\n\n## Suggestions for the thesis\n\n" + "\n\n".join(f"- {n}" for n in recommendations)
    report += "\n\nReproduce: `python annotation_analysis/data_inventory.py`\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Human-annotated dataset</title>',
        '<style>body{font:15px system-ui;margin:32px;color:#182330}table{border-collapse:collapse;font-size:13px}',
        'th,td{padding:8px 12px;border:1px solid #d8dfe7;text-align:left}th{background:#edf2f7}',
        'tr:nth-child(even){background:#f7f9fb}.scroll{overflow:auto}p,li{max-width:1100px;line-height:1.5}.caption{font-size:13px}</style>',
        '<h1>Human-annotated dataset statistics</h1>', f'<p>{html.escape(headline)}</p>']
    for name, caption in captions.items():
        rows = tables[name]
        page.append(f'<h2>{name.replace("_", " ").title()}</h2><div class="scroll"><table><thead><tr>')
        page.extend(f'<th>{html.escape(k.replace("_", " "))}</th>' for k in rows[0])
        page.append('</tr></thead><tbody>')
        for row in rows:
            page.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row.values()) + '</tr>')
        page.append(f'</tbody></table></div><p class="caption">{html.escape(caption)}</p>')
    for title, paragraphs in [("Counting rules", notes), ("Suggestions for the thesis", recommendations)]:
        page.append(f'<h2>{title}</h2><ul>' + ''.join(f'<li>{html.escape(n)}</li>' for n in paragraphs) + '</ul>')
    page.append('</html>')
    (output / "report.html").write_text("\n".join(page), encoding="utf-8")
    print(headline)
    print(markdown_table(tables["overview"]))
    print(markdown_table(tables["annotator_distribution"]))
    print(f"Wrote {len(tables)} CSV tables, REPORT.md and report.html to {output}")


if __name__ == "__main__":
    main()
