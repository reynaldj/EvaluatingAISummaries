import pandas as pd
from pathlib import Path

BASE = Path(".")
CORE_SEL = BASE / "annotator_datasets" / "core_selection.txt"
OUT = BASE / "summaries_strict_9"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    BASE / "summaries_strict_8",
    BASE / "summaries_strict_extended_degraded",
]

EXTRA_FILES = [
    "20210722_André_Schollbach_1",
    "Stadtratssitzung_vom_29.01.2026\\4_transcription.txt",
    "T_317b3d53_Jahresplanung_2025_2.txt",
]

# Parse core_selection.txt
selected_long = set()
selected_short = set()
with open(CORE_SEL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        kind, name = parts
        if kind == "LONG":
            selected_long.add(name)
        elif kind == "SHORT":
            selected_short.add(name)

# Add extra files to appropriate sets
for fname in EXTRA_FILES:
    if "\\" in fname or fname.startswith("T_"):
        selected_long.add(fname)
    else:
        selected_short.add(fname)

print(f"Selected LONG:  {len(selected_long)}")
print(f"Selected SHORT: {len(selected_short)}")

# Process each source directory
for src_dir in SOURCES:
    if not src_dir.exists():
        print(f"Skipping {src_dir} — does not exist")
        continue

    for csv_path in sorted(src_dir.glob("*.csv")):
        name = csv_path.name
        # Determine which filter set to use
        if name.startswith("long_degraded_"):
            filter_set = selected_long
        elif name.startswith("short_degraded_"):
            filter_set = selected_short
        else:
            print(f"  Skipping {name} — unknown prefix")
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str, on_bad_lines="warn")
        before = len(df)

        # Filter by filename column
        df = df[df["filename"].isin(filter_set)]
        after = len(df)

        if after == 0:
            print(f"  {name}: 0/{before} rows match — skipping output")
            continue

        # Ensure degraded column exists
        if "degraded" not in df.columns:
            df["degraded"] = ""

        out_path = OUT / name
        # Check if output already exists (from previous source dir)
        if out_path.exists():
            existing = pd.read_csv(out_path, sep=None, engine="python", dtype=str, on_bad_lines="warn")
            # Merge: keep rows from existing, add new rows not already present
            existing_keys = set(existing["filename"])
            df = df[~df["filename"].isin(existing_keys)]
            if len(df) == 0:
                print(f"  {name}: {after}/{before} rows, all already present in output — skipped")
                continue
            df = pd.concat([existing, df], ignore_index=True)
            print(f"  {name}: merged {len(df)} total rows (+{len(df) - len(existing)} new)")
        else:
            print(f"  {name}: {after}/{before} rows matched")

        df.to_csv(out_path, index=False, encoding="utf-8")

print(f"\nDone. Output in {OUT}")
