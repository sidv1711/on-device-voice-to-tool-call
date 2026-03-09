#!/usr/bin/env python3
"""
Download and explore the STOP (Spoken Task Oriented Parsing) dataset on Modal.

STOP: ~200K real speech utterances across 8 domains with ~80 intents.
Source: https://dl.fbaipublicfiles.com/stop/stop.tar.gz

The raw TSV files have a header row with columns including:
  - file_id: relative path to audio file
  - normalized_utterance: transcription text
  - decoupled_normalized_seqlogical: linearized semantic parse

Phase 1: Download, parse, report intent counts, save parsed data + audio to volume.

Usage:
    modal run download_stop.py
"""

import modal

app = modal.App("download-stop")
vol = modal.Volume.from_name("training-data-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget")
)


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=7200,          # 2 hours
)
def download_and_explore():
    import csv
    import json
    import os
    import re
    import shutil
    import subprocess
    from collections import defaultdict
    from pathlib import Path

    STOP_URL = "https://dl.fbaipublicfiles.com/stop/stop.tar.gz"
    EPHEMERAL = Path("/tmp/stop_work")
    EPHEMERAL.mkdir(parents=True, exist_ok=True)

    TAR_PATH = EPHEMERAL / "stop.tar.gz"
    EXTRACT_DIR = EPHEMERAL / "stop_extracted"

    PARSED_OUT = Path("/data/stop_parsed")
    AUDIO_OUT = Path("/data/stop_audio")

    # -------------------------------------------------------------------------
    # 1. Download
    # -------------------------------------------------------------------------
    if not TAR_PATH.exists():
        print("Downloading stop.tar.gz (~50GB)...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(TAR_PATH), STOP_URL],
            check=True,
        )
        print(f"Downloaded: {TAR_PATH.stat().st_size / 1e9:.1f} GB")
    else:
        print("stop.tar.gz already present, skipping download")

    # -------------------------------------------------------------------------
    # 2. Extract
    # -------------------------------------------------------------------------
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True)
        print("Extracting (this may take a while)...")
        subprocess.run(
            ["tar", "-xzf", str(TAR_PATH), "-C", str(EXTRACT_DIR)],
            check=True,
        )
        print("Extraction complete")
    else:
        print("Already extracted, skipping")

    # Find the manifests directory
    manifest_dir = None
    for candidate in [
        EXTRACT_DIR / "stop" / "manifests",
        EXTRACT_DIR / "manifests",
    ]:
        if candidate.exists():
            manifest_dir = candidate
            break

    if manifest_dir is None:
        # Brute-force search
        print("Searching for manifests directory...")
        for root, dirs, files in os.walk(EXTRACT_DIR):
            if "train.tsv" in files:
                manifest_dir = Path(root)
                break

    if manifest_dir is None:
        print("ERROR: Could not find manifest files!")
        for p in sorted(EXTRACT_DIR.rglob("*"))[:100]:
            print(f"  {p}")
        return

    print(f"Manifest dir: {manifest_dir}")

    # Find the STOP root (parent of manifests, contains audio dirs)
    stop_root = manifest_dir.parent
    print(f"STOP root: {stop_root}")

    # List top-level contents
    print("\nSTOP root contents:")
    for p in sorted(stop_root.iterdir()):
        if p.is_dir():
            n_files = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p.name}/ ({n_files} files)")
        else:
            print(f"  {p.name} ({p.stat().st_size / 1e6:.1f}MB)")

    print("\nManifest files:")
    for p in sorted(manifest_dir.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1e6:.1f}MB)")

    # -------------------------------------------------------------------------
    # 3. Parse all splits from TSV files
    # -------------------------------------------------------------------------
    RAW_SPLITS = ["train", "eval", "test"]

    def extract_intent(parse_line: str) -> str:
        """Extract outermost IN: intent from linearized parse tree."""
        match = re.match(r'\[IN:(\S+)', parse_line.strip())
        if match:
            return match.group(1)
        return "UNKNOWN"

    def extract_slots(parse_line: str) -> dict:
        """Extract slot-value pairs from linearized parse tree."""
        slots = {}
        pattern = r'\[SL:(\S+)\s+((?:(?!\[SL:|\[IN:)[^\]])*)\]'
        for match in re.finditer(pattern, parse_line):
            slot_name = match.group(1)
            slot_value = match.group(2).strip()
            slot_value = re.sub(r'\[.*?\]', '', slot_value).strip()
            if slot_value:
                slots[slot_name] = slot_value
        return slots

    all_entries = []
    intent_counts = defaultdict(lambda: defaultdict(int))
    split_totals = defaultdict(int)

    for split in RAW_SPLITS:
        tsv_file = manifest_dir / f"{split}.tsv"
        if not tsv_file.exists():
            print(f"\n  WARNING: {split}.tsv not found, skipping")
            continue

        print(f"\nParsing {split}.tsv...")

        # Read header to discover columns
        with open(tsv_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            columns = reader.fieldnames
            print(f"  Columns: {columns}")

            # Match exact STOP column names, fallback to fuzzy matching
            EXACT = {
                "file_id": "file_col",
                "normalized_utterance": "text_col",
                "decoupled_normalized_seqlogical": "parse_col",
            }
            file_col = text_col = parse_col = None
            for col in columns:
                if col == "file_id":
                    file_col = col
                elif col == "normalized_utterance":
                    text_col = col
                elif col == "decoupled_normalized_seqlogical":
                    parse_col = col

            # Fuzzy fallback
            if file_col is None:
                for col in columns:
                    if "file" in col.lower():
                        file_col = col; break
            if text_col is None:
                for col in columns:
                    if "utterance" in col.lower() or col.lower() in ("text", "sentence"):
                        text_col = col; break
            if parse_col is None:
                for col in columns:
                    if "seqlogical" in col.lower() or "parse" in col.lower():
                        parse_col = col; break

            print(f"  file_col={file_col}, text_col={text_col}, parse_col={parse_col}")

            count = 0
            for row in reader:
                file_id = row.get(file_col, "") if file_col else ""
                transcript = row.get(text_col, "") if text_col else ""
                parse = row.get(parse_col, "") if parse_col else ""

                if not parse:
                    continue

                intent = extract_intent(parse)
                slots = extract_slots(parse)

                entry = {
                    "split": split,
                    "index": count,
                    "audio_file": file_id,
                    "transcript": transcript,
                    "parse": parse,
                    "intent": intent,
                    "slots": slots,
                }
                # Include all other columns as extra metadata
                for col in columns:
                    if col not in (file_col, text_col, parse_col) and row.get(col):
                        entry[col] = row[col]

                all_entries.append(entry)
                intent_counts[intent][split] += 1
                split_totals[split] += 1
                count += 1

            print(f"  Parsed {count:,} entries from {split}")

    # -------------------------------------------------------------------------
    # 4. Print report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STOP Dataset Report")
    print("=" * 70)
    print(f"\nTotal samples: {len(all_entries):,}")
    print(f"\nPer-split counts:")
    for split in RAW_SPLITS:
        print(f"  {split:12s}: {split_totals[split]:>7,}")

    print(f"\nTotal unique intents: {len(intent_counts)}")

    intent_totals = {}
    for intent, split_counts in intent_counts.items():
        intent_totals[intent] = sum(split_counts.values())

    sorted_intents = sorted(intent_totals.items(), key=lambda x: -x[1])

    print(f"\nPer-intent counts (all splits combined):")
    print(f"  {'Intent':<45s} {'Total':>7s}  {'train':>7s}  {'eval':>7s}  {'test':>7s}")
    print(f"  {'-'*45} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    tier_500 = []
    tier_200 = []
    tier_low = []

    for intent, total in sorted_intents:
        splits = intent_counts[intent]
        print(f"  IN:{intent:<41s} {total:>7,}  "
              f"{splits.get('train', 0):>7,}  "
              f"{splits.get('eval', 0):>7,}  "
              f"{splits.get('test', 0):>7,}")

        if total >= 500:
            tier_500.append((intent, total))
        elif total >= 200:
            tier_200.append((intent, total))
        else:
            tier_low.append((intent, total))

    print(f"\n{'='*70}")
    print(f"Tier 1 (500+ samples, → own function): {len(tier_500)} intents")
    for intent, total in tier_500:
        print(f"  IN:{intent:<41s} {total:>7,}")

    print(f"\nTier 2 (200-499 samples, → own function if distinct): {len(tier_200)} intents")
    for intent, total in tier_200:
        print(f"  IN:{intent:<41s} {total:>7,}")

    print(f"\nTier 3 (<200 samples, → merge or drop): {len(tier_low)} intents")
    for intent, total in tier_low:
        print(f"  IN:{intent:<41s} {total:>7,}")

    # Domain grouping
    print(f"\n{'='*70}")
    print("Domain grouping (by intent name prefix):")
    domain_intents = defaultdict(list)
    ACTION_VERBS = {"GET", "CREATE", "SET", "DELETE", "UPDATE", "CANCEL", "SEND",
                    "PLAY", "PAUSE", "RESUME", "ADD", "REMOVE", "SNOOZE", "CHECK",
                    "IS", "START", "STOP", "SILENCE", "UNSUPPORTED", "LIKE",
                    "DISLIKE", "SKIP", "PREVIOUS", "NEXT", "REPEAT", "REPLAY",
                    "RESTART", "LOOP", "SHUFFLE"}
    for intent, total in sorted_intents:
        parts = intent.split("_")
        if len(parts) >= 2 and parts[0] in ACTION_VERBS:
            domain_hint = parts[1]
        else:
            domain_hint = parts[0] if parts else intent
        domain_intents[domain_hint.upper()].append((intent, total))

    for domain, intents in sorted(domain_intents.items(), key=lambda x: -sum(t for _, t in x[1])):
        domain_total = sum(t for _, t in intents)
        print(f"\n  {domain} (total: {domain_total:,}):")
        for intent, total in sorted(intents, key=lambda x: -x[1]):
            print(f"    IN:{intent:<39s} {total:>7,}")

    # Print a few example entries
    print(f"\n{'='*70}")
    print("Sample entries (first 5):")
    for entry in all_entries[:5]:
        print(f"  split={entry['split']} intent={entry['intent']}")
        print(f"    transcript: {entry['transcript']}")
        print(f"    parse: {entry['parse']}")
        print(f"    slots: {entry['slots']}")
        print(f"    audio: {entry['audio_file']}")
        print()

    # -------------------------------------------------------------------------
    # 5. Save parsed data to volume
    # -------------------------------------------------------------------------
    print(f"{'='*70}")
    print("Saving parsed data to volume...")

    if PARSED_OUT.exists():
        shutil.rmtree(PARSED_OUT)
    PARSED_OUT.mkdir(parents=True)

    for split in RAW_SPLITS:
        split_entries = [e for e in all_entries if e["split"] == split]
        if not split_entries:
            continue
        out_file = PARSED_OUT / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for entry in split_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  Saved {len(split_entries):,} entries to {out_file}")

    summary = {
        "total_samples": len(all_entries),
        "total_intents": len(intent_counts),
        "split_counts": dict(split_totals),
        "intent_counts": {k: dict(v) for k, v in intent_counts.items()},
        "intent_totals": intent_totals,
    }
    with open(PARSED_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    # -------------------------------------------------------------------------
    # 6. Copy audio files to volume
    # -------------------------------------------------------------------------
    print(f"\nCopying audio files to volume...")

    if AUDIO_OUT.exists():
        shutil.rmtree(AUDIO_OUT)

    copied = 0
    missing_audio = 0

    for split in RAW_SPLITS:
        split_entries = [e for e in all_entries if e["split"] == split]
        if not split_entries:
            continue

        split_audio_dir = AUDIO_OUT / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)

        for entry in split_entries:
            audio_file = entry["audio_file"]
            if not audio_file:
                missing_audio += 1
                continue

            # file_id is relative to STOP root
            src = stop_root / audio_file
            if not src.exists():
                # Try relative to extract dir
                src = EXTRACT_DIR / audio_file
            if not src.exists():
                # Try under stop/ prefix
                src = EXTRACT_DIR / "stop" / audio_file
            if not src.exists():
                missing_audio += 1
                if missing_audio <= 10:
                    print(f"    Missing: {audio_file}")
                continue

            # Preserve full relative path to avoid basename collisions
            # (STOP reuses numeric filenames across domain subdirectories)
            dst = split_audio_dir / audio_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1

            if copied % 10000 == 0:
                print(f"    Copied {copied:,} files...")

    print(f"  Total copied: {copied:,}")
    if missing_audio:
        print(f"  Missing audio: {missing_audio:,}")

    # -------------------------------------------------------------------------
    # 7. Commit volume
    # -------------------------------------------------------------------------
    print("\nCommitting volume...")
    vol.commit()
    print("Done! Volume committed.")
    print(f"\nData saved to:")
    print(f"  Parsed: /data/stop_parsed/ (JSONL per split + summary.json)")
    print(f"  Audio:  /data/stop_audio/  ({copied:,} files)")
