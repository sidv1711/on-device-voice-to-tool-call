#!/usr/bin/env python3
"""
Merge STOP processed data with existing merged_new (SLURP+synthetic) data.
Creates /data/merged_stop/{train,val,test}/metadata.jsonl.

Fixes audio_path mismatches from merged_new:
  - SLURP: "data/slurp/raw/audio/slurp_real/audio-XXX.flac" → "slurp_audio/audio-XXX.flac"
  - Synthetic: "train/audio/sample_XXXXX.mp3" → "synthetic_full/train/audio/sample_XXXXX.mp3"
  - STOP: "stop_audio/{split}/XXX.wav" → kept as-is (already correct)

Usage:
    modal run merge_stop.py
"""
import modal

app = modal.App("merge-stop")
vol = modal.Volume.from_name("training-data-vol")


def fix_audio_path(audio_path: str, source: str) -> str:
    """Fix audio_path to match actual volume locations."""
    import os

    if source == "slurp":
        # "data/slurp/raw/audio/slurp_real/audio-XXX.flac" → "slurp_audio/audio-XXX.flac"
        basename = os.path.basename(audio_path)
        return f"slurp_audio/{basename}"
    elif source == "synthetic":
        # "train/audio/sample_XXXXX.mp3" → "synthetic_full/train/audio/sample_XXXXX.mp3"
        return f"synthetic_full/{audio_path}"
    else:
        # STOP paths are already correct
        return audio_path


@app.function(
    volumes={"/data": vol},
    timeout=600,
)
def merge():
    import json
    import os
    import random
    import shutil
    from collections import defaultdict
    from pathlib import Path

    random.seed(42)

    STOP_DIR = Path("/data/stop")
    MERGED_NEW_DIR = Path("/data/merged_new")
    OUTPUT_DIR = Path("/data/merged_stop")

    if not STOP_DIR.exists():
        print("ERROR: /data/stop/ not found. Run process_stop.py first.")
        return
    if not MERGED_NEW_DIR.exists():
        print("ERROR: /data/merged_new/ not found.")
        return

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    stats = defaultdict(lambda: {"slurp": 0, "synthetic": 0, "stop": 0, "skipped": 0})
    func_counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(exist_ok=True)

        entries = []

        # Load existing SLURP+synthetic data with path fixes
        src = MERGED_NEW_DIR / split / "metadata.jsonl"
        if src.exists():
            with open(src) as f:
                for line in f:
                    entry = json.loads(line)
                    source = entry.get("source", "unknown")
                    old_path = entry.get("audio_path", "")
                    new_path = fix_audio_path(old_path, source)
                    entry["audio_path"] = new_path

                    # Verify audio exists
                    full_path = f"/data/{new_path}"
                    if not os.path.exists(full_path):
                        stats[split]["skipped"] += 1
                        continue

                    entries.append(entry)
                    stats[split][source] += 1
                    func_counts[entry["tool_call"]["function"]] += 1

        # Load STOP data with stratified downsampling
        # Cap per-function at MAX_PER_FUNC, total STOP at MAX_STOP_TOTAL
        MAX_PER_FUNC = 3000
        MAX_STOP_TOTAL = 30000
        src = STOP_DIR / split / "metadata.jsonl"
        if src.exists():
            # First pass: group STOP entries by function
            stop_by_func = defaultdict(list)
            with open(src) as f:
                for line in f:
                    entry = json.loads(line)
                    fn = entry["tool_call"]["function"]
                    stop_by_func[fn].append(entry)

            stop_total_raw = sum(len(v) for v in stop_by_func.values())

            # Per-function cap
            stop_sampled = []
            for fn, fn_entries in stop_by_func.items():
                if len(fn_entries) > MAX_PER_FUNC:
                    random.shuffle(fn_entries)
                    fn_entries = fn_entries[:MAX_PER_FUNC]
                stop_sampled.extend(fn_entries)

            # Total STOP cap (proportional downsample if still over)
            if len(stop_sampled) > MAX_STOP_TOTAL:
                random.shuffle(stop_sampled)
                stop_sampled = stop_sampled[:MAX_STOP_TOTAL]

            for entry in stop_sampled:
                entries.append(entry)
                stats[split]["stop"] += 1
                func_counts[entry["tool_call"]["function"]] += 1

            print(f"  STOP {split}: {stop_total_raw:,} raw → {len(stop_sampled):,} after sampling")

        # Shuffle and write merged
        random.shuffle(entries)
        out_file = split_dir / "metadata.jsonl"
        with open(out_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        s = stats[split]
        total = s["slurp"] + s["synthetic"] + s["stop"]
        print(f"{split}: {total:,} entries ({s['slurp']:,} SLURP, {s['synthetic']:,} synth, {s['stop']:,} STOP, {s['skipped']:,} skipped)")

    # Report
    total_all = sum(s["slurp"] + s["synthetic"] + s["stop"] for s in stats.values())
    print(f"\nTotal: {total_all:,}")
    print(f"Functions: {len(func_counts)}")
    print(f"\nPer-function counts (top 30):")
    for fn, count in sorted(func_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"  {fn:<30s} {count:>8,}")

    # Sample entries per source
    print(f"\nSample entries:")
    for split in ["train"]:
        meta_file = OUTPUT_DIR / split / "metadata.jsonl"
        if meta_file.exists():
            shown = {"slurp": False, "synthetic": False, "stop": False}
            with open(meta_file) as f:
                for line in f:
                    entry = json.loads(line)
                    src = entry.get("source", "unknown")
                    if src in shown and not shown[src]:
                        print(f"  [{src}] audio_path={entry['audio_path']}")
                        print(f"         tool_call={entry['tool_call']}")
                        shown[src] = True
                    if all(shown.values()):
                        break

    vol.commit()
    print(f"\nVolume committed. Output: /data/merged_stop/")
