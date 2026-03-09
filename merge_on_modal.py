#!/usr/bin/env python3
"""
Merge synthetic + STOP datasets on Modal.

Usage:
  1. Upload synthetic data first:
     modal volume put training-data-vol ./data/synthetic_full synthetic_full --force
  2. Ensure STOP data exists at /data/stop/ (from process_stop.py)
  3. Run merge:
     modal run merge_on_modal.py
"""

import modal

app = modal.App("merge-datasets")
vol = modal.Volume.from_name("training-data-vol")


@app.function(
    volumes={"/data": vol},
    timeout=600,
)
def merge():
    import json
    import random
    import shutil
    from pathlib import Path
    from collections import defaultdict

    random.seed(42)

    synthetic_dir = Path("/data/synthetic_full")
    stop_dir = Path("/data/stop")
    output_dir = Path("/data/merged_stop")

    # Load target functions from synthetic schema
    with open(synthetic_dir / "tool_schema.json") as f:
        schema = json.load(f)
    target_functions = {fn["name"] for fn in schema["functions"]}
    print(f"Target functions ({len(target_functions)}): {sorted(target_functions)}")

    # Clean previous output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stats = defaultdict(lambda: {"synthetic": 0, "stop": 0, "stop_filtered": 0})

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        merged_entries = []

        # Load synthetic data
        synthetic_meta = synthetic_dir / split / "metadata.jsonl"
        if synthetic_meta.exists():
            with open(synthetic_meta) as f:
                for line in f:
                    entry = json.loads(line)
                    merged_entries.append({
                        "audio_path": entry["audio_path"],
                        "transcript": entry["transcript"],
                        "tool_call": entry["tool_call"],
                        "domain": entry["domain"],
                        "source": "synthetic",
                        "voice": entry.get("voice", "unknown"),
                    })
                    stats[split]["synthetic"] += 1

        # Load STOP data
        stop_meta = stop_dir / split / "metadata.jsonl"
        if stop_meta.exists():
            with open(stop_meta) as f:
                for line in f:
                    entry = json.loads(line)
                    func_name = entry["tool_call"]["function"]
                    if func_name in target_functions:
                        merged_entries.append({
                            "audio_path": entry["audio_path"],
                            "transcript": entry["transcript"],
                            "tool_call": entry["tool_call"],
                            "domain": entry["domain"],
                            "source": "stop",
                        })
                        stats[split]["stop"] += 1
                    else:
                        stats[split]["stop_filtered"] += 1

        random.shuffle(merged_entries)

        with open(split_dir / "metadata.jsonl", "w") as f:
            for entry in merged_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"  Synthetic: {stats[split]['synthetic']}")
        print(f"  STOP kept: {stats[split]['stop']}")
        print(f"  STOP filtered: {stats[split]['stop_filtered']}")
        print(f"  Total: {len(merged_entries)}")

    # Copy tool schema
    shutil.copy(synthetic_dir / "tool_schema.json", output_dir / "tool_schema.json")
    print(f"\nCopied tool schema ({len(target_functions)} functions)")

    # Write dataset info
    info = {
        "description": "Merged synthetic + STOP dataset",
        "stats": dict(stats),
        "total": {s: stats[s]["synthetic"] + stats[s]["stop"] for s in ["train", "val", "test"]},
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    total = sum(s["synthetic"] + s["stop"] for s in stats.values())
    print(f"\n{'='*50}")
    print(f"Merge complete! Total: {total} samples")
    print(f"Output: {output_dir}")

    vol.commit()
    print("Volume committed.")
