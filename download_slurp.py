#!/usr/bin/env python3
"""
Download and convert SLURP dataset to our tool-call format.

SLURP: Spoken Language Understanding Resource Package
- ~72k utterances with real human audio
- 18 scenarios (domains)
- Maps to our function-calling schema
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import urllib.request
import tarfile

# =============================================================================
# SLURP INTENT → OUR TOOL MAPPING
# =============================================================================

INTENT_TO_TOOL = {
    # Alarm
    "alarm_set": {"function": "alarm_set", "map_entities": {"time": "time"}},
    "alarm_query": {"function": "alarm_set", "map_entities": {}},  # querying alarms → alarm context
    "alarm_remove": {"function": "alarm_cancel", "map_entities": {}},

    # Audio/Volume
    "audio_volume_up": {"function": "media_volume_control", "fixed_args": {"action": "up"}},
    "audio_volume_down": {"function": "media_volume_control", "fixed_args": {"action": "down"}},
    "audio_volume_mute": {"function": "media_volume_control", "fixed_args": {"action": "mute"}},
    "audio_volume_other": {"function": "media_volume_control", "map_entities": {}},
    "volume_other": {"function": "media_volume_control", "map_entities": {}},

    # Calendar
    "calendar_set": {"function": "calendar_create_event", "map_entities": {"date": "date", "time": "time", "person": "title"}},
    "calendar_query": {"function": "calendar_get_events", "map_entities": {"date": "date"}},
    "calendar_remove": None,  # was calendar_create_event — contradictory

    # Datetime
    "datetime_query": {"function": "info_get_time", "map_entities": {"location": "location"}},
    "datetime_convert": {"function": "utility_convert", "map_entities": {}},
    "convert": {"function": "utility_convert", "map_entities": {}},

    # Email
    "email_sendemail": {"function": "email_send", "map_entities": {"person": "recipient"}},
    "email_query": {"function": "email_check", "map_entities": {}},
    "email_querycontact": {"function": "email_check", "map_entities": {"person": "contact"}},
    "email_addcontact": None,  # adding contacts is not email_send — skip ambiguous

    # IoT / Smart Home (bare keys removed — composite lookup handles them)
    "iot_hue_lighton": {"function": "smart_light_control", "fixed_args": {"state": "on"}},
    "iot_hue_lightoff": {"function": "smart_light_control", "fixed_args": {"state": "off"}},
    "iot_hue_lightup": {"function": "smart_light_control", "fixed_args": {"action": "brighten"}},
    "iot_hue_lightdim": {"function": "smart_light_control", "fixed_args": {"action": "dim"}},
    "iot_hue_lightchange": {"function": "smart_light_control", "map_entities": {"color_type": "color"}},
    "iot_wemo_on": {"function": "smart_plug_control", "fixed_args": {"state": "on"}},
    "iot_wemo_off": {"function": "smart_plug_control", "fixed_args": {"state": "off"}},
    "iot_coffee": {"function": "smart_plug_control", "fixed_args": {"device": "coffee maker", "state": "on"}},
    "iot_cleaning": {"function": "smart_vacuum_control", "fixed_args": {"action": "start"}},

    # Lists / Shopping
    "lists_createoradd": {"function": "shopping_add_to_list", "map_entities": {"list_name": "list", "todo": "item"}},
    "lists_query": {"function": "shopping_read_list", "map_entities": {}},
    "lists_remove": None,  # was shopping_add_to_list — contradictory

    # Music / Media
    "music_query": {"function": "media_play_music", "map_entities": {"artist_name": "artist", "song_name": "song"}},
    "music_likeness": {"function": "media_playback_control", "fixed_args": {"action": "like"}},
    "music_dislikeness": {"function": "media_playback_control", "fixed_args": {"action": "dislike"}},
    "music_settings": {"function": "media_playback_control", "map_entities": {}},
    "play_music": {"function": "media_play_music", "map_entities": {"artist_name": "artist", "song_name": "song"}},
    "play_audiobook": {"function": "media_play_music", "fixed_args": {"type": "audiobook"}},
    "play_podcasts": {"function": "media_podcast_play", "map_entities": {"podcast_name": "podcast_name"}},
    "play_radio": {"function": "media_play_music", "fixed_args": {"type": "radio"}},
    "play_game": None,  # games are not music — skip ambiguous
    "music": {"function": "media_play_music", "map_entities": {}},
    "podcasts": {"function": "media_podcast_play", "map_entities": {}},
    "radio": {"function": "media_play_music", "fixed_args": {"type": "radio"}},
    "game": None,  # games are not music — skip

    # News
    "news_query": {"function": "info_get_news", "map_entities": {"news_topic": "topic"}},

    # QA / Information
    "qa_factoid": {"function": "info_search", "map_entities": {}},
    "qa_definition": {"function": "utility_define", "map_entities": {"word": "word"}},
    "qa_maths": {"function": "utility_calculate", "map_entities": {}},
    "qa_currency": {"function": "utility_convert", "fixed_args": {"type": "currency"}},
    "qa_stock": {"function": "finance_check_stocks", "map_entities": {"business_name": "symbol"}},

    # Recommendations
    "recommendation_events": {"function": "info_search", "map_entities": {}},  # event recs → search, not news
    "recommendation_locations": {"function": "navigation_start", "map_entities": {"place_name": "destination"}},
    "recommendation_movies": {"function": "info_search", "fixed_args": {"type": "movies"}},

    # Social
    "social_post": {"function": "message_send", "map_entities": {}},
    "social_query": {"function": "message_read", "map_entities": {}},

    # Takeaway / Food
    "takeaway_order": {"function": "food_order", "map_entities": {"food_type": "item", "business_name": "restaurant"}},
    "takeaway_query": {"function": "food_find_restaurant", "map_entities": {"food_type": "cuisine"}},

    # Transport
    "transport_taxi": {"function": "travel_book_ride", "map_entities": {"place_name": "destination"}},
    "transport_traffic": {"function": "navigation_traffic", "map_entities": {}},
    "transport_query": {"function": "navigation_start", "map_entities": {"place_name": "destination"}},
    "transport_ticket": {"function": "travel_book_ride", "map_entities": {"place_name": "destination"}},

    # Weather
    "weather_query": {"function": "weather_get_current", "map_entities": {"place_name": "location", "date": "date"}},

    # General (skip - greetings, jokes, etc. aren't tool calls)
    "general_greet": None,
    "general_joke": None,
    "general_quirky": None,
    "greet": None,
    "joke": None,
    "quirky": None,

    # Contradictory / ambiguous (skip)
    "remove": None,  # generic remove — ambiguous

    # Cooking (too dissimilar to factoid QA — skip)
    "cooking_recipe": None,
    "cooking_query": None,
}

# Domain mapping from SLURP scenario to our domain
SCENARIO_TO_DOMAIN = {
    "alarm": "calendar",
    "audio": "media",
    "calendar": "calendar",
    "cooking": "information",
    "datetime": "information",
    "email": "communication",
    "general": None,  # skip
    "iot": "smart_home",
    "lists": "shopping",
    "music": "media",
    "news": "information",
    "play": "media",
    "qa": "information",
    "recommendation": "information",
    "social": "communication",
    "takeaway": "food",
    "transport": "navigation",
    "weather": "weather",
}


def convert_slurp_entry(entry: dict) -> dict | None:
    """Convert a SLURP entry to our format."""
    intent = entry.get("intent", "")
    scenario = entry.get("scenario", "")

    # Get mapping — try scenario-prefixed key first for generic intents
    mapping = INTENT_TO_TOOL.get(f"{scenario}_{intent}") or INTENT_TO_TOOL.get(intent)
    if mapping is None:
        return None  # Skip unmapped intents (greetings, etc.)

    # Build arguments from entities
    arguments = dict(mapping.get("fixed_args", {}))

    # Extract entities
    entity_map = mapping.get("map_entities", {})
    tokens = entry.get("tokens", [])
    for entity in entry.get("entities", []):
        entity_type = entity.get("type", "")
        if entity_type in entity_map:
            # Extract entity text from tokens
            span = entity.get("span", [])
            entity_text = " ".join(tokens[i]["surface"] for i in span if i < len(tokens))
            arguments[entity_map[entity_type]] = entity_text

    # Build tool call
    tool_call = {
        "function": mapping["function"],
        "arguments": arguments,
    }

    # Get domain
    domain = SCENARIO_TO_DOMAIN.get(scenario, "general")
    if domain is None:
        return None

    return {
        "transcript": entry.get("sentence", ""),
        "tool_call": tool_call,
        "domain": domain,
        "slurp_intent": intent,
        "slurp_scenario": scenario,
        "recordings": entry.get("recordings", []),
    }


def download_slurp_metadata(output_dir: Path):
    """Download SLURP metadata files."""
    base_url = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp"

    for split in ["train", "devel", "test"]:
        url = f"{base_url}/{split}.jsonl"
        output_file = output_dir / f"{split}.jsonl"

        if output_file.exists():
            print(f"  {split}.jsonl already exists, skipping")
            continue

        print(f"  Downloading {split}.jsonl...")
        urllib.request.urlretrieve(url, output_file)


def download_slurp_audio(output_dir: Path):
    """Download SLURP audio files from Zenodo."""
    # SLURP real audio is hosted on Zenodo
    # https://zenodo.org/record/4274930
    zenodo_urls = {
        "slurp_real": "https://zenodo.org/record/4274930/files/slurp_real.tar.gz",
    }

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    for name, url in zenodo_urls.items():
        tar_file = audio_dir / f"{name}.tar.gz"
        extract_dir = audio_dir / name

        if extract_dir.exists():
            print(f"  {name} already extracted, skipping")
            continue

        if not tar_file.exists():
            print(f"  Downloading {name}.tar.gz (~5GB, this may take a while)...")
            urllib.request.urlretrieve(url, tar_file)

        print(f"  Extracting {name}.tar.gz...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(audio_dir)


def convert_dataset(slurp_dir: Path, output_dir: Path, include_audio: bool = True):
    """Convert SLURP to our format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split mapping (SLURP uses 'devel' for validation)
    split_map = {"train": "train", "devel": "val", "test": "test"}

    stats = defaultdict(lambda: {"total": 0, "converted": 0, "skipped": 0})

    for slurp_split, our_split in split_map.items():
        input_file = slurp_dir / f"{slurp_split}.jsonl"
        if not input_file.exists():
            print(f"  Warning: {input_file} not found, skipping")
            continue

        split_dir = output_dir / our_split
        split_dir.mkdir(exist_ok=True)
        (split_dir / "audio").mkdir(exist_ok=True)

        converted_entries = []

        with open(input_file) as f:
            for line in f:
                entry = json.loads(line)
                stats[our_split]["total"] += 1

                converted = convert_slurp_entry(entry)
                if converted is None:
                    stats[our_split]["skipped"] += 1
                    continue

                # Get first valid audio file (prefer headset version)
                audio_file = None
                for rec in converted["recordings"]:
                    if rec.get("status") == "correct":
                        audio_file = rec["file"]
                        if "headset" in audio_file:
                            break  # Prefer headset recordings

                if audio_file is None and converted["recordings"]:
                    audio_file = converted["recordings"][0]["file"]

                if audio_file:
                    # Build audio path
                    src_audio = slurp_dir / "audio" / "slurp_real" / audio_file
                    if include_audio and src_audio.exists():
                        # Copy or link audio file
                        dst_audio = split_dir / "audio" / audio_file
                        if not dst_audio.exists():
                            os.link(src_audio, dst_audio)
                        converted["audio_path"] = f"{our_split}/audio/{audio_file}"
                    else:
                        converted["audio_path"] = f"slurp_real/{audio_file}"  # Reference path

                # Remove internal fields
                del converted["recordings"]

                converted_entries.append(converted)
                stats[our_split]["converted"] += 1

        # Write metadata
        metadata_file = split_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for entry in converted_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"  {our_split}: {stats[our_split]['converted']} converted, {stats[our_split]['skipped']} skipped")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert SLURP dataset")
    parser.add_argument("--output", "-o", default="./data/slurp", help="Output directory")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio download (metadata only)")
    parser.add_argument("--metadata-only", action="store_true", help="Only download and convert metadata")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    print("Step 1: Downloading SLURP metadata...")
    download_slurp_metadata(raw_dir)

    if not args.metadata_only and not args.skip_audio:
        print("\nStep 2: Downloading SLURP audio (this may take a while)...")
        download_slurp_audio(raw_dir)

    print("\nStep 3: Converting to our format...")
    converted_dir = output_dir / "converted"
    stats = convert_dataset(raw_dir, converted_dir, include_audio=not args.skip_audio)

    print("\n" + "=" * 50)
    print("SLURP conversion complete!")
    print(f"Output: {converted_dir}")

    total_converted = sum(s["converted"] for s in stats.values())
    total_skipped = sum(s["skipped"] for s in stats.values())
    print(f"Total: {total_converted} converted, {total_skipped} skipped")


if __name__ == "__main__":
    main()
