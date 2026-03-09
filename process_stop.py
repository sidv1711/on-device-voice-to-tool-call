#!/usr/bin/env python3
"""
Process STOP parsed data into our training format.

Reads /data/stop_parsed/ JSONL files, applies intent→function mapping,
extracts slots→arguments, writes /data/stop/{train,val,test}/metadata.jsonl.

Usage:
    modal run process_stop.py
"""

import modal

app = modal.App("process-stop")
vol = modal.Volume.from_name("training-data-vol")


# =============================================================================
# STOP INTENT → FUNCTION MAPPING (31 functions)
# =============================================================================

INTENT_TO_FUNCTION = {
    # --- Alarm (6) ---
    "CREATE_ALARM": {
        "function": "alarm_set",
        "domain": "alarm",
        "slot_map": {"DATE_TIME": "time", "ALARM_NAME": "name", "RECURRING_DATE_TIME": "recurrence"},
    },
    "GET_ALARM": {
        "function": "alarm_check",
        "domain": "alarm",
        "slot_map": {"DATE_TIME": "time", "ALARM_NAME": "name"},
    },
    "DELETE_ALARM": {
        "function": "alarm_cancel",
        "domain": "alarm",
        "slot_map": {"DATE_TIME": "time", "ALARM_NAME": "name"},
    },
    "UPDATE_ALARM": {
        "function": "alarm_update",
        "domain": "alarm",
        "slot_map": {"DATE_TIME": "time", "DATE_TIME_NEW": "new_time", "ALARM_NAME": "name"},
    },
    "SNOOZE_ALARM": {
        "function": "alarm_snooze",
        "domain": "alarm",
        "slot_map": {"DATE_TIME": "duration"},
    },
    "SILENCE_ALARM": {
        "function": "alarm_silence",
        "domain": "alarm",
        "slot_map": {"ALARM_NAME": "name"},
    },

    # --- Weather (1) ---
    "GET_WEATHER": {
        "function": "weather_get_current",
        "domain": "weather",
        "slot_map": {"LOCATION": "location", "DATE_TIME": "date", "WEATHER_ATTRIBUTE": "type"},
    },
    "GET_SUNSET": {
        "function": "weather_get_current",
        "domain": "weather",
        "slot_map": {"LOCATION": "location", "DATE_TIME": "date"},
        "fixed_args": {"type": "sunset"},
    },
    "GET_SUNRISE": {
        "function": "weather_get_current",
        "domain": "weather",
        "slot_map": {"LOCATION": "location", "DATE_TIME": "date"},
        "fixed_args": {"type": "sunrise"},
    },

    # --- Reminder (4) ---
    "CREATE_REMINDER": {
        "function": "reminder_set",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "DATE_TIME": "time", "PERSON_REMINDED": "person", "LOCATION": "location", "RECURRING_DATE_TIME": "recurrence"},
    },
    "GET_REMINDER": {
        "function": "reminder_check",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "PERSON_REMINDED": "person", "DATE_TIME": "time", "LOCATION": "location"},
    },
    "GET_REMINDER_DATE_TIME": {
        "function": "reminder_check",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "PERSON_REMINDED": "person"},
        "fixed_args": {"query": "date_time"},
    },
    "GET_REMINDER_AMOUNT": {
        "function": "reminder_check",
        "domain": "reminder",
        "slot_map": {"DATE_TIME": "time", "TODO": "text"},
        "fixed_args": {"query": "count"},
    },
    "GET_REMINDER_LOCATION": {
        "function": "reminder_check",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "PERSON_REMINDED": "person"},
        "fixed_args": {"query": "location"},
    },
    "DELETE_REMINDER": {
        "function": "reminder_cancel",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "DATE_TIME": "time", "PERSON_REMINDED": "person", "LOCATION": "location"},
    },
    "UPDATE_REMINDER": {
        "function": "reminder_update",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "DATE_TIME": "time", "PERSON_REMINDED": "person", "LOCATION": "location"},
    },
    "UPDATE_REMINDER_DATE_TIME": {
        "function": "reminder_update",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "DATE_TIME": "time", "DATE_TIME_NEW": "new_time", "PERSON_REMINDED": "person"},
    },
    "UPDATE_REMINDER_TODO": {
        "function": "reminder_update",
        "domain": "reminder",
        "slot_map": {"TODO": "text", "TODO_NEW": "new_text", "DATE_TIME": "time"},
    },

    # --- Timer (6) ---
    "CREATE_TIMER": {
        "function": "timer_set",
        "domain": "timer",
        "slot_map": {"DATE_TIME": "duration", "TIMER_NAME": "name"},
    },
    "GET_TIMER": {
        "function": "timer_check",
        "domain": "timer",
        "slot_map": {"TIMER_NAME": "name"},
    },
    "DELETE_TIMER": {
        "function": "timer_cancel",
        "domain": "timer",
        "slot_map": {"TIMER_NAME": "name", "DATE_TIME": "duration"},
    },
    "PAUSE_TIMER": {
        "function": "timer_pause",
        "domain": "timer",
        "slot_map": {"TIMER_NAME": "name"},
    },
    "RESUME_TIMER": {
        "function": "timer_resume",
        "domain": "timer",
        "slot_map": {"TIMER_NAME": "name"},
    },
    "RESTART_TIMER": {
        "function": "timer_resume",
        "domain": "timer",
        "slot_map": {"TIMER_NAME": "name"},
        "fixed_args": {"action": "restart"},
    },
    "UPDATE_TIMER": {
        "function": "timer_modify",
        "domain": "timer",
        "slot_map": {"DATE_TIME": "duration", "TIMER_NAME": "name"},
    },
    "ADD_TIME_TIMER": {
        "function": "timer_modify",
        "domain": "timer",
        "slot_map": {"DATE_TIME": "duration", "TIMER_NAME": "name"},
        "fixed_args": {"action": "add"},
    },
    "SUBTRACT_TIME_TIMER": {
        "function": "timer_modify",
        "domain": "timer",
        "slot_map": {"DATE_TIME": "duration", "TIMER_NAME": "name"},
        "fixed_args": {"action": "subtract"},
    },

    # --- Messaging (3) ---
    "SEND_MESSAGE": {
        "function": "message_send",
        "domain": "messaging",
        "slot_map": {"RECIPIENT": "recipient", "CONTENT_EXACT": "text", "GROUP": "group", "SENDER": "sender"},
    },
    "GET_MESSAGE": {
        "function": "message_read",
        "domain": "messaging",
        "slot_map": {"SENDER": "sender", "RECIPIENT": "recipient", "GROUP": "group", "DATE_TIME": "time", "CONTENT_EXACT": "text"},
    },
    "REACT_MESSAGE": {
        "function": "message_react",
        "domain": "messaging",
        "slot_map": {"SENDER": "sender", "RECIPIENT": "recipient", "GROUP": "group", "TYPE_REACTION": "reaction"},
    },

    # --- Music (5) ---
    "PLAY_MUSIC": {
        "function": "media_play_music",
        "domain": "music",
        "slot_map": {"MUSIC_ARTIST_NAME": "artist", "MUSIC_TRACK_TITLE": "song", "MUSIC_ALBUM_TITLE": "album",
                     "MUSIC_GENRE": "genre", "MUSIC_PLAYLIST_TITLE": "playlist", "MUSIC_TYPE": "type",
                     "MUSIC_PROVIDER_NAME": "provider"},
    },
    "REPLAY_MUSIC": {
        "function": "media_play_music",
        "domain": "music",
        "slot_map": {"MUSIC_TRACK_TITLE": "song", "MUSIC_ARTIST_NAME": "artist"},
        "fixed_args": {"action": "replay"},
    },
    "SKIP_TRACK_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {"ORDINAL": "count"},
        "fixed_args": {"action": "next"},
    },
    "PREVIOUS_TRACK_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {"ORDINAL": "count"},
        "fixed_args": {"action": "previous"},
    },
    "PAUSE_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {},
        "fixed_args": {"action": "pause"},
    },
    "STOP_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {},
        "fixed_args": {"action": "stop"},
    },
    "START_SHUFFLE_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {},
        "fixed_args": {"action": "shuffle"},
    },
    "LOOP_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {"MUSIC_TRACK_TITLE": "song"},
        "fixed_args": {"action": "loop"},
    },
    "LIKE_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {"MUSIC_TRACK_TITLE": "song", "MUSIC_ARTIST_NAME": "artist"},
        "fixed_args": {"action": "like"},
    },
    "DISLIKE_MUSIC": {
        "function": "media_playback_control",
        "domain": "music",
        "slot_map": {"MUSIC_TRACK_TITLE": "song", "MUSIC_ARTIST_NAME": "artist"},
        "fixed_args": {"action": "dislike"},
    },
    "ADD_TO_PLAYLIST_MUSIC": {
        "function": "music_playlist",
        "domain": "music",
        "slot_map": {"MUSIC_PLAYLIST_TITLE": "playlist", "MUSIC_TRACK_TITLE": "song", "MUSIC_ARTIST_NAME": "artist"},
        "fixed_args": {"action": "add"},
    },
    "REMOVE_FROM_PLAYLIST_MUSIC": {
        "function": "music_playlist",
        "domain": "music",
        "slot_map": {"MUSIC_PLAYLIST_TITLE": "playlist", "MUSIC_TRACK_TITLE": "song", "MUSIC_ARTIST_NAME": "artist"},
        "fixed_args": {"action": "remove"},
    },
    "CREATE_PLAYLIST_MUSIC": {
        "function": "music_playlist",
        "domain": "music",
        "slot_map": {"MUSIC_PLAYLIST_TITLE": "playlist"},
        "fixed_args": {"action": "create"},
    },

    # --- Navigation (5) ---
    "GET_DIRECTIONS": {
        "function": "navigation_start",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "PATH": "route",
                     "TRAVEL_MODE": "mode", "DATE_TIME": "time"},
    },
    "UPDATE_DIRECTIONS": {
        "function": "navigation_start",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "PATH": "route",
                     "TRAVEL_MODE": "mode"},
        "fixed_args": {"action": "update"},
    },
    "GET_INFO_TRAFFIC": {
        "function": "navigation_traffic",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "PATH": "route",
                     "DATE_TIME": "time", "LOCATION": "location"},
    },
    "GET_INFO_ROAD_CONDITION": {
        "function": "navigation_traffic",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "PATH": "route",
                     "LOCATION": "location"},
        "fixed_args": {"query": "road_condition"},
    },
    "GET_ESTIMATED_DURATION": {
        "function": "nav_duration",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "PATH": "via",
                     "TRAVEL_MODE": "mode", "DATE_TIME": "time"},
    },
    "GET_ESTIMATED_ARRIVAL": {
        "function": "nav_eta",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "DATE_TIME": "time",
                     "TRAVEL_MODE": "mode"},
        "fixed_args": {"type": "arrival"},
    },
    "GET_ESTIMATED_DEPARTURE": {
        "function": "nav_eta",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "DATE_TIME": "time",
                     "TRAVEL_MODE": "mode"},
        "fixed_args": {"type": "departure"},
    },
    "GET_DISTANCE": {
        "function": "nav_distance",
        "domain": "navigation",
        "slot_map": {"DESTINATION": "destination", "SOURCE": "origin", "TRAVEL_MODE": "mode",
                     "UNIT_DISTANCE": "unit"},
    },

    # --- Event (1) ---
    "GET_EVENT": {
        "function": "calendar_get_events",
        "domain": "event",
        "slot_map": {"NAME_EVENT": "name", "DATE_TIME": "date", "LOCATION": "location",
                     "CATEGORY_EVENT": "type", "ATTRIBUTE_EVENT": "attribute",
                     "ORGANIZER_EVENT": "organizer"},
    },
}

# Split mapping: STOP split → our split
SPLIT_MAP = {
    "train": "train",
    "eval": "val",
    "test": "test",
}


TEMPORAL_KEYS = {"time", "date", "duration", "new_time", "recurrence"}
FILLER_PREPOSITIONS = {"for", "on", "at", "by", "around", "in", "to", "until", "from", "before", "after"}
UNIT_MAP = {
    "mileage": "miles",
    "the mileage": "miles",
    "kilometers": "km",
    "kilometres": "km",
    "minutes": "min",
    "minute": "min",
}


def normalize_arg_value(key: str, value: str) -> str:
    """Normalize a STOP argument value for cleaner training targets."""
    # Global: lowercase + strip whitespace
    value = value.lower().strip()

    if key in TEMPORAL_KEYS:
        # Strip leading filler prepositions
        words = value.split()
        while words and words[0] in FILLER_PREPOSITIONS:
            words.pop(0)
        # Strip leading "the" after preposition removal
        if words and words[0] == "the":
            words.pop(0)
        value = " ".join(words) if words else value

    elif key == "unit":
        # Strip leading "the"/"in"
        words = value.split()
        while words and words[0] in {"the", "in"}:
            words.pop(0)
        value = " ".join(words) if words else value
        # Map common variants
        if value in UNIT_MAP:
            value = UNIT_MAP[value]

    elif key == "count":
        if value == "currently":
            value = "current"

    return value


@app.function(
    volumes={"/data": vol},
    timeout=1800,
)
def process():
    import json
    import re
    import shutil
    from collections import defaultdict
    from pathlib import Path

    PARSED_DIR = Path("/data/stop_parsed")
    AUDIO_DIR = Path("/data/stop_audio")
    OUTPUT_DIR = Path("/data/stop")

    if not PARSED_DIR.exists():
        print("ERROR: /data/stop_parsed/ not found. Run download_stop.py first.")
        return

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    def extract_slots_from_parse(parse_line: str) -> dict:
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

    stats = defaultdict(lambda: {"total": 0, "mapped": 0, "dropped": 0, "no_audio": 0})
    function_counts = defaultdict(int)
    dropped_intents = defaultdict(int)

    for stop_split, our_split in SPLIT_MAP.items():
        parsed_file = PARSED_DIR / f"{stop_split}.jsonl"
        if not parsed_file.exists():
            print(f"WARNING: {parsed_file} not found, skipping")
            continue

        split_dir = OUTPUT_DIR / our_split
        split_dir.mkdir(exist_ok=True)

        entries = []
        with open(parsed_file) as f:
            for line in f:
                entry = json.loads(line)
                stats[our_split]["total"] += 1

                intent = entry["intent"]
                mapping = INTENT_TO_FUNCTION.get(intent)
                if mapping is None:
                    stats[our_split]["dropped"] += 1
                    dropped_intents[intent] += 1
                    continue

                # Check audio exists
                audio_file = entry.get("audio_file", "")
                if not audio_file:
                    stats[our_split]["no_audio"] += 1
                    continue
                # Audio was copied preserving full relative path:
                # /data/stop_audio/{stop_split}/{audio_file}
                audio_path_on_vol = f"stop_audio/{stop_split}/{audio_file}"
                full_audio_path = Path("/data") / audio_path_on_vol

                if not full_audio_path.exists():
                    stats[our_split]["no_audio"] += 1
                    continue

                # Build arguments from slots
                slots = extract_slots_from_parse(entry["parse"])
                arguments = dict(mapping.get("fixed_args", {}))
                slot_map = mapping.get("slot_map", {})
                for stop_slot, our_arg in slot_map.items():
                    if stop_slot in slots:
                        arguments[our_arg] = normalize_arg_value(our_arg, slots[stop_slot])

                tool_call = {
                    "function": mapping["function"],
                    "arguments": arguments,
                }

                out_entry = {
                    "audio_path": audio_path_on_vol,
                    "transcript": entry["transcript"],
                    "tool_call": tool_call,
                    "domain": mapping["domain"],
                    "source": "stop",
                    "stop_intent": intent,
                }

                entries.append(out_entry)
                stats[our_split]["mapped"] += 1
                function_counts[mapping["function"]] += 1

        # Write metadata
        meta_file = split_dir / "metadata.jsonl"
        with open(meta_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"{our_split}: {len(entries):,} entries written")

    # Write tool schema
    functions = sorted(set(m["function"] for m in INTENT_TO_FUNCTION.values()))
    schema = {"functions": [{"name": fn} for fn in functions]}
    with open(OUTPUT_DIR / "tool_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("Processing Report")
    print(f"{'='*60}")
    for split in ["train", "val", "test"]:
        s = stats[split]
        print(f"  {split}: {s['mapped']:,} mapped, {s['dropped']:,} dropped, {s['no_audio']:,} no audio (of {s['total']:,})")

    total_mapped = sum(s["mapped"] for s in stats.values())
    print(f"\n  Total mapped: {total_mapped:,}")
    print(f"  Total functions: {len(functions)}")

    print(f"\nPer-function counts:")
    for fn, count in sorted(function_counts.items(), key=lambda x: -x[1]):
        print(f"  {fn:<25s} {count:>7,}")

    print(f"\nDropped intents:")
    for intent, count in sorted(dropped_intents.items(), key=lambda x: -x[1]):
        print(f"  IN:{intent:<35s} {count:>7,}")

    # Sample entries
    print(f"\nSample entries:")
    for split in ["train", "val", "test"]:
        meta_file = OUTPUT_DIR / split / "metadata.jsonl"
        if meta_file.exists():
            with open(meta_file) as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        break
                    entry = json.loads(line)
                    print(f"  [{split}] {entry['tool_call']['function']}: {entry['transcript'][:60]}")
                    print(f"         args={entry['tool_call']['arguments']}")

    vol.commit()
    print(f"\nVolume committed. Output: /data/stop/")
