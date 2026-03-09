"""
ASR Keyword Override module.

Post-processing layer that uses Whisper ASR transcripts to correct known
collapse predictions. When the model predicts a known "attractor" function
(e.g. vehicle_climate_turn_on), we run Whisper ASR on the audio and check
for keywords that indicate the correct function.

Only fires on known collapse targets — well-performing functions are untouched.
"""
import json
import re


# ── Keyword → function override mappings ──
# Each entry: collapse_target → list of (keywords, correct_function)
# Keywords are checked against the ASR transcript (case-insensitive).
# Order matters: first match wins. More specific patterns go first.

OVERRIDE_RULES = {
    "vehicle_climate_turn_on": [
        # unlock before lock (more specific first); exclude smart home doors
        (["unlock the car", "unlock car", "unlock vehicle", "unlock my car",
          "unlock door"], "vehicle_door_unlock"),
        (["lock the car", "lock car", "lock vehicle", "lock my car",
          "lock door"], "vehicle_door_lock"),
        (["engine start", "start the engine", "start engine", "start the car",
          "start my car", "start car", "remote start"], "vehicle_engine_start"),
        # climate_turn_off BEFORE engine_stop — "turn off the car AC" must not match engine_stop
        (["turn off climate", "turn off the climate", "turn off ac",
          "turn off the ac", "turn off air", "turn off the air", "climate off",
          "ac off", "air conditioning off", "stop the ac", "stop the climate",
          "stop climate", "stop ac", "disable climate", "disable ac",
          "off the car ac", "off car ac", "the car ac off",
          "off the car air"], "vehicle_climate_turn_off"),
        (["engine stop", "engine off", "stop the engine", "stop engine",
          "turn off the engine", "kill the engine", "shut off engine",
          "turn off engine", "turn off the car", "stop the car",
          "shut off the car"], "vehicle_engine_stop"),
        (["trunk", "boot"], "vehicle_trunk_open"),
        (["window"], "vehicle_window_control"),
        (["headlight", "headlamp", "lights on", "lights off",
          "turn on the light", "turn off the light", "fog light", "high beam",
          "hazard", "turn on light", "turn off light",
          "interior light"], "vehicle_lights_control"),
        (["tv", "television", "channel", "watch"], "media_tv_control"),
        (["translate", "translation", "how do you say", "in spanish",
          "in french", "in german", "in japanese", "in chinese", "in italian",
          "in portuguese", "in korean", "in arabic", "in russian",
          "in hindi"], "utility_translate"),
        (["order", "buy", "purchase"], "shopping_order"),
        (["plug", "air purifier", "coffee maker",
          "coffee machine"], "smart_plug_control"),
        (["remind", "reminder", "don't forget"], "reminder_create"),
        (["restaurant", "food near", "place to eat", "where to eat",
          "dining"], "food_find_restaurant"),
        (["answer the call", "pick up the call", "pick up the phone",
          "answer the phone", "answer call", "pick up call",
          "accept the call", "accept call", "decline the call", "decline call",
          "ignore the call", "ignore call", "reject the call",
          "reject call"], "phone_answer"),
    ],
    "smart_fan_control": [
        (["garage", "garage door"], "smart_garage_control"),
        (["blind", "blinds", "curtain", "curtains", "shade",
          "shades"], "smart_blinds_control"),
    ],
    "navigation_stop": [
        (["parking", "park", "where can i park", "find parking",
          "nearest parking"], "navigation_find_parking"),
    ],
    "fitness_stop_workout": [
        (["stats", "check", "progress", "how many steps", "heart rate",
          "calories", "distance", "how far"], "fitness_check_stats"),
    ],
    "vehicle_climate_set_temperature": [
        (["thermostat", "home temp", "house temp", "set the temperature",
          "set temperature to", "change the temperature", "change temperature",
          "set the temp", "cool mode", "heat mode", "eco mode",
          "auto mode"], "smart_thermostat_set"),
    ],
    "shopping_read_list": [
        (["price", "cost", "how much", "how much does", "how much is",
          "what does it cost", "check the price",
          "check price"], "shopping_check_price"),
    ],
    "media_playback_control": [
        (["call", "ring", "phone", "dial"], "phone_call"),
    ],
    "media_volume_control": [
        (["answer the call", "pick up the call", "pick up the phone",
          "answer the phone", "answer call", "pick up call",
          "accept the call", "accept call", "decline the call", "decline call",
          "ignore the call", "ignore call", "reject the call",
          "reject call"], "phone_answer"),
    ],
}

# Set of all collapse target function names (for fast lookup)
COLLAPSE_TARGETS = set(OVERRIDE_RULES.keys())

# ── Argument extraction from transcripts ──
# Simple regex-based extractors for common patterns.

_DOOR_PATTERN = re.compile(
    r'\b(front|rear|driver|passenger|back|all)\b.*\b(?:door|doors)\b'
    r'|\b(?:door|doors)\b.*\b(front|rear|driver|passenger|back|all)\b',
    re.IGNORECASE,
)
_WINDOW_PATTERN = re.compile(
    r'\b(open|close|roll\s*up|roll\s*down)\b.*\b(?:window|windows)\b'
    r'|\b(?:window|windows)\b.*\b(open|close|roll\s*up|roll\s*down)\b',
    re.IGNORECASE,
)
_LIGHT_TYPE_PATTERN = re.compile(
    r'\b(headlight|headlamp|high\s*beam|fog\s*light|interior|hazard)s?\b',
    re.IGNORECASE,
)
_LIGHT_STATE_PATTERN = re.compile(
    r'\b(turn\s*on|turn\s*off|on|off)\b',
    re.IGNORECASE,
)
_GARAGE_ACTION_PATTERN = re.compile(
    r'\b(open|close)\b',
    re.IGNORECASE,
)
_BLINDS_ACTION_PATTERN = re.compile(
    r'\b(open|close)\b',
    re.IGNORECASE,
)
_BLINDS_ROOM_PATTERN = re.compile(
    r'\b(bedroom|living\s*room|kitchen|bathroom|office|dining\s*room)\b',
    re.IGNORECASE,
)
_PHONE_ACTION_PATTERN = re.compile(
    r'\b(answer|accept|pick\s*up|decline|ignore|reject)\b',
    re.IGNORECASE,
)
_THERMOSTAT_TEMP_PATTERN = re.compile(
    r'\b(\d+)\s*(?:degrees?|°)?\b',
    re.IGNORECASE,
)
_THERMOSTAT_MODE_PATTERN = re.compile(
    r'\b(heat|cool|auto|off|eco)\s*mode\b|\bmode\s*(heat|cool|auto|off|eco)\b',
    re.IGNORECASE,
)
_TRANSLATE_LANG_PATTERN = re.compile(
    r'\b(?:to|in|into)\s+(spanish|french|german|japanese|chinese|italian|'
    r'portuguese|korean|arabic|russian|hindi|english)\b',
    re.IGNORECASE,
)
_PLUG_DEVICE_PATTERN = re.compile(
    r'\b(air purifier|coffee maker|coffee machine|lamp|fan|heater|'
    r'humidifier|dehumidifier|charger)\b',
    re.IGNORECASE,
)
_PLUG_STATE_PATTERN = re.compile(
    r'\b(turn\s*on|turn\s*off|on|off|switch\s*on|switch\s*off)\b',
    re.IGNORECASE,
)
_FITNESS_STAT_PATTERN = re.compile(
    r'\b(steps?|calories?|heart\s*rate|distance|pace)\b',
    re.IGNORECASE,
)
_PRICE_ITEM_PATTERN = re.compile(
    r'(?:price|cost)\s+(?:of\s+)?(.+?)(?:\?|$)'
    r'|how\s+much\s+(?:is|does|are)\s+(?:the\s+|a\s+)?(.+?)(?:\?|$|\s+cost)',
    re.IGNORECASE,
)
_ORDER_ITEM_PATTERN = re.compile(
    r'\b(?:order|buy|purchase)\s+(?:a\s+|some\s+|the\s+)?(.+?)(?:\?|$)',
    re.IGNORECASE,
)
_RESTAURANT_CUISINE_PATTERN = re.compile(
    r'\b(italian|chinese|japanese|mexican|indian|thai|french|korean|'
    r'american|mediterranean|sushi|pizza|burger|seafood|bbq|vegan|'
    r'vegetarian)\b',
    re.IGNORECASE,
)


def extract_args(function_name: str, transcript: str) -> dict:
    """Extract arguments from transcript for overridden functions.

    Returns a dict of arguments. Empty dict if no extraction possible.
    """
    t = transcript.lower().strip()
    args = {}

    if function_name == "vehicle_door_unlock" or function_name == "vehicle_door_lock":
        m = _DOOR_PATTERN.search(transcript)
        if m:
            door = (m.group(1) or m.group(2) or "").lower()
            if door == "back":
                door = "rear"
            if door:
                args["door"] = door

    elif function_name == "vehicle_window_control":
        m = _WINDOW_PATTERN.search(transcript)
        action = None
        if m:
            raw = (m.group(1) or m.group(2) or "").lower()
            if "open" in raw or "down" in raw:
                action = "open"
            elif "close" in raw or "up" in raw:
                action = "close"
        if not action:
            if "open" in t:
                action = "open"
            elif "close" in t or "up" in t:
                action = "close"
        if action:
            args["action"] = action

    elif function_name == "vehicle_lights_control":
        m = _LIGHT_TYPE_PATTERN.search(transcript)
        if m:
            lt = m.group(1).lower().replace(" ", "_")
            if "headlamp" in lt:
                lt = "headlights"
            elif "headlight" in lt:
                lt = "headlights"
            elif "high" in lt:
                lt = "high_beams"
            elif "fog" in lt:
                lt = "fog_lights"
            args["light_type"] = lt
        ms = _LIGHT_STATE_PATTERN.search(transcript)
        if ms:
            s = ms.group(1).lower()
            args["state"] = "on" if "on" in s else "off"
        else:
            args["state"] = "on"

    elif function_name == "smart_garage_control":
        m = _GARAGE_ACTION_PATTERN.search(transcript)
        args["action"] = m.group(1).lower() if m else "open"

    elif function_name == "smart_blinds_control":
        m = _BLINDS_ACTION_PATTERN.search(transcript)
        args["action"] = m.group(1).lower() if m else "open"
        mr = _BLINDS_ROOM_PATTERN.search(transcript)
        if mr:
            args["room"] = mr.group(1).lower()

    elif function_name == "phone_answer":
        m = _PHONE_ACTION_PATTERN.search(transcript)
        if m:
            raw = m.group(1).lower()
            if raw in ("answer", "accept", "pick up", "pick"):
                args["action"] = "answer"
            elif raw in ("decline", "reject"):
                args["action"] = "decline"
            elif raw == "ignore":
                args["action"] = "ignore"

    elif function_name == "smart_thermostat_set":
        m = _THERMOSTAT_TEMP_PATTERN.search(transcript)
        if m:
            args["temperature"] = int(m.group(1))
        mm = _THERMOSTAT_MODE_PATTERN.search(transcript)
        if mm:
            args["mode"] = (mm.group(1) or mm.group(2)).lower()

    elif function_name == "utility_translate":
        m = _TRANSLATE_LANG_PATTERN.search(transcript)
        if m:
            args["to_language"] = m.group(1).capitalize()
        # Extract text to translate (everything before "to/in <language>")
        lang_match = re.search(
            r'^(?:translate\s+)?(.+?)\s+(?:to|in|into)\s+\w+$',
            transcript, re.IGNORECASE,
        )
        if lang_match:
            text = lang_match.group(1).strip()
            text = re.sub(r'^translate\s+', '', text, flags=re.IGNORECASE).strip()
            if text:
                args["text"] = text

    elif function_name == "smart_plug_control":
        m = _PLUG_DEVICE_PATTERN.search(transcript)
        if m:
            args["device"] = m.group(1).lower()
        ms = _PLUG_STATE_PATTERN.search(transcript)
        if ms:
            s = ms.group(1).lower()
            args["state"] = "on" if "on" in s else "off"

    elif function_name == "fitness_check_stats":
        m = _FITNESS_STAT_PATTERN.search(transcript)
        if m:
            stat = m.group(1).lower()
            if "step" in stat:
                stat = "steps"
            elif "calorie" in stat:
                stat = "calories"
            elif "heart" in stat:
                stat = "heart_rate"
            args["stat"] = stat

    elif function_name == "shopping_check_price":
        m = _PRICE_ITEM_PATTERN.search(transcript)
        if m:
            item = (m.group(1) or m.group(2) or "").strip()
            if item:
                args["item"] = item

    elif function_name == "shopping_order":
        m = _ORDER_ITEM_PATTERN.search(transcript)
        if m:
            args["item"] = m.group(1).strip()

    elif function_name == "food_find_restaurant":
        m = _RESTAURANT_CUISINE_PATTERN.search(transcript)
        if m:
            args["cuisine"] = m.group(1).lower()

    elif function_name == "reminder_create":
        # Extract task from "remind me to <task>"
        m = re.search(r'remind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+at\s+|\s+in\s+|$)',
                       transcript, re.IGNORECASE)
        if m:
            args["task"] = m.group(1).strip()

    elif function_name == "phone_call":
        m = re.search(r'(?:call|ring|dial|phone)\s+(.+?)(?:\s*$)',
                       transcript, re.IGNORECASE)
        if m:
            contact = m.group(1).strip()
            contact = re.sub(r'\s*(?:please|now|for me)\s*$', '', contact, flags=re.IGNORECASE).strip()
            if contact:
                args["contact"] = contact

    elif function_name == "media_tv_control":
        if any(w in t for w in ["turn on", "switch on", "power on"]):
            args["action"] = "on"
        elif any(w in t for w in ["turn off", "switch off", "power off"]):
            args["action"] = "off"
        elif "channel" in t:
            args["action"] = "change_channel"
            m = re.search(r'channel\s+(\w+)', transcript, re.IGNORECASE)
            if m:
                args["channel"] = m.group(1)

    # Functions with no required args: vehicle_engine_start, vehicle_engine_stop,
    # vehicle_trunk_open, vehicle_climate_turn_off, navigation_find_parking
    # → empty args dict is fine

    return args


def keyword_match(transcript: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the transcript (case-insensitive)."""
    t = transcript.lower()
    return any(kw.lower() in t for kw in keywords)


def asr_override(predicted_function: str, transcript: str,
                 predicted_json: dict | None = None) -> tuple[str | None, dict | None]:
    """Apply ASR keyword override to a prediction.

    Args:
        predicted_function: The function name predicted by the model.
        transcript: The ASR transcript of the audio.
        predicted_json: The full predicted JSON (for preserving valid args).

    Returns:
        (corrected_json, override_info) if override applied, (None, None) otherwise.
        corrected_json is the full tool call dict with corrected function + args.
        override_info has details about what was overridden.
    """
    if predicted_function not in OVERRIDE_RULES:
        return None, None

    for keywords, correct_function in OVERRIDE_RULES[predicted_function]:
        if keyword_match(transcript, keywords):
            # Extract arguments from transcript
            new_args = extract_args(correct_function, transcript)
            corrected = {
                "function": correct_function,
                "arguments": new_args,
            }
            info = {
                "original_function": predicted_function,
                "corrected_function": correct_function,
                "matched_keywords": [kw for kw in keywords if kw.lower() in transcript.lower()],
                "transcript": transcript,
            }
            return corrected, info

    return None, None
