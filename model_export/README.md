# Speech-to-Tool-Call Model

Converts spoken audio into structured JSON function calls. 77.6% exact match on STOP real speech benchmark.

**Architecture:** Whisper-small encoder → Conv1d compressor → MLP projector → Qwen3-0.6B (4-bit LoRA)

## Setup

```bash
pip install torch transformers accelerate peft bitsandbytes librosa soundfile
```

Requires a CUDA GPU with ~6GB VRAM. Base models (~800MB) are downloaded automatically from HuggingFace on first run.

## Usage

```bash
python inference.py --audio path/to/audio.wav
```

Output:
```json
{
  "function": "alarm_set",
  "arguments": {
    "time": "seven am tomorrow"
  }
}
```

## Supported Functions (30)

| Domain | Functions |
|---|---|
| Alarm | alarm_set, alarm_cancel |
| Weather | weather_get_current |
| Navigation | navigation_start, navigation_traffic |
| Messaging | message_send, message_read |
| Media | media_play_music, media_playback_control, media_podcast_play, media_volume_control |
| Calendar | calendar_get_events, calendar_create_event |
| Email | email_send, email_check |
| Smart Home | smart_light_control, smart_plug_control, smart_vacuum_control |
| Shopping | shopping_add_to_list, shopping_read_list |
| Food | food_find_restaurant, food_order |
| Travel | travel_book_ride |
| Finance | finance_check_stocks |
| Utilities | utility_calculate, utility_convert, utility_define |
| Information | info_search, info_get_news, info_get_time |

## Model Files

| File | Size | Description |
|---|---|---|
| `adapter_model.bin` | 35MB | LoRA weights (rank 32, q/k/v/o attention) |
| `projector.pt` | 133MB | Audio projector + fine-tuned Whisper top-4 layers |
| `config.json` | 1KB | Model configuration |
| `adapter_config.json` | 1KB | LoRA configuration |
| `inference.py` | - | Standalone inference script |

Base models downloaded on first run:
- `unsloth/Qwen3-0.6B` (~600MB, 4-bit quantized)
- `openai/whisper-small` encoder (~240MB)

## Performance

Evaluated on STOP real speech test set (21,729 samples):

| Metric | Score |
|---|---|
| Exact Match | 77.6% |
| Function Correct | 98.6% |
| Wrong Function | 1.4% |
