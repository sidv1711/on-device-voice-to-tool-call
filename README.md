# Model Description

End-to-end model that converts spoken audio directly into structured JSON tool calls — no ASR pipeline needed.

**0.6B parameters | 168MB weights | 77.6% exact match on STOP real speech**

```
"Set an alarm for seven AM tomorrow"  →  {"function": "alarm_set", "arguments": {"time": "seven am tomorrow"}}
"What's the weather in Chicago?"      →  {"function": "weather_get_current", "arguments": {"location": "chicago"}}
"Play some jazz music"                →  {"function": "media_play_music", "arguments": {"genre": "jazz"}}
```

30 functions across 16 domains: alarm, weather, navigation, messaging, media, calendar, email, smart home, shopping, food, travel, finance, utilities, and more.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

---

## Live Demo

Try the model in your browser — record a voice command and see the JSON tool call output in real time.

**[Launch Demo](https://runanywhere--runanywhere-demo-demo-serve.modal.run)**

The demo runs on a Modal A10G GPU. First load takes ~40s (cold start + model loading), then inference runs in ~1.5–2s per query. The E2E tool call result streams back immediately, followed by a Whisper ASR transcript for reference.

### Deploy your own instance

Requires a [Modal](https://modal.com) account and the trained model weights on the `hybrid-model-storage` volume.

```bash
pip install modal
modal setup  # one-time auth

# Dev mode (hot reload, temporary URL)
modal serve demo.py

# Production (persistent URL)
modal deploy demo.py
```

The demo auto-provisions an A10G GPU, loads model weights from the Modal volume, and serves a web UI. The container scales to zero when idle and wakes on the next request (~40s cold start).

---

## Quick Start (Inference Only)

Run the model on your own audio files. Requires a CUDA GPU with ~6GB VRAM.

### 1. Install dependencies

```bash
pip install torch transformers accelerate peft bitsandbytes librosa soundfile
```

### 2. Download model weights

The fine-tuned weights (~168MB) are stored on Modal volumes. Download them locally:

```bash
pip install modal
modal setup  # one-time auth

# Download to model_export/best_model_qwen3_finetuned/
mkdir -p model_export/best_model_qwen3_finetuned
modal volume get hybrid-model-storage best_model_qwen3_finetuned/ model_export/best_model_qwen3_finetuned/
```

Or use the helper script:
```bash
python download_model.py
```

Base models (~800MB total) are downloaded automatically from HuggingFace on first run:
- `unsloth/Qwen3-0.6B` — 4-bit quantized LLM
- `openai/whisper-small` — speech encoder

### 3. Run inference

```bash
python model_export/inference.py --audio path/to/audio.wav
```

Accepts any audio format supported by librosa (wav, mp3, flac, ogg, etc.).

Output:
```json
{
  "function": "alarm_set",
  "arguments": {
    "time": "seven am tomorrow"
  }
}
```

---

## Full Training Pipeline

To reproduce the model from scratch. All training runs on [Modal](https://modal.com) (cloud GPU).

### Prerequisites

```bash
pip install modal
modal setup  # authenticate with Modal
```

You'll need:
- A Modal account (free tier works for small runs)
- An A100 GPU allocation for training (~11 hours)
- An A10G GPU for evaluation (~8 hours)

### Step 1: Generate synthetic training data

Creates 12K synthetic audio samples across 30 functions using Edge TTS:

```bash
python generate_synthetic_dataset.py
modal volume put training-data-vol ./data/synthetic_full synthetic_full --force
```

### Step 2: Download and process STOP dataset

Downloads the STOP dataset (~50GB), parses manifests, copies audio with correct directory structure:

```bash
modal run download_stop.py          # ~1 hour (download + extract + copy)
modal run process_stop.py           # ~2 min (map intents → functions)
```

### Step 3: Download and process SLURP dataset (optional)

```bash
modal run download_slurp.py         # ~30 min
```

### Step 4: Merge datasets

Combines synthetic + STOP (+ optionally SLURP) into a single training set:

```bash
modal run merge_on_modal.py
```

This creates `/data/merged_stop/` on the Modal volume with train/val/test splits.

### Step 5: Train the model

Fine-tunes Whisper-small + Qwen3-0.6B LoRA on the merged dataset. Warm-starts from a prior checkpoint if available:

```bash
modal run --detach finetune_v3b.py              # ~11 hours on A100
modal run --detach finetune_v3b.py --test-run   # quick sanity check (~5 min)
```

Training uses triple learning rates:
- Whisper top 4 layers: 1e-5
- Audio projector: 2e-4
- Qwen3 LoRA (r=32): 5e-5

Best model is saved to `/models/best_model_qwen3_finetuned` on the Modal volume based on validation loss.

### Step 6: Evaluate

```bash
modal run --detach error_analysis.py                    # E2E model on full test set
modal run --detach error_analysis.py --cascaded         # cascaded baseline comparison
modal run --detach error_analysis.py --slurp-only       # SLURP subset only
modal run --detach error_analysis.py --max-samples 100  # quick spot check
```

### Step 7: Download trained weights

```bash
python download_model.py
# or manually:
mkdir -p model_export/best_model_qwen3_finetuned
modal volume get hybrid-model-storage best_model_qwen3_finetuned/ model_export/best_model_qwen3_finetuned/
```

---

## Repository Structure

```
.
├── README.md                      # This file
├── ARCHITECTURE.md                # Detailed technical architecture doc
├── requirements.txt               # Python dependencies
│
├── model_export/                  # Inference package (send this to others)
│   ├── inference.py               # Standalone inference script
│   ├── README.md                  # Quick-start for inference only
│   └── best_model_qwen3_finetuned/
│       ├── adapter_model.bin      # LoRA weights (35MB) [not in git]
│       ├── projector.pt           # Audio projector (133MB) [not in git]
│       ├── config.json            # Model config + function list
│       └── adapter_config.json    # LoRA config
│
├── finetune_v3b.py                # E2E model training (Modal)
├── finetune_text.py               # Text-only cascaded model training (Modal)
├── error_analysis.py              # Full evaluation with error categorization
├── evaluate.py                    # Evaluation script
├── inference_fixed.py             # E2E inference script (Modal)
├── inference_cascaded.py          # Cascaded inference: Whisper ASR → text model
│
├── generate_synthetic_dataset.py  # Synthetic data generation (Edge TTS)
├── download_stop.py               # STOP dataset download + audio extraction
├── download_slurp.py              # SLURP dataset download
├── process_stop.py                # STOP intent → function mapping
├── merge_on_modal.py              # Dataset merging (synthetic + STOP)
├── merge_stop.py                  # Alternative merge script
│
├── demo.py                        # Live demo: FastAPI + custom UI on Modal (modal deploy demo.py)
├── download_model.py              # Download model weights from Modal
├── promote_checkpoint.py          # Convert training checkpoint to final format
├── asr_override.py                # ASR keyword override rules
└── debug_keys.py                  # PEFT adapter key debugging utility
```

---

## Model Files

The trained model consists of adapter weights on top of two base models:

| File | Size | Description |
|------|------|-------------|
| `adapter_model.bin` | 35MB | LoRA weights (rank 32, q/k/v/o attention) |
| `projector.pt` | 133MB | Audio projector + fine-tuned Whisper top-4 layers + classifier |
| `config.json` | 1KB | Model configuration (base models, functions, instruction prompt) |
| `adapter_config.json` | 1KB | LoRA configuration |

Base models (auto-downloaded from HuggingFace):
- `unsloth/Qwen3-0.6B` — ~600MB (4-bit quantized)
- `openai/whisper-small` — ~240MB (encoder only)

---

## Results

### STOP Real Speech Test Set (21,729 samples)

| Metric | Score |
|--------|-------|
| **Exact Match** | **77.6%** |
| Function Correct | 98.6% |
| Wrong Function | 1.4% |
| Wrong Arg Keys | 7.6% |
| Wrong Arg Values | 13.4% |

### Per-Function Accuracy (top 10 by volume)

| Function | Samples | EM% | Function Correct% |
|----------|---------|-----|-------------------|
| media_playback_control | 1,254 | 93.3% | 96.5% |
| alarm_cancel | 790 | 92.4% | 98.9% |
| alarm_set | 3,663 | 87.5% | 99.8% |
| weather_get_current | 5,295 | 86.5% | 99.3% |
| message_read | 407 | 79.9% | 98.5% |
| navigation_traffic | 2,550 | 77.7% | 98.4% |
| navigation_start | 606 | 70.3% | 96.7% |
| calendar_get_events | 2,476 | 68.7% | 98.9% |
| media_play_music | 2,320 | 65.1% | 97.1% |
| message_send | 2,368 | 52.1% | 98.1% |

### vs Published STOP Baselines

| Model | Encoder | EM |
|-------|---------|-----|
| STOP Cascaded (Wav2Vec2 → BART) | Wav2Vec2-Large (300M) | ~76% |
| STOP E2E (HuBERT-Large Seq2Seq) | HuBERT-Large (300M) | 75.1% |
| **Ours** | **Whisper-small (39M)** | **77.6%** |

7.7x smaller encoder, competitive or better accuracy.

---

## Modal Volumes

Training data and models are stored on two Modal volumes:

| Volume | Contents |
|--------|----------|
| `training-data-vol` | Datasets: synthetic_full/, stop/, merged_stop/, slurp/ |
| `hybrid-model-storage` | Model checkpoints and eval results |

To inspect:
```bash
modal volume ls training-data-vol
modal volume ls hybrid-model-storage
```
