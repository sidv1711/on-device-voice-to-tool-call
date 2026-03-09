#End-to-End Speech-to-Function-Call Model

## Executive Summary

This is a lightweight end-to-end model that converts spoken natural language directly into structured JSON function calls — bypassing traditional ASR pipelines entirely. At just **0.6B parameters and ~730 MB on disk**, it achieves **77% exact match accuracy** on the STOP benchmark's real-speech test set, competitive with Meta's own baselines that use significantly larger encoders (HuBERT-Large, 75% EM).

The system supports **30 callable functions** across 8 domains (alarm, weather, navigation, messaging, media, calendar, timers, reminders), handling real human speech with diverse accents, noise, and speaking styles.

---

## What Makes This Different

### The Landscape

Most production voice assistants use a **cascaded pipeline**: Speech → ASR → NLU → Intent/Slot → Action. This introduces compounding errors at each stage and adds latency. Recent large models from OpenAI (gpt-4o-realtime), Google (Gemini 2.5 Native Audio), and Mistral (Voxtral) have added native audio function calling, but at massive scale (3B–70B+ parameters) and behind closed APIs.

Open-source alternatives like Ultravox (8–70B), Qwen2-Audio (7B), and LFM2-Audio (1.5B) support audio understanding but are primarily designed for conversational AI, not structured tool invocation.

**This model occupies a unique position**: the smallest known end-to-end model that generates full JSON function calls directly from audio, at a scale suitable for on-device or edge deployment.

| System | Parameters | Audio → JSON Tool Calls | Open Source |
|--------|-----------|------------------------|-------------|
| GPT-4o Realtime (OpenAI) | Unknown (large) | Yes | No (API only) |
| Gemini 2.5 Native Audio | Unknown (large) | Yes | No (API only) |
| Voxtral-Mini (Mistral) | 3B | Yes | Yes |
| Ultravox (Fixie AI) | 8B–70B | Yes | Yes |
| Qwen2-Audio | 7B | Partial | Yes |
| LFM2-Audio (Liquid AI) | 1.5B | Partial | Yes |
| **Ours** | **0.6B** | **Yes** | **Yes** |

### Key Differentiators

1. **10x smaller than the nearest open competitor** (0.6B vs Voxtral-Mini's 3B), while targeting the same audio → JSON function call task
2. **End-to-end architecture** that bypasses ASR entirely — audio goes in, structured JSON comes out, with no intermediate text representation
3. **Competitive with baselines using much larger encoders** — 77% EM vs STOP's 75% EM with HuBERT-Large (300M encoder alone, vs our 39M Whisper-small encoder)
4. **Novel multi-rate training** with separate learning rates for the speech encoder, audio projector, and language model

---

## Architecture

### Overview

```
                        ┌─────────────────────┐
    Raw Audio ─────────>│  Whisper-small       │
    (16kHz PCM)         │  Encoder (frozen*)   │
                        │  768-dim, 12 layers  │
                        └────────┬────────────┘
                                 │ (batch, 1500, 768)
                        ┌────────▼────────────┐
                        │  2x Conv1d           │
                        │  stride=2, ReLU      │
                        │  4x temporal         │
                        │  compression         │
                        └────────┬────────────┘
                                 │ (batch, 375, 768)
                        ┌────────▼────────────┐
                        │  LayerNorm(768)      │
                        └────────┬────────────┘
                                 │
                        ┌────────▼────────────┐
                        │  MLP Projector       │
                        │  768 → 1024 (GELU)   │
                        │  1024 → 1024         │
                        └────────┬────────────┘
                                 │ (batch, 375, 1024)
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     ┌────────▼───┐    ┌────────▼───┐    ┌──────────▼──────┐
     │ Audio      │    │ Instruction│    │ "<|audio|>      │
     │ Embeddings │    │ Prefix     │    │ Convert to tool │
     │ (375 tok)  │    │ (5 tok)    │    │ call JSON:\n"   │
     └────────┬───┘    └────────┬───┘    └──────────┬──────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │ Concatenated embeddings
                        ┌────────▼────────────┐
                        │  Qwen3-0.6B (4-bit) │
                        │  + LoRA (r=32)      │
                        │  q/k/v/o attention  │
                        │  Greedy decode      │
                        └────────┬────────────┘
                                 │
                        ┌────────▼────────────┐
                        │ {"function":        │
                        │  "alarm_set",       │
                        │  "arguments":       │
                        │  {"time": "5pm"}}   │
                        └─────────────────────┘
```

### Component Details

**Whisper-small Encoder (39M params)**
- Pretrained OpenAI Whisper-small speech encoder
- 12 transformer layers, 768-dim hidden state
- Bottom 8 layers frozen; top 4 layers fine-tuned at low learning rate (1e-5) for domain adaptation to STOP's telephone-quality audio
- Outputs 1,500 time steps per 30-second audio window

**Temporal Compression (590K params)**
- Two Conv1d layers (kernel=3, stride=2, padding=1) with ReLU activations
- Each halves the sequence length: 1,500 → 750 → 375 tokens
- 4x compression keeps the audio prefix manageable for the LLM's context window
- LayerNorm stabilizes the compressed representations before projection

**Audio Projector (1.6M params)**
- 2-layer MLP bridging Whisper's 768-dim space to Qwen3's 1024-dim embedding space
- GELU activation between layers
- The audio embeddings are injected directly into the LLM's input embedding sequence, allowing the language model to attend to audio features as if they were token embeddings

**Qwen3-0.6B Language Model (560M params, 26M trainable)**
- Alibaba's Qwen3-0.6B instruction-tuned model, 4-bit NF4 quantized via bitsandbytes
- LoRA adaptation: rank 32, applied to q/k/v/o attention projections across all 24 layers
- Only ~5% of parameters are trainable; the rest leverage pretrained language understanding
- Greedy decoding with brace-counting early stop: generation terminates when open/close brace counts balance, producing well-formed JSON without temperature sampling

**Total trainable parameters: ~44M out of ~600M (7.3%)**

### Why This Design

The architecture makes several deliberate choices:

1. **Whisper over HuBERT/Wav2Vec2**: Whisper's multilingual pretraining and robustness to noise make it more practical than self-supervised encoders. At 39M params (Whisper-small), it's also much smaller than HuBERT-Large (300M) used in STOP baselines.

2. **Conv1d compression over attention pooling**: Fixed-stride convolution provides deterministic 4x compression with minimal parameters. Alternatives like Q-Former or cross-attention pooling add complexity and parameters without clear benefit at this scale.

3. **4-bit quantization for the LLM**: Enables fitting the full training pipeline (Whisper encoder + projector + LLM) on a single A100 GPU. Critical insight: training and inference must use identical quantization — float16 inference on a 4-bit-trained model produces garbage outputs.

4. **No intermediate text representation**: Traditional cascaded systems (ASR → NLU → action) lose information at each stage — Whisper transcribes "seven thirty" but the NLU may misparse it, or ASR outputs "7.30 a.m." when downstream expects "seven thirty am". We validated this empirically: a cascaded variant (Whisper ASR → text-only Qwen3 LoRA) achieved only 50.1% EM on the same test set, compared to 77.1% for the end-to-end model. The 27-point gap comes almost entirely from ASR format mismatches (digit vs word numerals, punctuation, name misspellings) that compound into argument extraction errors. By skipping the text layer entirely, the E2E model learns direct audio-to-JSON mappings and avoids this information bottleneck.

5. **Selective Whisper layer unfreezing**: Rather than freezing the entire Whisper encoder (common practice) or fine-tuning all layers (risks catastrophic forgetting of speech representations), we unfreeze only the top 4 of 12 encoder layers (layers 8–11) at a 20x lower learning rate (1e-5 vs 2e-4 for the projector). The lower layers encode acoustic features (phonemes, spectral patterns) that generalize well; the upper layers encode higher-level semantic representations that benefit from adaptation to the target domain's audio characteristics (telephone-quality STOP recordings vs Whisper's web-crawled training data). This selective unfreezing adds ~90M trainable parameters but yields measurably better domain adaptation without destroying Whisper's pretrained robustness.

6. **LoRA over full fine-tuning**: At r=32, LoRA provides enough capacity for function-call generation while preserving the LLM's pretrained language structure. Attention-only targets (q/k/v/o) are sufficient; including MLP layers caused training crashes with 4-bit quantization.

7. **Direct embedding injection over cross-attention**: Rather than using cross-attention between audio and text (as in Flamingo-style models), audio embeddings are concatenated directly with the instruction prefix in the LLM's input space. This is simpler, parameter-efficient, and leverages the LLM's existing self-attention mechanism.

---

## Training

### Multi-Rate Optimization

A custom `TripleLRTrainer` assigns different learning rates to each component, reflecting their different optimization landscapes:

| Component | Learning Rate | Rationale |
|-----------|--------------|-----------|
| Whisper top 4 layers | 1e-5 | Gentle adaptation to preserve speech encoding |
| Audio projector + Conv1d | 2e-4 | Randomly initialized; needs faster learning |
| Qwen3 LoRA | 5e-5 | Fine-tuning on pretrained weights |

All three groups use AdamW-8bit with cosine LR scheduling and 100 warmup steps.

### Training Data

| Source | Samples | Type | Description |
|--------|---------|------|-------------|
| STOP (Meta) | 116K train / 10K val | Real speech | 800+ speakers, telephone quality, 30 functions |
| Synthetic | 12K train / 1.5K val | TTS (Edge TTS) | 8 voices, 30 functions, diverse entities |

**Warm-starting**: Training initializes from a prior checkpoint trained on SLURP + synthetic data (72.5% EM on those datasets), then fine-tunes on the larger STOP dataset.
---

## Results

### STOP Real Speech Test Set (21,729 samples)

| Metric | Score |
|--------|-------|
| **Exact Match** | **77.1%** |
| Function Correct | ~92% |
| Valid JSON | ~99.3% |
| Wrong Function | 1.5% |
| Wrong Arg Keys | 7.9% |
| Wrong Arg Values | 13.6% |

### Comparison to Published STOP Baselines

| Model | Architecture | Encoder | EM |
|-------|-------------|---------|-----|
| STOP Cascaded (Wav2Vec2 → BART) | Cascaded | Wav2Vec2-Large (300M) | ~76% |
| STOP E2E (HuBERT-Large Seq2Seq) | End-to-end | HuBERT-Large (300M) | 75.1% |
| STOP E2E Ensemble | End-to-end | HuBERT-Large (300M) | 75.9% |
| **Our Model** | **End-to-end** | **Whisper-small (39M)** | **77.1%** |

Our model matches or exceeds these baselines while using a **7.7x smaller encoder** (39M vs 300M parameters).

## On-Device Deployment (MLX)

The model was trained with bitsandbytes NF4 4-bit quantization (CUDA-only), but can be converted for local Apple Silicon inference via MLX. The conversion pipeline:

```
1. Load Qwen3-0.6B in float16 (not quantized)
2. Apply LoRA adapter weights (stored in float16)
3. merge_and_unload() → standard HuggingFace float16 model
4. mlx_lm.convert → MLX format (optionally 4-bit quantized)
```

The key insight: LoRA delta weights are always stored in float16, independent of the base model's quantization during training. Merging into an fp16 base avoids the known issues with `merge_and_unload()` on NF4 quantized models (rounding errors, crashes).

### MLX Architecture

The MLX port (`mlx_model.py`) mirrors the PyTorch `HybridQwen` exactly:

- **Whisper encoder**: via `mlx-whisper` library, with fine-tuned top-4 layers loaded from `projector.pt`
- **Conv1d + projector**: Custom `mlx.nn` module. Note: MLX Conv1d uses channels-last format `(batch, seq, channels)` — no transposes needed (unlike PyTorch). Weight conversion: PyTorch `(out, in, kernel)` → MLX `(out, kernel, in)`.
- **Qwen3 generation**: via `mlx_lm.load()` with KV cache. Custom greedy decode loop with brace-counting early stop (same as PyTorch version).

### Latency (M3 Max, 4-bit quantized)

| Stage | Time |
|-------|------|
| Whisper encoder + projector | ~130ms |
| Qwen3 prefill (380 tokens) | ~100ms |
| Qwen3 generation (~18 tokens) | ~220-350ms |
| **Total** | **~370-685ms** |

vs ~1.5-2s on Modal A10G (including network latency).

---

## Infrastructure

### Model Artifacts

```
best_model_qwen3_finetuned/
├── adapter_model.bin          # LoRA weights (26M params, ~100MB)
├── adapter_config.json        # PEFT configuration
├── projector.pt               # Conv1d + MLP + classifier (~2MB)
├── config.json                # Model metadata + function names
└── README.md
```

Total model size: ~102MB (excluding the base Qwen3-0.6B weights which are downloaded at runtime).

---

## Supported Functions

30 functions across 8 domains:

| Domain | Functions |
|--------|-----------|
| **Alarm** | alarm_set, alarm_cancel, alarm_check, alarm_silence, alarm_snooze, alarm_update |
| **Weather** | weather_get_current |
| **Navigation** | navigation_start, navigation_traffic, nav_duration, nav_eta, nav_distance |
| **Messaging** | message_send, message_read, message_react |
| **Media** | media_play_music, media_playback_control, music_playlist |
| **Calendar** | calendar_get_events |
| **Timer** | timer_set, timer_check, timer_modify, timer_pause, timer_resume, timer_cancel |
| **Reminder** | reminder_set, reminder_check, reminder_cancel, reminder_update |

---
