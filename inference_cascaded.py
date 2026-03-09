"""
Cascaded inference: Whisper ASR -> text-only Qwen3 LoRA -> JSON tool call.

Two-stage pipeline that sidesteps audio domain gaps:
  1. Whisper ASR transcribes audio to text (handles diverse audio well)
  2. Text-only Qwen3 LoRA maps transcript to JSON tool call

Usage:
    modal run inference_cascaded.py
    modal run inference_cascaded.py --audio-path /path/to/audio.mp3
    modal run inference_cascaded.py --text-model-dir best_cascaded_text_model
"""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "transformers", "accelerate", "peft", "bitsandbytes",
        "librosa", "soundfile", "numpy"
    )
)

data_vol = modal.Volume.from_name("training-data-vol")
model_vol = modal.Volume.from_name("hybrid-model-storage")

app = modal.App("cascaded-inference")

INSTRUCTION_PREFIX = "Convert to tool call JSON:\n"


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
    timeout=300,
)
def run_inference(
    audio_path: str = None,
    num_samples: int = 5,
    text_model_dir: str = "best_cascaded_text_model",
):
    """Run cascaded inference: Whisper ASR -> text model -> JSON."""
    import torch
    import json
    import librosa
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        WhisperForConditionalGeneration, WhisperProcessor,
    )
    from peft import LoraConfig, get_peft_model

    print("=" * 60)
    print("CASCADED INFERENCE (Whisper ASR -> Text Model)")
    print("=" * 60)

    # ── Stage 1: Load Whisper ASR ──
    whisper_model_name = "openai/whisper-small"
    print(f"\nLoading Whisper ASR: {whisper_model_name}")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
    whisper_asr.eval()
    print("Whisper ASR ready.")

    # ── Stage 2: Load text-only Qwen3 LoRA ──
    model_path = f"/models/{text_model_dir}"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Available:")
        for item in os.listdir("/models"):
            print(f"  /models/{item}")
        raise FileNotFoundError(f"No model at {model_path}")

    # Load cascaded config
    config_path = os.path.join(model_path, "cascaded_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
    lora_r = config.get("lora_r", 32)
    instruction_prefix = config.get("instruction_prefix", INSTRUCTION_PREFIX)
    print(f"Text model: {model_path}")
    print(f"Base LLM: {base_llm}")
    print(f"LoRA rank: {lora_r}")

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_llm,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA then load trained weights
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none",
    )
    model = get_peft_model(model, lora_config)

    adapter_bin_path = os.path.join(model_path, "adapter_model.bin")
    if os.path.exists(adapter_bin_path):
        print("Loading trained LoRA weights...")
        adapter_state = torch.load(adapter_bin_path, map_location="cpu")

        # Remap keys (Unsloth -> PEFT .default convention)
        remapped_state = {}
        for k, v in adapter_state.items():
            new_key = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped_state[new_key] = v

        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        print(f"  LoRA loaded: {len(remapped_state)} tensors, {len(missing)} missing (base), {len(unexpected)} unexpected")
    else:
        print(f"WARNING: No adapter found at {adapter_bin_path}")

    model.eval()
    print("Text model ready.")

    # ── Generate function ──

    @torch.no_grad()
    def generate_from_text(transcript, max_new_tokens=128):
        """Generate JSON tool call from transcript using greedy decoding with brace-counting early stop."""
        prompt = instruction_prefix + transcript
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids

        generated_ids = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1)

            token_id = next_token_id.item()
            generated_ids.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break

            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if full_text.strip().endswith('}'):
                open_b = full_text.count('{')
                close_b = full_text.count('}')
                if open_b > 0 and open_b == close_b:
                    break

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Get test samples ──
    if audio_path:
        test_samples = [{"audio_path": audio_path, "expected": "N/A"}]
    else:
        metadata_path = None
        for dir_name in ["merged_stop", "merged_new", "merged"]:
            candidate = f"/data/{dir_name}/test/metadata.jsonl"
            if os.path.exists(candidate):
                metadata_path = candidate
                break
        if not metadata_path:
            metadata_path = "/data/synthetic_full/train/metadata.jsonl"

        print(f"\nLoading test samples from {metadata_path}...")
        test_samples = []
        with open(metadata_path) as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                entry = json.loads(line)

                raw_path = entry.get("audio_path", "")
                source = entry.get("source", "")
                if source == "slurp":
                    audio_file = f"/data/slurp_audio/{os.path.basename(raw_path)}"
                elif source == "stop":
                    audio_file = f"/data/{raw_path}"
                else:
                    audio_file = os.path.join("/data/synthetic_full", raw_path)

                if os.path.exists(audio_file):
                    test_samples.append({
                        "audio_path": audio_file,
                        "expected": json.dumps(entry["tool_call"]),
                        "transcript": entry.get("transcript", ""),
                    })

    print(f"\nRunning cascaded inference on {len(test_samples)} samples...\n")

    results = []
    for i, sample in enumerate(test_samples):
        print(f"--- Sample {i+1} ---")

        try:
            wav, _ = librosa.load(sample["audio_path"], sr=16000)
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue

        # Stage 1: Whisper ASR
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        audio_features = inputs.input_features.to("cuda")
        with torch.no_grad():
            asr_ids = whisper_asr.generate(audio_features, max_length=448)
        asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

        # Stage 2: Text model -> JSON (uppercase to match training convention)
        prediction = generate_from_text(asr_transcript.upper())

        print(f"Audio: {os.path.basename(sample['audio_path'])}")
        if "transcript" in sample:
            print(f"Gold transcript: {sample['transcript']}")
        print(f"ASR transcript:  {asr_transcript}")
        print(f"Expected:  {sample['expected']}")
        print(f"Predicted: {prediction}")
        print()

        results.append({
            "audio": sample["audio_path"],
            "asr_transcript": asr_transcript,
            "expected": sample["expected"],
            "predicted": prediction,
        })

    correct = sum(1 for r in results if r["expected"] == r["predicted"])
    print("=" * 60)
    print(f"CASCADED INFERENCE COMPLETE: {correct}/{len(results)} exact matches")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(
    audio_path: str = None,
    num_samples: int = 5,
    text_model_dir: str = "best_cascaded_text_model",
):
    results = run_inference.remote(
        audio_path=audio_path,
        num_samples=num_samples,
        text_model_dir=text_model_dir,
    )
    print(f"\nReturned {len(results)} results")
