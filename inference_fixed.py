"""
Inference script for the fixed hybrid audio-to-tool-call model.

Key: Uses the same instruction prompt as training so the model knows to generate JSON.
Supports ASR keyword override to correct known collapse predictions.

Usage:
    modal run inference_fixed.py
    modal run inference_fixed.py --audio-path /path/to/audio.mp3
    modal run inference_fixed.py --asr-override  # enable ASR override
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
    .add_local_file("asr_override.py", "/root/asr_override.py")
)

data_vol = modal.Volume.from_name("training-data-vol")
model_vol = modal.Volume.from_name("hybrid-model-storage")

app = modal.App("hybrid-inference-fixed")

# Must match training!
DEFAULT_INSTRUCTION = "<|audio|>Convert to tool call JSON:\n"


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
    timeout=300,
)
def run_inference(audio_path: str = None, num_samples: int = 5, asr_override: bool = False):
    """Run inference on audio samples."""
    import torch
    import torch.nn as nn
    import json
    import librosa
    import numpy as np
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, WhisperModel, WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model

    sys.path.insert(0, "/root")
    from asr_override import COLLAPSE_TARGETS, asr_override as apply_asr_override

    class HybridQwen(nn.Module):
        def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                     version="v3", whisper_model_name="openai/whisper-tiny", function_names=None):
            super().__init__()
            self.whisper = whisper_encoder
            self.version = version
            for param in self.whisper.parameters():
                param.requires_grad = False

            self.target_dtype = base_model.get_input_embeddings().weight.dtype
            hidden_size = base_model.config.hidden_size

            if version == "v3":
                whisper_dim = whisper_encoder.config.d_model  # 384=tiny, 512=base, 768=small
                self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.conv_norm = nn.LayerNorm(whisper_dim).to("cuda")
                self.audio_proj = nn.Sequential(
                    nn.Linear(whisper_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ).to("cuda")
            else:
                self.audio_conv = nn.Conv1d(384, 384, kernel_size=2, stride=2).to("cuda")
                self.audio_proj = nn.Linear(384, hidden_size).to("cuda")
                self.activation = nn.GELU()

            # Classification head for function routing
            self.function_names = function_names or []
            num_funcs = len(self.function_names) if self.function_names else 1
            self.function_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_funcs),
            ).to("cuda")

            self.llm = base_model
            self.tokenizer = tokenizer

            self.instruction = instruction_prompt
            self.instruction_ids = tokenizer.encode(
                self.instruction,
                add_special_tokens=False,
                return_tensors="pt"
            ).to("cuda")

        def load_pretrained(self, checkpoint_dir):
            """Load projector weights."""
            projector_path = os.path.join(checkpoint_dir, "projector.pt")
            if os.path.exists(projector_path):
                print(f"Loading projector weights from {projector_path}")
                state_dict = torch.load(projector_path, map_location="cuda")
                if self.version == "v3":
                    self.audio_conv1.load_state_dict(state_dict["audio_conv1"])
                    self.audio_conv2.load_state_dict(state_dict["audio_conv2"])
                    self.conv_norm.load_state_dict(state_dict["conv_norm"])
                    self.audio_proj.load_state_dict(state_dict["audio_proj"])
                else:
                    self.audio_conv.load_state_dict(state_dict["audio_conv"])
                    self.audio_proj.load_state_dict(state_dict["audio_proj"])
                if "function_classifier" in state_dict and self.function_names:
                    self.function_classifier.load_state_dict(state_dict["function_classifier"])
                    print("Projector + classifier loaded!")
                else:
                    print("Projector loaded!")
                if "whisper_top_layers" in state_dict:
                    for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                        self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
                    print(f"Whisper top {len(state_dict['whisper_top_layers'])} layers loaded!")

        def encode_audio(self, audio_values, debug=False):
            """Encode audio to embeddings."""
            with torch.no_grad():
                enc_out = self.whisper(audio_values).last_hidden_state

            if debug:
                print(f"  Whisper output: shape={enc_out.shape}, norm={enc_out.norm().item():.4f}")

            if self.version == "v3":
                x = enc_out.transpose(1, 2).float()
                x = torch.relu(self.audio_conv1(x))
                x = torch.relu(self.audio_conv2(x))
                x = x.transpose(1, 2)
                x = self.conv_norm(x)
                if debug:
                    print(f"  After 2x conv: shape={x.shape}, norm={x.norm().item():.4f}")
                projected = self.audio_proj(x)
            else:
                enc_out = enc_out.transpose(1, 2).float()
                enc_out = self.audio_conv(enc_out)
                enc_out = enc_out.transpose(1, 2)
                if debug:
                    print(f"  After conv: shape={enc_out.shape}, norm={enc_out.norm().item():.4f}")
                projected = self.audio_proj(enc_out)
                projected = self.activation(projected)

            if debug:
                print(f"  After proj: shape={projected.shape}, norm={projected.norm().item():.4f}")

            return projected.to(self.target_dtype)

        def classify_function(self, audio_embeds):
            """Classify function from mean-pooled audio embeddings."""
            pooled = audio_embeds.mean(dim=1).float()
            return self.function_classifier(pooled)

        @torch.no_grad()
        def generate(self, audio_values, max_new_tokens=128, debug=False):
            """Generate tool call from audio with classifier-constrained decoding."""
            audio_embeds = self.encode_audio(audio_values, debug=debug)
            bs = audio_embeds.shape[0]

            embed_tokens = self.llm.get_input_embeddings()

            instr_ids = self.instruction_ids.expand(bs, -1)
            instr_embeds = embed_tokens(instr_ids)

            # Free generation (classifier disabled — undertrained after 3 epochs)
            forced_prefix = ""
            current_embeds = torch.cat([audio_embeds, instr_embeds], dim=1)

            if debug:
                print(f"  Initial embeds shape: {current_embeds.shape}")

            generated_ids = []
            for step in range(max_new_tokens):
                outputs = self.llm(inputs_embeds=current_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = next_token_logits.argmax(dim=-1)

                token_id = next_token_id.item()
                generated_ids.append(token_id)

                if debug and step < 5:
                    decoded = self.tokenizer.decode([token_id])
                    print(f"  Step {step}: token={token_id}, decoded={repr(decoded)}")

                if token_id == self.tokenizer.eos_token_id:
                    break

                full_text = forced_prefix + self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if full_text.strip().endswith('}'):
                    open_braces = full_text.count('{')
                    close_braces = full_text.count('}')
                    if open_braces > 0 and open_braces == close_braces:
                        break

                next_token_embed = embed_tokens(next_token_id.unsqueeze(0))
                current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return forced_prefix + generated_text

    print("=" * 60)
    print("HYBRID MODEL INFERENCE (FIXED)")
    print("=" * 60)

    # Load model - try latest first, fall back
    for candidate in ["/models/final_model_qwen3_finetuned", "/models/final_model_v3b", "/models/final_model_v3", "/models/final_model_fixed_v2", "/models/final_model_fixed"]:
        if os.path.exists(candidate):
            model_path = candidate
            break
    else:
        print("No model found. Available:")
        for item in os.listdir("/models"):
            print(f"  /models/{item}")
        raise FileNotFoundError("No trained model found")

    print(f"\nLoading model from {model_path}...")

    # Load config
    config_path = os.path.join(model_path, "config.json")
    model_version = "v1"
    whisper_model_name = "openai/whisper-tiny"
    instruction_prompt = DEFAULT_INSTRUCTION
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
        model_version = config.get("version", "v1")
        whisper_model_name = config.get("whisper_model", "openai/whisper-tiny")
    function_names = config.get("function_names", [])
    print(f"Model version: {model_version}")
    print(f"Whisper model: {whisper_model_name}")
    print(f"Function names: {len(function_names)} functions")
    print(f"Instruction: {repr(instruction_prompt)}")

    # Load base model in 4-bit to match training (Unsloth uses bitsandbytes NF4 internally)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base_llm = config.get("base_llm", "unsloth/Qwen2.5-0.5B-Instruct")
    print(f"Base LLM: {base_llm}")
    model = AutoModelForCausalLM.from_pretrained(
        base_llm,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_llm)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA (matching training config) then load trained weights
    lora_r = config.get("lora_r", 16)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
    )
    print(f"LoRA rank: {lora_r}")
    model = get_peft_model(model, lora_config)

    adapter_bin_path = os.path.join(model_path, "adapter_model.bin")
    if os.path.exists(adapter_bin_path):
        print("Loading trained LoRA weights...")
        adapter_state = torch.load(adapter_bin_path, map_location="cpu")

        # Remap keys: Unsloth saves with base_model.model.model prefix,
        # PEFT expects base_model.model.model prefix + .default in lora layers
        remapped_state = {}
        for k, v in adapter_state.items():
            new_key = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped_state[new_key] = v

        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        print(f"  LoRA keys loaded: {len(remapped_state)}")
        print(f"  Missing (base model params, expected): {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")
        print("LoRA adapter loaded!")
    else:
        print(f"WARNING: No adapter found at {adapter_bin_path}")

    # Load Whisper (matching training)
    print(f"Loading Whisper: {whisper_model_name}")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

    # Create hybrid model
    hybrid_model = HybridQwen(model, whisper_encoder, tokenizer, instruction_prompt,
                              version=model_version, whisper_model_name=whisper_model_name,
                              function_names=function_names)
    hybrid_model.load_pretrained(model_path)
    hybrid_model.eval()

    # Load full Whisper for ASR override
    whisper_asr = None
    if asr_override:
        print("Loading full Whisper model for ASR override...")
        whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        whisper_asr.eval()
        print(f"ASR override enabled. Collapse targets: {sorted(COLLAPSE_TARGETS)}")

    # Get test samples
    if audio_path:
        test_samples = [{"audio_path": audio_path, "expected": "N/A"}]
    else:
        # Load from TRAIN set to verify model learned
        metadata_path = "/data/synthetic_full/train/metadata.jsonl"
        print(f"\nTesting with TRAINING samples to verify model learned...")
        test_samples = []
        with open(metadata_path) as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                entry = json.loads(line)
                audio_file = os.path.join("/data/synthetic_full", entry["audio_path"])
                if os.path.exists(audio_file):
                    test_samples.append({
                        "audio_path": audio_file,
                        "expected": json.dumps(entry["tool_call"]),
                        "transcript": entry.get("transcript", "")
                    })

    print(f"\nRunning inference on {len(test_samples)} samples...\n")

    results = []
    for i, sample in enumerate(test_samples):
        print(f"--- Sample {i+1} ---")

        try:
            wav, _ = librosa.load(sample["audio_path"], sr=16000)
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue

        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        audio_values = inputs.input_features.to("cuda")

        # Generate (debug first sample)
        prediction = hybrid_model.generate(audio_values, debug=(i == 0))

        # ASR override
        asr_transcript = None
        if asr_override and whisper_asr is not None:
            try:
                pred_parsed = json.loads(prediction)
                pred_func = pred_parsed.get("function", "") if isinstance(pred_parsed, dict) else ""
            except (json.JSONDecodeError, TypeError):
                pred_func = ""

            if pred_func in COLLAPSE_TARGETS:
                with torch.no_grad():
                    asr_ids = whisper_asr.generate(audio_values, max_new_tokens=128)
                asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

                corrected_json, override_info = apply_asr_override(
                    pred_func, asr_transcript, pred_parsed
                )
                if corrected_json is not None:
                    print(f"  ASR OVERRIDE: '{asr_transcript}' -> {pred_func} => {corrected_json['function']}")
                    prediction = json.dumps(corrected_json)

        print(f"Audio: {os.path.basename(sample['audio_path'])}")
        if "transcript" in sample:
            print(f"Transcript: {sample['transcript']}")
        if asr_transcript:
            print(f"ASR: {asr_transcript}")
        print(f"Expected: {sample['expected']}")
        print(f"Predicted: {prediction}")
        print()

        results.append({
            "audio": sample["audio_path"],
            "expected": sample["expected"],
            "predicted": prediction
        })

    # Calculate accuracy (exact match)
    correct = sum(1 for r in results if r["expected"] == r["predicted"])
    print("=" * 60)
    print(f"INFERENCE COMPLETE: {correct}/{len(results)} exact matches")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(audio_path: str = None, num_samples: int = 5, asr_override: bool = False):
    results = run_inference.remote(audio_path=audio_path, num_samples=num_samples, asr_override=asr_override)
    print(f"\nReturned {len(results)} results")
