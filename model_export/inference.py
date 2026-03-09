#!/usr/bin/env python3
"""
Audio-to-Tool-Call Inference

Converts speech audio into structured JSON function calls.
Architecture: Whisper-small encoder → audio projector → Qwen3-0.6B (LoRA)

Supports 30 functions across 16 domains including alarms, weather,
navigation, messaging, media, calendar, smart home, and more.

Requirements:
    pip install torch transformers accelerate peft bitsandbytes librosa soundfile

Usage:
    python inference.py --audio path/to/audio.wav
    python inference.py --audio path/to/audio.mp3
    python inference.py --audio recording.wav --model-dir ./best_model_qwen3_finetuned
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import librosa
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    WhisperModel,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model


class HybridQwen(nn.Module):
    """Whisper encoder → audio projector → Qwen3 LLM → JSON tool call."""

    def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                 function_names=None):
        super().__init__()
        self.whisper = whisper_encoder
        self.whisper.eval()

        self.target_dtype = base_model.get_input_embeddings().weight.dtype
        hidden_size = base_model.config.hidden_size
        whisper_dim = whisper_encoder.config.d_model

        # Audio compression: 2x Conv1d (4x total) + LayerNorm
        self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1)
        self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1)
        self.conv_norm = nn.LayerNorm(whisper_dim)

        # Project whisper dim → LLM hidden dim
        self.audio_proj = nn.Sequential(
            nn.Linear(whisper_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Classification head (not used for generation, kept for weight loading)
        num_funcs = len(function_names) if function_names else 1
        self.function_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_funcs),
        )

        self.llm = base_model
        self.tokenizer = tokenizer
        self.instruction = instruction_prompt
        self.instruction_ids = tokenizer.encode(
            self.instruction, add_special_tokens=False, return_tensors="pt"
        )

    def load_pretrained(self, checkpoint_dir):
        """Load projector + whisper top-layer weights from checkpoint."""
        projector_path = os.path.join(checkpoint_dir, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"No projector.pt found in {checkpoint_dir}")

        state_dict = torch.load(projector_path, map_location="cpu")
        self.audio_conv1.load_state_dict(state_dict["audio_conv1"])
        self.audio_conv2.load_state_dict(state_dict["audio_conv2"])
        self.conv_norm.load_state_dict(state_dict["conv_norm"])
        self.audio_proj.load_state_dict(state_dict["audio_proj"])

        if "function_classifier" in state_dict:
            try:
                self.function_classifier.load_state_dict(state_dict["function_classifier"])
            except RuntimeError:
                pass  # size mismatch is fine — classifier not used for generation

        if "whisper_top_layers" in state_dict:
            for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
            print(f"  Loaded projector + {len(state_dict['whisper_top_layers'])} whisper layers")
        else:
            print("  Loaded projector")

    def to(self, device):
        """Move all submodules to device."""
        super().to(device)
        self.instruction_ids = self.instruction_ids.to(device)
        return self

    @torch.no_grad()
    def generate(self, audio_values, max_new_tokens=128):
        """Generate JSON tool call from audio input."""
        # Encode audio through whisper + projector
        enc_out = self.whisper(audio_values).last_hidden_state
        x = enc_out.transpose(1, 2).float()
        x = torch.relu(self.audio_conv1(x))
        x = torch.relu(self.audio_conv2(x))
        x = x.transpose(1, 2)
        x = self.conv_norm(x)
        audio_embeds = self.audio_proj(x).to(self.target_dtype)

        bs = audio_embeds.shape[0]
        embed_tokens = self.llm.get_input_embeddings()

        # Prepend instruction tokens
        instr_ids = self.instruction_ids.expand(bs, -1)
        instr_embeds = embed_tokens(instr_ids)
        current_embeds = torch.cat([audio_embeds, instr_embeds], dim=1)

        # Greedy decode with brace-counting early stop
        generated_ids = []
        for _ in range(max_new_tokens):
            outputs = self.llm(inputs_embeds=current_embeds)
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
            token_id = next_token_id.item()
            generated_ids.append(token_id)

            if token_id == self.tokenizer.eos_token_id:
                break

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if text.strip().endswith('}'):
                if text.count('{') > 0 and text.count('{') == text.count('}'):
                    break

            next_embed = embed_tokens(next_token_id.unsqueeze(0))
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_model(model_dir, device="cuda"):
    """Load the full model from a checkpoint directory."""
    # Read config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
    whisper_model_name = config.get("whisper_model", "openai/whisper-small")
    instruction_prompt = config.get("instruction_prompt", "<|audio|>Convert to tool call JSON:\n")
    function_names = config.get("function_names", [])
    lora_r = config.get("lora_r", 32)

    print(f"Loading base LLM: {base_llm}")
    print(f"Loading Whisper: {whisper_model_name}")
    print(f"Functions: {len(function_names)}")

    # Load Qwen3 in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    llm = AutoModelForCausalLM.from_pretrained(
        base_llm, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA and load trained weights
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none",
    )
    llm = get_peft_model(llm, lora_config)

    adapter_path = os.path.join(model_dir, "adapter_model.bin")
    if os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        # Remap Unsloth -> PEFT key format
        remapped = {}
        for k, v in adapter_state.items():
            new_key = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped[new_key] = v
        missing, unexpected = llm.load_state_dict(remapped, strict=False)
        print(f"  LoRA loaded: {len(remapped)} tensors")

    # Load Whisper encoder
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to(device)

    # Assemble hybrid model
    model = HybridQwen(llm, whisper_encoder, tokenizer, instruction_prompt,
                       function_names=function_names)
    model.load_pretrained(model_dir)
    model.to(device)
    model.eval()

    return model, processor


def transcribe(model, processor, audio_path, device="cuda"):
    """Run inference on a single audio file. Returns JSON tool call string."""
    wav, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    audio_features = inputs.input_features.to(device)
    return model.generate(audio_features)


def main():
    parser = argparse.ArgumentParser(description="Audio → JSON Tool Call")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav, mp3, flac, etc.)")
    parser.add_argument("--model-dir", default="./best_model_qwen3_finetuned",
                        help="Path to model checkpoint directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    if not os.path.exists(args.model_dir):
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)

    print("Loading model...")
    model, processor = load_model(args.model_dir, device=args.device)
    print("Model ready.\n")

    result = transcribe(model, processor, args.audio, device=args.device)

    # Try to pretty-print if valid JSON
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(result)


if __name__ == "__main__":
    main()
