"""
Merge LoRA into Qwen3 base and save as float16 HF model for MLX conversion.

Key insight: Load base in float16 (NOT quantized), apply LoRA, merge.
LoRA weights are stored in float16 regardless of training quantization.
Merging into NF4 is lossy and crashes — this approach avoids it entirely.

Usage:
    modal run merge_export.py
    # Then download:
    modal volume get hybrid-model-storage qwen3_merged_f16/ ./model_export/qwen3_merged_f16/
"""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch", "transformers>=4.51.0", "accelerate", "peft",
        "safetensors",
    )
)

model_vol = modal.Volume.from_name("hybrid-model-storage")
app = modal.App("runanywhere-merge-export")

MODEL_DIR = "/models/best_model_qwen3_finetuned"
OUTPUT_DIR = "/models/qwen3_merged_f16"

# Use the standard (non-quantized) Qwen3 base for fp16 merge
BASE_MODEL_FP16 = "Qwen/Qwen3-0.6B"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": model_vol},
    timeout=600,
)
def merge_and_export():
    import torch
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # --- Load config ---
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_r = config.get("lora_r", 32)

    # --- Load base model in FLOAT16 (not quantized!) ---
    print(f"Loading {BASE_MODEL_FP16} in float16...")
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_FP16,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_FP16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Apply LoRA config (same as training) ---
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
    )
    llm = get_peft_model(llm, lora_config)

    # --- Load LoRA weights ---
    adapter_path = os.path.join(MODEL_DIR, "adapter_model.bin")
    safetensors_path = os.path.join(MODEL_DIR, "adapter_model.safetensors")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path)
        print(f"Loaded adapter from safetensors ({len(adapter_state)} tensors)")
    elif os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        print(f"Loaded adapter from bin ({len(adapter_state)} tensors)")
    else:
        raise FileNotFoundError("No adapter weights found in model dir")

    # Remap keys: Unsloth format → PEFT format
    remapped = {}
    for k, v in adapter_state.items():
        nk = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
        nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
        nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
        remapped[nk] = v

    missing, unexpected = llm.load_state_dict(remapped, strict=False)
    print(f"LoRA loaded: {len(missing)} missing (base params, expected), {len(unexpected)} unexpected")
    if unexpected:
        print(f"WARNING: unexpected keys: {unexpected[:5]}")

    # --- Merge LoRA into float16 base ---
    print("Merging LoRA into float16 base (safe_merge=True)...")
    merged = llm.merge_and_unload(safe_merge=True)

    # --- Save as standard HuggingFace model ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving merged float16 model to {OUTPUT_DIR}...")
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Copy over our custom config for reference
    import shutil
    shutil.copy(config_path, os.path.join(OUTPUT_DIR, "runanywhere_config.json"))

    model_vol.commit()

    # Verify output
    output_files = os.listdir(OUTPUT_DIR)
    print(f"Output files: {output_files}")
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in output_files
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    )
    print(f"Total size: {total_size / 1e6:.1f} MB")
    print("Done! Download with:")
    print(f"  modal volume get hybrid-model-storage qwen3_merged_f16/ ./model_export/qwen3_merged_f16/")


@app.local_entrypoint()
def main():
    merge_and_export.remote()
