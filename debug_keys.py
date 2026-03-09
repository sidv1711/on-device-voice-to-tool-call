"""Debug: compare saved adapter keys vs expected PEFT keys for Qwen3."""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "transformers", "accelerate", "peft", "bitsandbytes",
    )
)

model_vol = modal.Volume.from_name("hybrid-model-storage")
app = modal.App("debug-keys")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_vol},
    timeout=600,
)
def check_keys(model_dir: str = "final_model_qwen3_finetuned"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    model_path = f"/models/{model_dir}"

    # 1. Load saved adapter keys
    adapter_path = os.path.join(model_path, "adapter_model.bin")
    saved_state = torch.load(adapter_path, map_location="cpu")
    print(f"\n=== SAVED ADAPTER KEYS ({len(saved_state)} keys) ===")
    for k in sorted(saved_state.keys())[:20]:
        print(f"  {k}  shape={saved_state[k].shape}")
    print(f"  ... ({len(saved_state)} total)")

    # 2. Load base model with PEFT and check expected keys
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Qwen3-0.6B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)

    expected_keys = [k for k in peft_model.state_dict().keys() if "lora" in k]
    print(f"\n=== EXPECTED PEFT KEYS ({len(expected_keys)} lora keys) ===")
    for k in sorted(expected_keys)[:20]:
        print(f"  {k}")
    print(f"  ... ({len(expected_keys)} total)")

    # 3. Try the current remapping and see what matches
    remapped = {}
    for k, v in saved_state.items():
        new_key = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
        new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        remapped[new_key] = v

    print(f"\n=== REMAPPED KEYS (first 10) ===")
    for k in sorted(remapped.keys())[:10]:
        print(f"  {k}")

    # 4. Check overlap
    expected_set = set(expected_keys)
    remapped_set = set(remapped.keys())
    matched = expected_set & remapped_set
    missing = expected_set - remapped_set
    unexpected = remapped_set - expected_set

    print(f"\n=== MATCH REPORT ===")
    print(f"  Matched: {len(matched)}/{len(expected_set)}")
    print(f"  Missing from expected: {len(missing)}")
    print(f"  Unexpected (in saved, not in expected): {len(unexpected)}")

    if missing:
        print(f"\n  Sample missing keys:")
        for k in sorted(missing)[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"\n  Sample unexpected keys:")
        for k in sorted(unexpected)[:5]:
            print(f"    {k}")


@app.local_entrypoint()
def main(model_dir: str = "final_model_qwen3_finetuned"):
    check_keys.remote(model_dir=model_dir)
