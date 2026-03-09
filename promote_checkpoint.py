#!/usr/bin/env python3
"""
Promote a Trainer checkpoint (pytorch_model.bin) to final model format.

Loads a checkpoint's weights into the HybridQwen model built from the v3b base,
then saves in the standard final model format (adapter_model.bin + projector.pt).

Usage:
    modal run promote_checkpoint.py --checkpoint checkpoints_v3b_finetune/checkpoint-2727
"""
import modal

app = modal.App("promote-checkpoint")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.1.2", "transformers==4.41.2", "bitsandbytes==0.43.1",
        "accelerate==0.30.1", "peft==0.10.0",
    )
)

model_vol = modal.Volume.from_name("hybrid-model-storage")


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/models": model_vol},
)
def promote(checkpoint: str, output: str = "final_model_v3b_finetuned"):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig

    checkpoint_path = f"/models/{checkpoint}"
    v3b_path = "/models/final_model_v3b"
    output_path = f"/models/{output}"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base: {v3b_path}")
    print(f"Output: {output_path}")

    # Load checkpoint state dict
    ckpt_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
    print(f"\nLoading checkpoint weights from {ckpt_bin}...")
    ckpt_state = torch.load(ckpt_bin, map_location="cpu")
    ckpt_keys = list(ckpt_state.keys())
    print(f"  {len(ckpt_keys)} keys in checkpoint")

    # Separate projector vs LoRA keys
    proj_keys = [k for k in ckpt_keys if any(k.startswith(p) for p in
        ("audio_conv1.", "audio_conv2.", "conv_norm.", "audio_proj.", "function_classifier."))]
    lora_keys = [k for k in ckpt_keys if "lora_" in k or ".default." in k]
    print(f"  Projector keys: {len(proj_keys)}")
    print(f"  LoRA keys: {len(lora_keys)}")

    # Build projector.pt
    proj_state = {}
    for key in proj_keys:
        # strip top-level prefix if present (e.g. "audio_conv1.weight")
        proj_state[key] = ckpt_state[key]

    # Group into sub-dicts matching save format
    def extract_subdict(state, prefix):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    projector_save = {
        "audio_conv1": extract_subdict(proj_state, "audio_conv1."),
        "audio_conv2": extract_subdict(proj_state, "audio_conv2."),
        "conv_norm": extract_subdict(proj_state, "conv_norm."),
        "audio_proj": extract_subdict(proj_state, "audio_proj."),
    }
    fc_dict = extract_subdict(proj_state, "function_classifier.")
    if fc_dict:
        projector_save["function_classifier"] = fc_dict
    os.makedirs(output_path, exist_ok=True)
    torch.save(projector_save, os.path.join(output_path, "projector.pt"))
    print("  Saved projector.pt")

    # Build adapter_model.bin — remap keys to PEFT format
    # Checkpoint keys look like: llm.base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
    # PEFT adapter format:       base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
    adapter_state = {}
    for k, v in ckpt_state.items():
        if "lora_" not in k and ".default." not in k:
            continue
        # strip leading "llm." prefix
        if k.startswith("llm."):
            new_k = k[len("llm."):]
        else:
            new_k = k
        adapter_state[new_k] = v
    print(f"  Remapped {len(adapter_state)} LoRA keys")

    # Copy config files from v3b base
    import shutil
    for fname in ["adapter_config.json", "config.json"]:
        src = os.path.join(v3b_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, fname))
            print(f"  Copied {fname}")

    # Also copy README if present
    readme = os.path.join(v3b_path, "README.md")
    if os.path.exists(readme):
        shutil.copy(readme, os.path.join(output_path, "README.md"))

    # Save adapter weights
    torch.save(adapter_state, os.path.join(output_path, "adapter_model.bin"))
    print(f"  Saved adapter_model.bin ({len(adapter_state)} keys)")

    # Write config.json for the HybridModel
    import json
    model_config = {
        "instruction_prompt": "<|audio|>Convert to tool call JSON:\n",
        "version": "v3b_finetuned",
        "whisper_model": "openai/whisper-base",
        "base_model": "v3b",
        "finetune": "slurp_mixed",
        "promoted_from": checkpoint,
    }
    with open(os.path.join(output_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    model_vol.commit()
    print(f"\nDone. Promoted {checkpoint} → {output_path}")
    print(f"Files: {os.listdir(output_path)}")


@app.local_entrypoint()
def main(checkpoint: str = "checkpoints_v3b_finetune/checkpoint-2727",
         output: str = "final_model_v3b_finetuned"):
    promote.remote(checkpoint=checkpoint, output=output)
