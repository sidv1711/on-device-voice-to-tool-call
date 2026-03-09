"""
Fine-tune Qwen3-0.6B (text-only LoRA) on transcript -> JSON tool calls.

Cascaded approach: Whisper ASR handles audio transcription separately,
this model only learns text -> structured JSON mapping.

Uses merged_stop dataset's `transcript` field instead of audio.

Usage:
    modal run finetune_text.py
    modal run finetune_text.py --test-run  # Quick test with 100 samples
"""
import modal
import os
import json

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers", "accelerate", "peft", "bitsandbytes",
        "numpy", "trl",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    )
)

data_vol = modal.Volume.from_name("training-data-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("hybrid-model-storage", create_if_missing=True)

app = modal.App("cascaded-text-finetune")

INSTRUCTION_PREFIX = "Convert to tool call JSON:\n"


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
    timeout=86400,
)
def finetune(
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    max_samples: int = None,
):
    import torch
    import random
    from torch.utils.data import Dataset
    from unsloth import FastLanguageModel
    import transformers
    from transformers import TrainingArguments, Trainer
    import shutil

    # ── Find training data ──
    meta_path = None
    for dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
        candidate = f"/data/{dir_name}/train/metadata.jsonl"
        if os.path.exists(candidate):
            meta_path = candidate
            break
    if not meta_path:
        raise FileNotFoundError("No training metadata found in /data/")

    val_meta_path = None
    for dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
        candidate = f"/data/{dir_name}/val/metadata.jsonl"
        if os.path.exists(candidate):
            val_meta_path = candidate
            break

    # ── Dataset ──

    class TextToolCallDataset(Dataset):
        """Text transcript -> JSON tool call dataset."""

        def __init__(self, metadata_path, tokenizer, max_samples=None, max_input_len=256, max_output_len=128):
            self.tokenizer = tokenizer
            self.max_input_len = max_input_len
            self.max_output_len = max_output_len
            self.max_total_len = max_input_len + max_output_len
            self.data = []

            print(f"Loading samples from {metadata_path}...")
            with open(metadata_path) as f:
                for line in f:
                    entry = json.loads(line)
                    transcript = entry.get("transcript", "").strip()
                    if not transcript:
                        continue
                    self.data.append({
                        "transcript": transcript,
                        "target": json.dumps(entry["tool_call"]),
                    })

            random.seed(42)
            random.shuffle(self.data)
            if max_samples:
                self.data = self.data[:max_samples]
            print(f"  Loaded {len(self.data)} samples")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            prompt = INSTRUCTION_PREFIX + item["transcript"]
            full_text = prompt + item["target"]

            # Tokenize the full sequence
            full_tokens = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_total_len,
                truncation=True,
            )

            # Tokenize just the prompt to find where labels start
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_len,
            )
            prompt_len = prompt_tokens.input_ids.shape[1]

            input_ids = full_tokens.input_ids[0]
            attention_mask = full_tokens.attention_mask[0]

            # Labels: mask prompt prefix with -100, keep target tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            # Mask padding
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    # ── Best checkpoint callback ──

    class BestCheckpointCallback(transformers.TrainerCallback):
        """Saves model whenever eval loss improves."""

        def __init__(self, model_ref, tokenizer_ref, save_path, volume):
            self.model_ref = model_ref
            self.tokenizer_ref = tokenizer_ref
            self.save_path = save_path
            self.volume = volume
            self.best_eval_loss = float("inf")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            eval_loss = metrics.get("eval_loss")
            if eval_loss is None:
                return
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                step = state.global_step
                epoch = state.epoch or 0
                print(f"\n  >> New best eval_loss={eval_loss:.4f} at step {step} (epoch {epoch:.1f})")
                self.model_ref.save_pretrained(self.save_path, safe_serialization=False)
                # Write config AFTER save_pretrained to avoid HF overwriting
                with open(os.path.join(self.save_path, "cascaded_config.json"), "w") as f:
                    json.dump({
                        "model_type": "cascaded_text",
                        "base_llm": "unsloth/Qwen3-0.6B",
                        "lora_r": 32,
                        "instruction_prefix": INSTRUCTION_PREFIX,
                    }, f)
                self.volume.commit()
                print(f"  >> Best model saved to {self.save_path}")

    # ── MAIN ──
    print("=" * 60)
    print("CASCADED TEXT MODEL FINE-TUNING")
    print("  Qwen3-0.6B + LoRA on transcript -> JSON")
    print("=" * 60)
    print(f"LR: {lr}, Epochs: {num_epochs}, Batch size: {batch_size}")

    # Load base model with Unsloth (4-bit quantized)
    BASE_LLM = "unsloth/Qwen3-0.6B"
    print(f"\nLoading base model: {BASE_LLM}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_LLM,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    print("\nBuilding datasets...")
    train_dataset = TextToolCallDataset(meta_path, tokenizer, max_samples=max_samples)

    val_dataset = None
    if val_meta_path:
        val_max = min(1000, max_samples) if max_samples else 1000
        val_dataset = TextToolCallDataset(val_meta_path, tokenizer, max_samples=val_max)

    if len(train_dataset) == 0:
        raise ValueError("No training samples loaded!")

    # Training arguments — clear old checkpoints
    output_dir = "/models/checkpoints_cascaded_text"
    if max_samples:
        output_dir = "/models/checkpoints_cascaded_text_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared stale checkpoints: {output_dir}")

    total_steps = len(train_dataset) * num_epochs // batch_size
    print(f"Estimated total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=500 if val_dataset else None,
        load_best_model_at_end=False,
    )

    best_save_path = "/models/best_cascaded_text_model"
    best_callback = BestCheckpointCallback(model, tokenizer, best_save_path, model_vol)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[best_callback],
    )

    print(f"\n{'='*60}")
    print("STARTING FINE-TUNING")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
    print(f"  LR: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    save_path = "/models/cascaded_text_model"
    model.save_pretrained(save_path, safe_serialization=False)
    with open(os.path.join(save_path, "cascaded_config.json"), "w") as f:
        json.dump({
            "model_type": "cascaded_text",
            "base_llm": "unsloth/Qwen3-0.6B",
            "lora_r": 32,
            "instruction_prefix": INSTRUCTION_PREFIX,
        }, f)
    model_vol.commit()

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"Final model saved to: {save_path}")
    print(f"Best model saved to: {best_save_path} (eval_loss={best_callback.best_eval_loss:.4f})")
    print(f"{'='*60}")

    return {
        "status": "success",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "best_eval_loss": best_callback.best_eval_loss,
    }


@app.local_entrypoint()
def main(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    max_samples: int = None,
    test_run: bool = False,
):
    if test_run:
        max_samples = 100
        epochs = 1

    result = finetune.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
    )
    print(f"Result: {result}")
