"""
Fine-tune v3b model on STOP real speech data.

Strategy:
- Single-stage joint fine-tuning (whisper top layers + projector + LoRA) with triple learning rates
- 221K STOP real speech samples across 30 functions
- Triple LRs: whisper 1e-5, LoRA 5e-5, projector 2e-4
- 3 epochs with validation on STOP val set, best-checkpoint selection

Usage:
    modal run finetune_v3b.py
    modal run finetune_v3b.py --test-run  # Quick test with 100 samples
"""
import modal
import os
import json

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch", "torchvision", "transformers", "accelerate", "peft", "bitsandbytes",
        "librosa", "soundfile", "numpy", "scipy", "trl",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    )
)

data_vol = modal.Volume.from_name("training-data-vol", create_if_missing=True)
model_vol = modal.Volume.from_name("hybrid-model-storage", create_if_missing=True)

app = modal.App("hybrid-finetune-v3b")

INSTRUCTION_PROMPT = "<|audio|>Convert to tool call JSON:\n"

# Will be populated at runtime from tool_schema.json
FUNCTION_NAMES = []


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
    timeout=86400,  # 24 hours
)
def finetune(
    num_epochs: int = 2,
    batch_size: int = 16,
    lora_lr: float = 5e-5,
    projector_lr: float = 2e-4,
    max_samples: int = None,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    from torch.utils.data import Dataset
    from unsloth import FastLanguageModel
    import transformers
    from transformers import (
        WhisperModel, WhisperProcessor,
        TrainingArguments, Trainer,
    )
    import librosa
    import numpy as np

    # ── Load function names for classification head ──
    global FUNCTION_NAMES

    # Extract function names from training metadata
    meta_path = None
    for data_dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
        candidate = f"/data/{data_dir_name}/train/metadata.jsonl"
        if os.path.exists(candidate):
            meta_path = candidate
            break
    if not meta_path:
        raise FileNotFoundError("No training metadata found in /data/")

    func_set = set()
    with open(meta_path) as f:
        for line in f:
            entry = json.loads(line)
            func_set.add(entry["tool_call"]["function"])
    FUNCTION_NAMES = sorted(func_set)
    FUNC_TO_IDX = {name: i for i, name in enumerate(FUNCTION_NAMES)}
    print(f"Function vocabulary: {len(FUNCTION_NAMES)} functions")

    def spec_augment(features, freq_mask_param=15, time_mask_param=70,
                     num_freq_masks=2, num_time_masks=2):
        """SpecAugment: frequency and time masking on mel features."""
        features = features.clone()
        num_freq, num_time = features.shape[-2], features.shape[-1]
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(1, num_freq - f))
            features[..., f0:f0+f, :] = 0
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, num_time))
            t0 = random.randint(0, max(1, num_time - t))
            features[..., :, t0:t0+t] = 0
        return features

    # ── HybridQwen with classification head ──

    class HybridQwen(nn.Module):
        def __init__(self, unsloth_model, whisper_encoder, tokenizer):
            super().__init__()
            self.whisper = whisper_encoder
            for param in self.whisper.parameters():
                param.requires_grad = False
            # Unfreeze top 4 encoder layers (8-11) for TTS adaptation
            for layer in self.whisper.layers[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True

            self.target_dtype = unsloth_model.get_input_embeddings().weight.dtype
            hidden_size = unsloth_model.config.hidden_size

            whisper_dim = whisper_encoder.config.d_model  # 512 for base, 768 for small
            self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
            self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
            self.conv_norm = nn.LayerNorm(whisper_dim).to("cuda")

            self.audio_proj = nn.Sequential(
                nn.Linear(whisper_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ).to("cuda")

            # Classification head for function routing
            num_functions = len(FUNCTION_NAMES)
            self.function_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_functions),
            ).to("cuda")

            self.llm = unsloth_model
            self.tokenizer = tokenizer

            self.instruction = INSTRUCTION_PROMPT
            self.instruction_ids = tokenizer.encode(
                self.instruction,
                add_special_tokens=False,
                return_tensors="pt"
            ).to("cuda")

        def encode_audio(self, audio_values):
            enc_out = self.whisper(audio_values).last_hidden_state

            x = enc_out.transpose(1, 2).float()
            x = torch.relu(self.audio_conv1(x))
            x = torch.relu(self.audio_conv2(x))
            x = x.transpose(1, 2)
            x = self.conv_norm(x)

            projected = self.audio_proj(x)
            return projected.to(self.target_dtype)

        def classify_function(self, audio_embeds):
            """Classify function from mean-pooled audio embeddings."""
            pooled = audio_embeds.mean(dim=1)  # [B, hidden_size]
            return self.function_classifier(pooled)  # [B, num_functions]

        def forward(self, audio_values, input_ids, labels=None, attention_mask=None,
                    function_labels=None):
            audio_embeds = self.encode_audio(audio_values)
            bs = audio_embeds.shape[0]
            audio_len = audio_embeds.shape[1]

            # Auxiliary classification loss
            cls_loss = None
            if function_labels is not None:
                cls_logits = self.classify_function(audio_embeds)
                cls_loss = F.cross_entropy(cls_logits, function_labels)

            embed_tokens = self.llm.get_input_embeddings()

            instr_ids = self.instruction_ids.expand(bs, -1)
            instr_embeds = embed_tokens(instr_ids)
            instr_len = instr_embeds.shape[1]

            text_embeds = embed_tokens(input_ids)

            combined_embeds = torch.cat([audio_embeds, instr_embeds, text_embeds], dim=1)

            prefix_len = audio_len + instr_len
            if labels is not None:
                pad_labels = torch.full(
                    (bs, prefix_len), -100,
                    device=labels.device, dtype=labels.dtype
                )
                combined_labels = torch.cat([pad_labels, labels], dim=1)
            else:
                combined_labels = None

            prefix_mask = torch.ones((bs, prefix_len), device=audio_embeds.device, dtype=torch.long)
            if attention_mask is not None:
                combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                combined_mask = torch.cat([prefix_mask, torch.ones_like(input_ids)], dim=1)

            output = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=combined_labels
            )

            # Classifier aux loss disabled — destabilizes generation quality
            # if cls_loss is not None and output.loss is not None:
            #     output.loss = output.loss + 0.1 * cls_loss

            return output

        def save_pretrained(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)

            torch.save({
                "audio_conv1": self.audio_conv1.state_dict(),
                "audio_conv2": self.audio_conv2.state_dict(),
                "conv_norm": self.conv_norm.state_dict(),
                "audio_proj": self.audio_proj.state_dict(),
                "function_classifier": self.function_classifier.state_dict(),
                "whisper_top_layers": {
                    i: layer.state_dict()
                    for i, layer in enumerate(self.whisper.layers[-4:], start=len(self.whisper.layers)-4)
                },
            }, os.path.join(output_dir, "projector.pt"))

            self.llm.save_pretrained(output_dir, safe_serialization=False)

            # Write AFTER llm.save_pretrained to avoid HF overwriting our config
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({
                    "instruction_prompt": self.instruction,
                    "version": "qwen3_finetuned",
                    "whisper_model": "openai/whisper-small",
                    "base_llm": "unsloth/Qwen3-0.6B",
                    "lora_r": 32,
                    "finetune": "slurp_mixed",
                    "function_names": FUNCTION_NAMES,
                }, f)

    # ── MixedFineTuneDataset ──

    class STOPTrainDataset(Dataset):
        """STOP real speech training dataset."""

        def __init__(self, data_dir, processor, tokenizer, max_samples=None,
                     augment=True):
            self.processor = processor
            self.tokenizer = tokenizer
            self.augment = augment
            self.data = []

            # Find training metadata
            metadata_path = None
            for dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
                candidate = os.path.join(data_dir, dir_name, "train", "metadata.jsonl")
                if os.path.exists(candidate):
                    metadata_path = candidate
                    break
            if not metadata_path:
                raise FileNotFoundError("No train metadata found")

            skipped = 0
            missing_audio = 0
            print(f"Loading samples from {metadata_path}...")
            with open(metadata_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    func_name = entry["tool_call"]["function"]

                    if func_name not in FUNC_TO_IDX:
                        skipped += 1
                        continue

                    # STOP audio: audio_path = "stop_audio/{split}/{filename}"
                    audio_path = os.path.join(data_dir, entry.get("audio_path", ""))

                    if os.path.exists(audio_path):
                        self.data.append({
                            "audio_path": audio_path,
                            "target": json.dumps(entry["tool_call"]),
                            "function_idx": FUNC_TO_IDX[func_name],
                            "domain": entry.get("domain", "unknown"),
                        })
                    else:
                        missing_audio += 1

            random.seed(42)
            random.shuffle(self.data)

            if max_samples:
                self.data = self.data[:max_samples]

            print(f"  Loaded {len(self.data)} samples (skipped {skipped} unknown funcs, {missing_audio} missing audio)")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            try:
                wav, _ = librosa.load(item["audio_path"], sr=16000)
            except Exception as e:
                print(f"Error loading {item['audio_path']}: {e}")
                wav = np.zeros(16000, dtype=np.float32)

            inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
            audio_features = inputs.input_features[0]

            # SpecAugment disabled — destabilizes fine-tuning on pretrained model
            # if self.augment:
            #     audio_features = spec_augment(audio_features)

            text_inputs = self.tokenizer(
                item["target"],
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )

            # Mask padding positions (attention_mask==0) to -100 for loss
            # Note: can't use labels[labels == pad_token_id] = -100 because
            # pad_token_id == eos_token_id in Qwen — would also mask real EOS
            labels = text_inputs.input_ids[0].clone()
            labels[text_inputs.attention_mask[0] == 0] = -100

            return {
                "audio": audio_features,
                "input_ids": text_inputs.input_ids[0],
                "labels": labels,
                "function_idx": item["function_idx"],
            }

    # ── Real-Speech Validation Dataset ──

    class RealSpeechValDataset(Dataset):
        """Real-speech validation dataset for best-checkpoint selection."""

        def __init__(self, data_dir, processor, tokenizer, max_samples=None):
            self.processor = processor
            self.tokenizer = tokenizer
            self.data = []

            # Find val metadata
            metadata_path = None
            for dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
                candidate = os.path.join(data_dir, dir_name, "val", "metadata.jsonl")
                if os.path.exists(candidate):
                    metadata_path = candidate
                    break

            if not metadata_path:
                print("WARNING: No validation set found, skipping eval")
                return

            print(f"  Loading val from: {metadata_path}")
            with open(metadata_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    func_name = entry["tool_call"]["function"]
                    if func_name not in FUNC_TO_IDX:
                        continue

                    audio_path = os.path.join(data_dir, entry.get("audio_path", ""))

                    if os.path.exists(audio_path):
                        self.data.append({
                            "audio_path": audio_path,
                            "target": json.dumps(entry["tool_call"]),
                            "function_idx": FUNC_TO_IDX[func_name],
                        })

            if max_samples:
                self.data = self.data[:max_samples]
            print(f"  Validation set: {len(self.data)} real-speech samples")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            try:
                wav, _ = librosa.load(item["audio_path"], sr=16000)
            except Exception as e:
                print(f"Error loading {item['audio_path']}: {e}")
                wav = np.zeros(16000, dtype=np.float32)

            inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
            # No SpecAugment on validation

            text_inputs = self.tokenizer(
                item["target"],
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )

            # Mask padding positions (attention_mask==0) to -100 for loss
            labels = text_inputs.input_ids[0].clone()
            labels[text_inputs.attention_mask[0] == 0] = -100

            return {
                "audio": inputs.input_features[0],
                "input_ids": text_inputs.input_ids[0],
                "labels": labels,
                "function_idx": item["function_idx"],
            }

    # ── Best checkpoint callback ──

    class BestCheckpointCallback(transformers.TrainerCallback):
        """Saves model in final format whenever eval loss improves."""

        def __init__(self, save_path, volume):
            self.save_path = save_path
            self.volume = volume
            self.best_eval_loss = float("inf")

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
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
                # Save using HybridQwen's save_pretrained
                hybrid = kwargs.get("model", model)
                if hybrid is None:
                    # Trainer passes model positionally in some versions
                    return
                hybrid.save_pretrained(self.save_path)
                self.volume.commit()
                print(f"  >> Best model saved to {self.save_path}")

    # ── TripleLRTrainer (whisper / projector / LoRA at different LRs) ──

    class TripleLRTrainer(Trainer):
        def __init__(self, *args, projector_lr=2e-4, whisper_lr=1e-5, **kwargs):
            self.projector_lr = projector_lr
            self.whisper_lr = whisper_lr
            super().__init__(*args, **kwargs)

        def create_optimizer(self):
            model = self.model

            whisper_params = []
            projector_params = []
            llm_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "whisper" in name:
                    whisper_params.append(param)
                    print(f"  Whisper param: {name} (lr={self.whisper_lr})")
                elif any(k in name for k in ("audio_conv", "audio_proj", "conv_norm", "function_classifier")):
                    projector_params.append(param)
                    print(f"  Projector param: {name} (lr={self.projector_lr})")
                else:
                    llm_params.append(param)

            whisper_n = sum(p.numel() for p in whisper_params)
            proj_n = sum(p.numel() for p in projector_params)
            llm_n = sum(p.numel() for p in llm_params)
            print(f"  Whisper params: {len(whisper_params)} tensors, {whisper_n:,} params (lr={self.whisper_lr})")
            print(f"  Projector params: {len(projector_params)} tensors, {proj_n:,} params (lr={self.projector_lr})")
            print(f"  LLM/LoRA params: {len(llm_params)} tensors, {llm_n:,} params (lr={self.args.learning_rate})")

            from bitsandbytes.optim import AdamW8bit
            self.optimizer = AdamW8bit(
                [
                    {"params": whisper_params, "lr": self.whisper_lr},
                    {"params": projector_params, "lr": self.projector_lr},
                    {"params": llm_params, "lr": self.args.learning_rate},
                ],
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer


    # ── MAIN ──
    print("=" * 60)
    print("FINE-TUNING Qwen3-0.6B ON STOP REAL SPEECH")
    print("=" * 60)
    print(f"LoRA LR: {lora_lr}, Projector LR: {projector_lr}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

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

    # Load Whisper encoder (frozen)
    whisper_model_name = "openai/whisper-small"
    print(f"Loading Whisper: {whisper_model_name}")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

    # Create hybrid model
    hybrid_model = HybridQwen(model, whisper_encoder, tokenizer)
    print(f"  Hidden size: {model.config.hidden_size}")

    # Warm-start from existing best model if available
    warmstart_path = "/models/best_model_qwen3_backup_72pct_2026-03-02"
    if os.path.exists(warmstart_path):
        print(f"  Warm-starting from {warmstart_path}")
        # Load projector + whisper weights
        proj_path = os.path.join(warmstart_path, "projector.pt")
        if os.path.exists(proj_path):
            ckpt = torch.load(proj_path, map_location="cuda", weights_only=True)
            hybrid_model.audio_conv1.load_state_dict(ckpt["audio_conv1"])
            hybrid_model.audio_conv2.load_state_dict(ckpt["audio_conv2"])
            hybrid_model.conv_norm.load_state_dict(ckpt["conv_norm"])
            hybrid_model.audio_proj.load_state_dict(ckpt["audio_proj"])
            # Classifier head may have different size — skip if mismatched
            try:
                hybrid_model.function_classifier.load_state_dict(ckpt["function_classifier"])
                print("    Classifier loaded")
            except RuntimeError:
                print("    Classifier size mismatch (new function set) — reinitialized")
            # Whisper top layers
            if "whisper_top_layers" in ckpt:
                for layer_idx_str, layer_state in ckpt["whisper_top_layers"].items():
                    hybrid_model.whisper.layers[int(layer_idx_str)].load_state_dict(layer_state)
            print("    Projector + Whisper weights loaded")
        # Load LoRA weights
        from peft import PeftModel
        lora_adapter = os.path.join(warmstart_path, "adapter_model.bin")
        lora_safetensors = os.path.join(warmstart_path, "adapter_model.safetensors")
        if os.path.exists(lora_adapter) or os.path.exists(lora_safetensors):
            # Load LoRA state dict directly into existing PEFT model
            import safetensors.torch
            if os.path.exists(lora_safetensors):
                lora_state = safetensors.torch.load_file(lora_safetensors)
            else:
                lora_state = torch.load(lora_adapter, map_location="cuda", weights_only=True)
            missing, unexpected = hybrid_model.llm.load_state_dict(lora_state, strict=False)
            print(f"    LoRA loaded ({len(lora_state)} tensors, {len(missing)} missing base params)")
        print("  Warm-start complete!")
    else:
        print(f"  Training from scratch (fresh LoRA + projector)")

    # Create datasets
    print("\nBuilding STOP dataset...")
    train_dataset = STOPTrainDataset(
        "/data", processor, tokenizer, max_samples=max_samples,
    )
    val_dataset = RealSpeechValDataset(
        "/data", processor, tokenizer,
        max_samples=min(500, max_samples) if max_samples else 500,
    )

    if len(train_dataset) == 0:
        raise ValueError("No training samples loaded!")

    def collate_fn(batch):
        return {
            "audio_values": torch.stack([x["audio"] for x in batch]),
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "attention_mask": torch.stack([x["input_ids"] for x in batch]).ne(tokenizer.pad_token_id).long(),
            "function_labels": torch.tensor([x["function_idx"] for x in batch], dtype=torch.long),
        }

    # Training arguments — always clear old checkpoints to prevent resume conflicts
    import shutil
    output_dir = "/models/checkpoints_qwen3_finetune"
    if max_samples:
        output_dir = "/models/checkpoints_qwen3_finetune_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared stale checkpoints: {output_dir}")
    total_steps = len(train_dataset) * num_epochs // batch_size
    print(f"\nEstimated total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        learning_rate=lora_lr,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
        # Validation
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=False,
    )
    # Force .bin format — safetensors chokes on shared embed/lm_head tensors
    training_args.save_safetensors = False

    best_save_path = "/models/best_model_qwen3_finetuned"
    best_callback = BestCheckpointCallback(best_save_path, model_vol)

    whisper_lr = 1e-5
    trainer = TripleLRTrainer(
        model=hybrid_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collate_fn,
        args=training_args,
        projector_lr=projector_lr,
        whisper_lr=whisper_lr,
        callbacks=[best_callback],
    )

    print(f"\n{'='*60}")
    print("STARTING FINE-TUNING")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Whisper LR: {whisper_lr}")
    print(f"  LoRA LR: {lora_lr}")
    print(f"  Projector LR: {projector_lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Effective batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Resume from checkpoint if available
    last_ckpt = None
    if os.path.exists(output_dir):
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            last_ckpt = os.path.join(output_dir, sorted(ckpts)[-1])
            print(f"Resuming from: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save final model
    print("\nSaving fine-tuned model...")
    save_path = "/models/final_model_qwen3_finetuned"
    hybrid_model.save_pretrained(save_path)
    model_vol.commit()

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"Final model saved to: {save_path}")
    print(f"Best model saved to: {best_save_path} (eval_loss={best_callback.best_eval_loss:.4f})")
    print(f"{'='*60}")

    return {"status": "success", "train_samples": len(train_dataset), "val_samples": len(val_dataset),
            "best_eval_loss": best_callback.best_eval_loss}


@app.local_entrypoint()
def main(
    epochs: int = 2,
    batch_size: int = 16,
    lora_lr: float = 5e-5,
    projector_lr: float = 2e-4,
    max_samples: int = None,
    test_run: bool = False,
):
    if test_run:
        max_samples = 100
        epochs = 1

    result = finetune.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lora_lr=lora_lr,
        projector_lr=projector_lr,
        max_samples=max_samples,
    )
    print(f"Result: {result}")
