"""
Full evaluation of the hybrid audio-to-tool-call model on the test set.

Supports ASR keyword override to correct known collapse predictions.

Usage:
    modal run evaluate.py
    modal run evaluate.py --max-samples 100  # quick test
    modal run evaluate.py --asr-override     # enable ASR override
"""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch", "torchvision", "transformers", "accelerate", "peft", "bitsandbytes",
        "librosa", "soundfile", "numpy",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
    .add_local_file("asr_override.py", "/root/asr_override.py")
)

data_vol = modal.Volume.from_name("training-data-vol")
model_vol = modal.Volume.from_name("hybrid-model-storage")

app = modal.App("hybrid-eval")

DEFAULT_INSTRUCTION = "<|audio|>Convert to tool call JSON:\n"


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
    timeout=86400,  # 24 hours
)
def run_eval(max_samples: int = 0, model_dir: str = "final_model_v3b", asr_override: bool = False):
    """Evaluate model on the full merged test set."""
    import torch
    import torch.nn as nn
    import json
    import librosa
    import time
    import sys
    from transformers import WhisperModel, WhisperForConditionalGeneration, WhisperProcessor
    from unsloth import FastLanguageModel

    sys.path.insert(0, "/root")
    from asr_override import COLLAPSE_TARGETS, asr_override as apply_asr_override

    class HybridQwen(nn.Module):
        def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                     whisper_model_name="openai/whisper-base", function_names=None):
            super().__init__()
            self.whisper = whisper_encoder
            for param in self.whisper.parameters():
                param.requires_grad = False

            self.target_dtype = base_model.get_input_embeddings().weight.dtype
            hidden_size = base_model.config.hidden_size

            whisper_dim = whisper_encoder.config.d_model  # 384=tiny, 512=base, 768=small
            self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
            self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
            self.conv_norm = nn.LayerNorm(whisper_dim).to("cuda")
            self.audio_proj = nn.Sequential(
                nn.Linear(whisper_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ).to("cuda")

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
                self.instruction, add_special_tokens=False, return_tensors="pt"
            ).to("cuda")

        def load_pretrained(self, checkpoint_dir):
            projector_path = os.path.join(checkpoint_dir, "projector.pt")
            if os.path.exists(projector_path):
                state_dict = torch.load(projector_path, map_location="cuda")
                self.audio_conv1.load_state_dict(state_dict["audio_conv1"])
                self.audio_conv2.load_state_dict(state_dict["audio_conv2"])
                self.conv_norm.load_state_dict(state_dict["conv_norm"])
                self.audio_proj.load_state_dict(state_dict["audio_proj"])
                if "function_classifier" in state_dict and self.function_names:
                    self.function_classifier.load_state_dict(state_dict["function_classifier"])
                    print("Projector + classifier loaded.")
                else:
                    print("Projector loaded.")
                if "whisper_top_layers" in state_dict:
                    for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                        self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
                    print(f"Whisper top {len(state_dict['whisper_top_layers'])} layers loaded.")

        def encode_audio(self, audio_values):
            with torch.no_grad():
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
            pooled = audio_embeds.mean(dim=1).float()
            return self.function_classifier(pooled)

        @torch.no_grad()
        def generate(self, audio_values, max_new_tokens=128):
            audio_embeds = self.encode_audio(audio_values)
            bs = audio_embeds.shape[0]
            embed_tokens = self.llm.get_input_embeddings()

            instr_ids = self.instruction_ids.expand(bs, -1)
            instr_embeds = embed_tokens(instr_ids)

            # Free generation (classifier disabled — undertrained after 3 epochs)
            forced_prefix = ""
            current_embeds = torch.cat([audio_embeds, instr_embeds], dim=1)

            generated_ids = []
            for step in range(max_new_tokens):
                outputs = self.llm(inputs_embeds=current_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = next_token_logits.argmax(dim=-1)

                token_id = next_token_id.item()
                generated_ids.append(token_id)

                if token_id == self.tokenizer.eos_token_id:
                    break

                full_text = forced_prefix + self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if full_text.strip().endswith('}'):
                    open_b = full_text.count('{')
                    close_b = full_text.count('}')
                    if open_b > 0 and open_b == close_b:
                        break

                next_token_embed = embed_tokens(next_token_id.unsqueeze(0))
                current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return forced_prefix + generated_text

    # ── Load model ──
    print("=" * 60)
    print("EVALUATION: Full Test Set")
    print("=" * 60)

    model_path = f"/models/{model_dir}"
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
    whisper_model_name = config.get("whisper_model", "openai/whisper-base")
    function_names = config.get("function_names", [])
    print(f"Whisper: {whisper_model_name}")
    print(f"Function names: {len(function_names)} functions")

    base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
    lora_r = config.get("lora_r", 32)
    print(f"Base LLM: {base_llm}")
    print(f"LoRA rank: {lora_r}")

    # Load with Unsloth — point directly at saved adapter dir
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path, load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

    hybrid_model = HybridQwen(model, whisper_encoder, tokenizer, instruction_prompt,
                              whisper_model_name=whisper_model_name, function_names=function_names)
    hybrid_model.load_pretrained(model_path)
    hybrid_model.eval()
    print("Model ready.")

    # ── Load full Whisper for ASR override ──
    whisper_asr = None
    if asr_override:
        print("Loading full Whisper model for ASR override...")
        whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        whisper_asr.eval()
        print(f"ASR override enabled. Collapse targets: {sorted(COLLAPSE_TARGETS)}")
    print()

    # ── Load test data ──
    metadata_path = None
    for dir_name in ["stop", "merged_stop", "merged_new", "merged"]:
        candidate = f"/data/{dir_name}/test/metadata.jsonl"
        if os.path.exists(candidate):
            metadata_path = candidate
            break
    if not metadata_path:
        raise FileNotFoundError("No test metadata found")
    print(f"Using metadata: {metadata_path}")
    samples = []
    skipped_missing = 0
    with open(metadata_path) as f:
        for line in f:
            entry = json.loads(line)
            raw_path = entry["audio_path"]

            # Resolve audio path based on source
            if entry.get("source") == "slurp":
                # SLURP audio is flat in /data/slurp_audio/
                filename = os.path.basename(raw_path)
                audio_path = f"/data/slurp_audio/{filename}"
            elif entry.get("source") == "stop":
                # STOP audio: audio_path = "stop_audio/{split}/{filename}"
                audio_path = f"/data/{raw_path}"
            else:
                # Synthetic: "data/synthetic_full/..." -> "/data/synthetic_full/..."
                audio_path = "/" + raw_path

            if not os.path.exists(audio_path):
                skipped_missing += 1
                continue

            samples.append({
                "audio_path": audio_path,
                "expected": entry["tool_call"],
                "transcript": entry.get("transcript", ""),
                "domain": entry.get("domain", "unknown"),
                "source": entry.get("source", "unknown"),
            })

    if skipped_missing:
        print(f"Skipped {skipped_missing} samples with missing audio files")

    if max_samples > 0:
        samples = samples[:max_samples]

    total = len(samples)
    print(f"Evaluating on {total} test samples\n")

    # ── Run evaluation ──
    exact_match = 0
    function_correct = 0
    valid_json = 0
    errors = 0
    domain_stats = {}  # domain -> {total, exact, func_correct}
    source_stats = {}  # synthetic vs slurp

    start_time = time.time()

    for i, sample in enumerate(samples):
        domain = sample["domain"]
        source = sample["source"]

        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "exact": 0, "func": 0, "json_valid": 0}
        if source not in source_stats:
            source_stats[source] = {"total": 0, "exact": 0, "func": 0, "json_valid": 0}

        domain_stats[domain]["total"] += 1
        source_stats[source]["total"] += 1

        try:
            wav, _ = librosa.load(sample["audio_path"], sr=16000)
        except Exception as e:
            errors += 1
            if i < 10 or i % 500 == 0:
                print(f"  [{i+1}] AUDIO ERROR: {sample['audio_path']}: {e}")
            continue

        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        audio_values = inputs.input_features.to("cuda")

        prediction_text = hybrid_model.generate(audio_values)

        # ASR override
        if asr_override and whisper_asr is not None:
            try:
                pred_parsed = json.loads(prediction_text)
                pred_func = pred_parsed.get("function", "") if isinstance(pred_parsed, dict) else ""
            except (json.JSONDecodeError, TypeError):
                pred_func = ""

            if pred_func in COLLAPSE_TARGETS:
                with torch.no_grad():
                    asr_ids = whisper_asr.generate(audio_values, max_new_tokens=128)
                asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

                corrected_json, _ = apply_asr_override(pred_func, asr_transcript, pred_parsed)
                if corrected_json is not None:
                    prediction_text = json.dumps(corrected_json)

        expected = sample["expected"]
        expected_str = json.dumps(expected)

        # Check valid JSON
        try:
            predicted = json.loads(prediction_text)
            valid_json += 1
            domain_stats[domain]["json_valid"] += 1
            source_stats[source]["json_valid"] += 1
        except json.JSONDecodeError:
            predicted = None

        # Exact match
        if prediction_text.strip() == expected_str.strip():
            exact_match += 1
            domain_stats[domain]["exact"] += 1
            source_stats[source]["exact"] += 1

        # Function name match
        if predicted and isinstance(predicted, dict):
            pred_func = predicted.get("function", "")
            exp_func = expected.get("function", "")
            if pred_func == exp_func:
                function_correct += 1
                domain_stats[domain]["func"] += 1
                source_stats[source]["func"] += 1

        # Progress logging
        processed = i + 1 - errors
        if processed > 0 and (i < 5 or (i + 1) % 100 == 0 or i + 1 == total):
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] exact={exact_match}/{processed} ({100*exact_match/processed:.1f}%) "
                  f"func={function_correct}/{processed} ({100*function_correct/processed:.1f}%) "
                  f"json_valid={valid_json}/{processed} ({100*valid_json/processed:.1f}%) "
                  f"errors={errors} | {rate:.1f} samples/s, ETA {eta/60:.1f}min")

            # Show sample predictions for first few
            if i < 5:
                print(f"    transcript: {sample['transcript']}")
                print(f"    expected:   {expected_str}")
                print(f"    predicted:  {prediction_text}")

        # Save checkpoint to volume every 500 samples
        if (i + 1) % 500 == 0 or i + 1 == total:
            checkpoint = {
                "samples_processed": i + 1,
                "evaluated": i + 1 - errors,
                "errors": errors,
                "exact_match": exact_match,
                "function_correct": function_correct,
                "valid_json": valid_json,
                "domain_stats": domain_stats,
                "source_stats": source_stats,
                "elapsed_seconds": time.time() - start_time,
            }
            os.makedirs("/models/eval_results", exist_ok=True)
            with open("/models/eval_results/checkpoint.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
            model_vol.commit()
            print(f"  >> Checkpoint saved to volume at sample {i+1}")

    # ── Final results ──
    elapsed = time.time() - start_time
    evaluated = total - errors

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Audio errors (skipped): {errors}")
    print(f"Evaluated: {evaluated}")
    print(f"Time: {elapsed/60:.1f} minutes ({evaluated/elapsed:.1f} samples/s)")
    print()
    print(f"  Exact match:     {exact_match}/{evaluated} = {100*exact_match/evaluated:.1f}%")
    print(f"  Function correct: {function_correct}/{evaluated} = {100*function_correct/evaluated:.1f}%")
    print(f"  Valid JSON:       {valid_json}/{evaluated} = {100*valid_json/evaluated:.1f}%")

    print("\n--- By Source ---")
    for source, stats in sorted(source_stats.items()):
        t = stats["total"]
        if t == 0:
            continue
        print(f"  {source}: exact={stats['exact']}/{t} ({100*stats['exact']/t:.1f}%)  "
              f"func={stats['func']}/{t} ({100*stats['func']/t:.1f}%)  "
              f"json={stats['json_valid']}/{t} ({100*stats['json_valid']/t:.1f}%)")

    print("\n--- By Domain ---")
    for domain, stats in sorted(domain_stats.items(), key=lambda x: -x[1]["total"]):
        t = stats["total"]
        if t == 0:
            continue
        print(f"  {domain:20s}: exact={stats['exact']:4d}/{t:4d} ({100*stats['exact']/t:5.1f}%)  "
              f"func={stats['func']:4d}/{t:4d} ({100*stats['func']/t:5.1f}%)")

    print("=" * 60)

    # Save final results to volume
    final_results = {
        "total": total,
        "evaluated": evaluated,
        "errors": errors,
        "exact_match": exact_match,
        "function_correct": function_correct,
        "valid_json": valid_json,
        "exact_match_pct": round(100 * exact_match / evaluated, 2) if evaluated > 0 else 0,
        "function_correct_pct": round(100 * function_correct / evaluated, 2) if evaluated > 0 else 0,
        "domain_stats": domain_stats,
        "source_stats": source_stats,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    os.makedirs("/models/eval_results", exist_ok=True)
    with open("/models/eval_results/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    model_vol.commit()
    print("Final results saved to volume: /eval_results/final_results.json")

    return {
        "total": total,
        "evaluated": evaluated,
        "errors": errors,
        "exact_match": exact_match,
        "function_correct": function_correct,
        "valid_json": valid_json,
        "exact_match_pct": round(100 * exact_match / evaluated, 2) if evaluated > 0 else 0,
        "function_correct_pct": round(100 * function_correct / evaluated, 2) if evaluated > 0 else 0,
        "domain_stats": domain_stats,
        "source_stats": source_stats,
    }


@app.local_entrypoint()
def main(max_samples: int = 0, model_dir: str = "final_model_v3b", asr_override: bool = False):
    results = run_eval.remote(max_samples=max_samples, model_dir=model_dir, asr_override=asr_override)
    print(f"\nFinal: {results['exact_match']}/{results['evaluated']} exact match ({results['exact_match_pct']}%)")
