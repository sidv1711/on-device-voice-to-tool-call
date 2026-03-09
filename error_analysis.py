"""
Error analysis of the hybrid audio-to-tool-call model.

Categorizes each prediction error into: exact_match, invalid_json,
wrong_function, wrong_arg_keys, or wrong_arg_values.
Outputs per-sample JSONL + aggregate summary JSON.

Supports ASR keyword override: when the model predicts a known collapse
target function, runs Whisper ASR on the audio and uses keyword matching
to correct the prediction.

Usage:
    modal run error_analysis.py
    modal run error_analysis.py --max-samples 50 --slurp-only
    modal run error_analysis.py --model-dir final_model_qwen3_finetuned
    modal run error_analysis.py --asr-override  # enable ASR override
    modal run error_analysis.py --cascaded      # use cascaded pipeline (Whisper ASR -> text model)
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

app = modal.App("hybrid-error-analysis")

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
def run_analysis(
    max_samples: int = 0,
    model_dir: str = "final_model_qwen3_finetuned",
    slurp_only: bool = False,
    asr_override: bool = False,
    cascaded: bool = False,
    text_model_dir: str = "best_cascaded_text_model",
):
    """Run error analysis on the test set."""
    import warnings
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    from unsloth import FastLanguageModel  # must be before transformers
    import torch
    import torch.nn as nn
    import json
    import librosa
    import time
    import sys
    from collections import Counter, defaultdict
    from transformers import (
        WhisperModel, WhisperForConditionalGeneration, WhisperProcessor,
    )

    # Import ASR override module (uploaded alongside this script)
    sys.path.insert(0, "/root")
    from asr_override import OVERRIDE_RULES, COLLAPSE_TARGETS, asr_override as apply_asr_override

    # ── HybridQwen (same as evaluate.py) ──

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

    # ── Error categorization ──

    def categorize_error(expected, prediction_text):
        """Classify a prediction into an error category.

        Returns (error_type, predicted_json_or_None, details_dict).
        """
        expected_str = json.dumps(expected)

        # 1. Exact match
        if prediction_text.strip() == expected_str.strip():
            return "exact_match", json.loads(prediction_text), {}

        # 2. Invalid JSON
        try:
            predicted = json.loads(prediction_text)
        except (json.JSONDecodeError, TypeError):
            return "invalid_json", None, {"raw": prediction_text}

        if not isinstance(predicted, dict):
            return "invalid_json", predicted, {"reason": "not a dict"}

        # 3. Wrong function
        pred_func = predicted.get("function", "")
        exp_func = expected.get("function", "")
        if pred_func != exp_func:
            return "wrong_function", predicted, {
                "expected_function": exp_func,
                "predicted_function": pred_func,
            }

        # 4. Wrong argument keys
        pred_args = predicted.get("arguments", {})
        exp_args = expected.get("arguments", {})
        if not isinstance(pred_args, dict):
            pred_args = {}
        if not isinstance(exp_args, dict):
            exp_args = {}

        pred_keys = set(pred_args.keys())
        exp_keys = set(exp_args.keys())
        if pred_keys != exp_keys:
            return "wrong_arg_keys", predicted, {
                "missing_keys": sorted(exp_keys - pred_keys),
                "extra_keys": sorted(pred_keys - exp_keys),
            }

        # 5. Wrong argument values (same function, same keys, different values)
        mismatched = {}
        for key in exp_keys:
            if str(pred_args[key]) != str(exp_args[key]):
                mismatched[key] = {
                    "expected": exp_args[key],
                    "got": pred_args[key],
                }
        if mismatched:
            return "wrong_arg_values", predicted, {"mismatched_values": mismatched}

        # Edge case: JSON equivalent but string != (e.g. whitespace differences)
        return "exact_match", predicted, {"note": "json_equivalent"}

    # ── Load model ──
    print("=" * 60)
    print("ERROR ANALYSIS" + (" (CASCADED)" if cascaded else ""))
    print(f"Model: {text_model_dir if cascaded else model_dir}")
    print(f"SLURP only: {slurp_only}")
    print(f"Max samples: {max_samples if max_samples > 0 else 'all'}")
    print("=" * 60)

    whisper_model_name = "openai/whisper-small"  # default, may be overridden
    hybrid_model = None
    cascaded_model = None
    cascaded_tokenizer = None
    cascaded_instruction = "Convert to tool call JSON:\n"
    whisper_asr = None
    processor = None

    if cascaded:
        # ── Cascaded pipeline: Whisper ASR + text-only Qwen3 LoRA ──
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model

        text_model_path = f"/models/{text_model_dir}"
        if not os.path.exists(text_model_path):
            print(f"Text model not found at {text_model_path}. Available:")
            for item in os.listdir("/models"):
                print(f"  /models/{item}")
            raise FileNotFoundError(f"No model at {text_model_path}")

        # Load cascaded config
        cascaded_config_path = os.path.join(text_model_path, "cascaded_config.json")
        if os.path.exists(cascaded_config_path):
            with open(cascaded_config_path) as f:
                cascaded_config = json.load(f)
        else:
            cascaded_config = {}
        base_llm = cascaded_config.get("base_llm", "unsloth/Qwen3-0.6B")
        lora_r = cascaded_config.get("lora_r", 32)
        cascaded_instruction = cascaded_config.get("instruction_prefix", cascaded_instruction)
        print(f"Cascaded text model: {text_model_path}")
        print(f"Base LLM: {base_llm}, LoRA rank: {lora_r}")

        # Load base model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        cascaded_model = AutoModelForCausalLM.from_pretrained(
            base_llm, quantization_config=bnb_config, device_map="auto",
        )
        cascaded_tokenizer = AutoTokenizer.from_pretrained(base_llm)
        if cascaded_tokenizer.pad_token is None:
            cascaded_tokenizer.pad_token = cascaded_tokenizer.eos_token

        # Apply LoRA and load weights
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0, bias="none",
        )
        cascaded_model = get_peft_model(cascaded_model, lora_config)

        adapter_bin_path = os.path.join(text_model_path, "adapter_model.bin")
        if os.path.exists(adapter_bin_path):
            adapter_state = torch.load(adapter_bin_path, map_location="cpu")
            remapped_state = {}
            for k, v in adapter_state.items():
                new_key = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
                new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                remapped_state[new_key] = v
            missing, unexpected = cascaded_model.load_state_dict(remapped_state, strict=False)
            print(f"  LoRA loaded: {len(remapped_state)} tensors, {len(missing)} missing (base)")
        cascaded_model.eval()

        # Load full Whisper for ASR
        print(f"Loading Whisper ASR: {whisper_model_name}")
        processor = WhisperProcessor.from_pretrained(whisper_model_name)
        whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        whisper_asr.eval()
        print("Cascaded pipeline ready.")

    else:
        # ── E2E pipeline: HybridQwen (audio -> JSON) ──
        model_path = f"/models/{model_dir}"
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
        whisper_model_name = config.get("whisper_model", "openai/whisper-base")
        function_names = config.get("function_names", [])
        print(f"Model config: {json.dumps(config, indent=2)}")
        print(f"Whisper: {whisper_model_name}")
        print(f"Function names: {len(function_names)} functions")

        base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
        lora_r = config.get("lora_r", 32)
        print(f"Base LLM: {base_llm}")
        print(f"LoRA rank: {lora_r}")

        # Load with Unsloth — point directly at saved adapter dir
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_path, load_in_4bit=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        processor = WhisperProcessor.from_pretrained(whisper_model_name)
        whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

        hybrid_model = HybridQwen(base_model, whisper_encoder, tokenizer, instruction_prompt,
                                  whisper_model_name=whisper_model_name, function_names=function_names)
        hybrid_model.load_pretrained(model_path)
        hybrid_model.eval()
        print("Model ready.")

        # Load full Whisper for ASR override
        if asr_override:
            print("Loading full Whisper model for ASR override...")
            whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
            whisper_asr.eval()
            print(f"ASR override enabled. Collapse targets: {sorted(COLLAPSE_TARGETS)}")
    print()

    # ── Load test data ──
    metadata_path = None
    for dir_name in ["merged_stop", "merged_new", "merged", "stop"]:
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

            if slurp_only and entry.get("source") != "slurp":
                continue

            raw_path = entry["audio_path"]
            if entry.get("source") == "slurp":
                filename = os.path.basename(raw_path)
                audio_path = f"/data/slurp_audio/{filename}"
            elif entry.get("source") == "stop":
                audio_path = f"/data/{raw_path}"
            else:
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
    print(f"Analyzing {total} test samples\n")

    # ── Run analysis ──
    results = []
    error_counts = Counter()
    error_by_source = defaultdict(Counter)
    error_by_domain = defaultdict(Counter)
    function_confusion = Counter()  # (expected, predicted) pairs
    missing_keys_counter = Counter()
    extra_keys_counter = Counter()
    per_function_stats = defaultdict(lambda: {"total": 0, "exact": 0, "func_correct": 0})
    audio_errors = 0

    start_time = time.time()

    # ── Cascaded text generation helper ──
    @torch.no_grad()
    def cascaded_generate(transcript, max_new_tokens=128):
        """Generate JSON from transcript using text-only model with brace-counting early stop."""
        prompt = cascaded_instruction + transcript
        inputs = cascaded_tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids

        generated_ids = []
        for step in range(max_new_tokens):
            outputs = cascaded_model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1)

            token_id = next_token_id.item()
            generated_ids.append(token_id)

            if token_id == cascaded_tokenizer.eos_token_id:
                break

            full_text = cascaded_tokenizer.decode(generated_ids, skip_special_tokens=True)
            if full_text.strip().endswith('}'):
                open_b = full_text.count('{')
                close_b = full_text.count('}')
                if open_b > 0 and open_b == close_b:
                    break

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        return cascaded_tokenizer.decode(generated_ids, skip_special_tokens=True)

    output_dir = "/models/eval_results"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "error_analysis_cascaded.jsonl" if cascaded else "error_analysis.jsonl")

    with open(jsonl_path, "w") as jsonl_file:
        for i, sample in enumerate(samples):
            domain = sample["domain"]
            source = sample["source"]
            expected = sample["expected"]
            exp_func = expected.get("function", "unknown") if isinstance(expected, dict) else "unknown"
            per_function_stats[exp_func]["total"] += 1

            # Load audio
            try:
                wav, _ = librosa.load(sample["audio_path"], sr=16000)
            except Exception as e:
                audio_errors += 1
                record = {
                    "idx": i,
                    "source": source,
                    "domain": domain,
                    "transcript": sample["transcript"],
                    "expected": expected,
                    "predicted_raw": None,
                    "predicted_json": None,
                    "error_type": "audio_error",
                    "details": {"error": str(e)},
                }
                jsonl_file.write(json.dumps(record) + "\n")
                error_counts["audio_error"] += 1
                error_by_source[source]["audio_error"] += 1
                error_by_domain[domain]["audio_error"] += 1
                if i < 10 or i % 500 == 0:
                    print(f"  [{i+1}] AUDIO ERROR: {sample['audio_path']}: {e}")
                continue

            # Generate prediction
            inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
            audio_values = inputs.input_features.to("cuda")

            asr_transcript = None
            override_info = None

            if cascaded:
                # Cascaded: Whisper ASR -> text model -> JSON
                with torch.no_grad():
                    asr_ids = whisper_asr.generate(audio_values, max_length=448)
                asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]
                # Uppercase to match STOP training data convention
                prediction_text = cascaded_generate(asr_transcript.upper())
            else:
                # E2E: HybridQwen (audio -> JSON)
                prediction_text = hybrid_model.generate(audio_values)

                # ASR override: if prediction is a collapse target, run Whisper ASR
                if asr_override and whisper_asr is not None:
                    try:
                        pred_parsed = json.loads(prediction_text)
                        pred_func = pred_parsed.get("function", "") if isinstance(pred_parsed, dict) else ""
                    except (json.JSONDecodeError, TypeError):
                        pred_func = ""

                    if pred_func in COLLAPSE_TARGETS:
                        with torch.no_grad():
                            asr_ids = whisper_asr.generate(audio_values, max_length=448)
                        asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

                        corrected_json, override_info = apply_asr_override(
                            pred_func, asr_transcript, pred_parsed
                        )
                        if corrected_json is not None:
                            prediction_text = json.dumps(corrected_json)

            # Categorize
            error_type, predicted_json, details = categorize_error(expected, prediction_text)

            # Track stats
            error_counts[error_type] += 1
            error_by_source[source][error_type] += 1
            error_by_domain[domain][error_type] += 1

            if error_type == "exact_match":
                per_function_stats[exp_func]["exact"] += 1
                per_function_stats[exp_func]["func_correct"] += 1
            elif error_type in ("wrong_arg_keys", "wrong_arg_values"):
                per_function_stats[exp_func]["func_correct"] += 1
            elif error_type == "wrong_function":
                pred_func = details.get("predicted_function", "unknown")
                function_confusion[(exp_func, pred_func)] += 1

            if error_type == "wrong_arg_keys":
                for k in details.get("missing_keys", []):
                    missing_keys_counter[k] += 1
                for k in details.get("extra_keys", []):
                    extra_keys_counter[k] += 1

            # Write per-sample record
            record = {
                "idx": i,
                "source": source,
                "domain": domain,
                "transcript": sample["transcript"],
                "expected": expected,
                "predicted_raw": prediction_text,
                "predicted_json": predicted_json,
                "error_type": error_type,
                "details": details,
            }
            if asr_transcript is not None:
                record["asr_transcript"] = asr_transcript
            if override_info is not None:
                record["asr_override"] = override_info
            jsonl_file.write(json.dumps(record) + "\n")

            # Progress
            processed = i + 1 - audio_errors
            if processed > 0 and (i < 5 or (i + 1) % 100 == 0 or i + 1 == total):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                exact = error_counts["exact_match"]
                print(
                    f"  [{i+1}/{total}] "
                    f"exact={exact}/{processed} ({100*exact/processed:.1f}%) "
                    f"| {dict(error_counts)} "
                    f"| {rate:.1f} samples/s, ETA {eta/60:.1f}min"
                )
                if i < 5:
                    print(f"    transcript: {sample['transcript']}")
                    if asr_transcript:
                        print(f"    asr_transcript: {asr_transcript}")
                    print(f"    expected:   {json.dumps(expected)}")
                    print(f"    predicted:  {prediction_text}")
                    print(f"    error_type: {error_type}")

            # Checkpoint every 500 samples
            if (i + 1) % 500 == 0:
                jsonl_file.flush()
                model_vol.commit()
                print(f"  >> Checkpoint saved at sample {i+1}")

    # ── Build aggregate summary ──
    elapsed = time.time() - start_time
    evaluated = total - audio_errors

    # Error type distribution
    error_distribution = {
        "overall": dict(error_counts),
        "by_source": {src: dict(cnts) for src, cnts in sorted(error_by_source.items())},
        "by_domain": {dom: dict(cnts) for dom, cnts in sorted(error_by_domain.items())},
    }

    # Function confusion matrix (top 30)
    confusion_list = [
        {"expected": exp, "predicted": pred, "count": count}
        for (exp, pred), count in function_confusion.most_common(30)
    ]

    # Most common missing/extra keys
    common_missing = [{"key": k, "count": c} for k, c in missing_keys_counter.most_common(20)]
    common_extra = [{"key": k, "count": c} for k, c in extra_keys_counter.most_common(20)]

    # Per-function accuracy
    function_accuracy = {}
    for func, stats in sorted(per_function_stats.items()):
        t = stats["total"]
        function_accuracy[func] = {
            "total": t,
            "exact_match": stats["exact"],
            "exact_match_pct": round(100 * stats["exact"] / t, 1) if t > 0 else 0,
            "func_correct": stats["func_correct"],
            "func_correct_pct": round(100 * stats["func_correct"] / t, 1) if t > 0 else 0,
        }

    summary_path = os.path.join(output_dir, "error_summary_cascaded.json" if cascaded else "error_summary.json")

    summary = {
        "model_dir": text_model_dir if cascaded else model_dir,
        "cascaded": cascaded,
        "slurp_only": slurp_only,
        "total_samples": total,
        "audio_errors": audio_errors,
        "evaluated": evaluated,
        "elapsed_minutes": round(elapsed / 60, 1),
        "error_distribution": error_distribution,
        "function_confusion_top30": confusion_list,
        "common_missing_keys": common_missing,
        "common_extra_keys": common_extra,
        "per_function_accuracy": function_accuracy,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    model_vol.commit()

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total: {total} | Evaluated: {evaluated} | Audio errors: {audio_errors}")
    print(f"Time: {elapsed/60:.1f} min ({evaluated/elapsed:.1f} samples/s)" if elapsed > 0 else "")

    print("\n--- Error Type Distribution ---")
    for etype in ["exact_match", "invalid_json", "wrong_function", "wrong_arg_keys", "wrong_arg_values", "audio_error"]:
        count = error_counts.get(etype, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"  {etype:20s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- By Source ---")
    for src, cnts in sorted(error_by_source.items()):
        src_total = sum(cnts.values())
        parts = [f"{k}={v}" for k, v in sorted(cnts.items())]
        print(f"  {src}: total={src_total}  {', '.join(parts)}")

    print("\n--- Top 10 Function Confusions ---")
    for item in confusion_list[:10]:
        print(f"  {item['expected']:30s} -> {item['predicted']:30s} ({item['count']}x)")

    print("\n--- Top 10 Missing Keys ---")
    for item in common_missing[:10]:
        print(f"  {item['key']:30s} ({item['count']}x)")

    print("\n--- Per-Function Accuracy (sorted by total) ---")
    for func, stats in sorted(function_accuracy.items(), key=lambda x: -x[1]["total"]):
        t = stats["total"]
        print(f"  {func:35s}: exact={stats['exact_match']:4d}/{t:4d} ({stats['exact_match_pct']:5.1f}%)  "
              f"func={stats['func_correct']:4d}/{t:4d} ({stats['func_correct_pct']:5.1f}%)")

    print("=" * 60)
    print(f"Per-sample results: {jsonl_path}")
    print(f"Summary: {summary_path}")

    return summary


@app.local_entrypoint()
def main(
    max_samples: int = 0,
    model_dir: str = "final_model_qwen3_finetuned",
    slurp_only: bool = False,
    asr_override: bool = False,
    cascaded: bool = False,
    text_model_dir: str = "best_cascaded_text_model",
):
    summary = run_analysis.remote(
        max_samples=max_samples,
        model_dir=model_dir,
        slurp_only=slurp_only,
        asr_override=asr_override,
        cascaded=cascaded,
        text_model_dir=text_model_dir,
    )
    dist = summary["error_distribution"]["overall"]
    evaluated = summary["evaluated"]
    exact = dist.get("exact_match", 0)
    print(f"\nFinal: {exact}/{evaluated} exact match "
          f"({100*exact/evaluated:.1f}%)" if evaluated > 0 else "")
