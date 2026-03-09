"""
Speech-to-Tool-Call — Live demo on Modal.

Records voice from the browser, runs E2E inference on GPU, displays JSON tool call.

Usage:
    modal serve demo.py          # Dev mode (hot reload, temporary URL)
    modal deploy demo.py         # Production (persistent URL)
"""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "transformers", "accelerate", "peft", "bitsandbytes",
        "librosa", "soundfile", "numpy", "gradio>=5.0",
    )
)

model_vol = modal.Volume.from_name("hybrid-model-storage")
app = modal.App("runanywhere-demo")

MODEL_DIR = "/models/best_model_qwen3_finetuned"
DEFAULT_INSTRUCTION = "<|audio|>Convert to tool call JSON:\n"


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_vol},
    timeout=600,
    scaledown_window=300,
)
class Demo:
    @modal.enter()
    def load_model(self):
        """Load models during container startup (before web server check)."""
        import torch
        import torch.nn as nn
        import json
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            WhisperModel, WhisperForConditionalGeneration, WhisperProcessor,
        )
        from peft import LoraConfig, get_peft_model

        # ── HybridQwen ──
        class HybridQwen(nn.Module):
            def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                         function_names=None):
                super().__init__()
                self.whisper = whisper_encoder
                for param in self.whisper.parameters():
                    param.requires_grad = False

                self.target_dtype = base_model.get_input_embeddings().weight.dtype
                hidden_size = base_model.config.hidden_size
                whisper_dim = whisper_encoder.config.d_model

                self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.conv_norm = nn.LayerNorm(whisper_dim).to("cuda")
                self.audio_proj = nn.Sequential(
                    nn.Linear(whisper_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ).to("cuda")

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
                    if "whisper_top_layers" in state_dict:
                        for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                            self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
                    print("Projector + whisper layers loaded.")

            @torch.no_grad()
            def generate(self, audio_values, max_new_tokens=128):
                enc_out = self.whisper(audio_values).last_hidden_state
                x = enc_out.transpose(1, 2).float()
                x = torch.relu(self.audio_conv1(x))
                x = torch.relu(self.audio_conv2(x))
                x = x.transpose(1, 2)
                x = self.conv_norm(x)
                projected = self.audio_proj(x).to(self.target_dtype)

                bs = projected.shape[0]
                embed_tokens = self.llm.get_input_embeddings()
                instr_ids = self.instruction_ids.expand(bs, -1)
                instr_embeds = embed_tokens(instr_ids)
                current_embeds = torch.cat([projected, instr_embeds], dim=1)

                generated_ids = []
                for _ in range(max_new_tokens):
                    outputs = self.llm(inputs_embeds=current_embeds)
                    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
                    token_id = next_token_id.item()
                    generated_ids.append(token_id)

                    if token_id == self.tokenizer.eos_token_id:
                        break
                    full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if full_text.strip().endswith('}'):
                        if full_text.count('{') > 0 and full_text.count('{') == full_text.count('}'):
                            break
                    next_token_embed = embed_tokens(next_token_id.unsqueeze(0))
                    current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

                return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # ── Load model ──
        print("Loading model...")
        config_path = os.path.join(MODEL_DIR, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
        whisper_model_name = config.get("whisper_model", "openai/whisper-small")
        function_names = config.get("function_names", [])
        base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
        lora_r = config.get("lora_r", 32)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(base_llm, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_llm)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0, bias="none",
        )
        llm = get_peft_model(llm, lora_config)

        adapter_path = os.path.join(MODEL_DIR, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapter_state = torch.load(adapter_path, map_location="cpu")
            remapped = {}
            for k, v in adapter_state.items():
                nk = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
                nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                remapped[nk] = v
            llm.load_state_dict(remapped, strict=False)
            print(f"LoRA loaded ({len(remapped)} tensors)")

        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

        self.hybrid_model = HybridQwen(llm, whisper_encoder, tokenizer, instruction_prompt,
                                       function_names=function_names)
        self.hybrid_model.load_pretrained(MODEL_DIR)
        self.hybrid_model.eval()

        # Full Whisper for ASR transcript
        self.whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        self.whisper_asr.eval()
        print(f"Model ready! {len(function_names)} functions.")

    @modal.web_server(8000, startup_timeout=30)
    def serve(self):
        """Start Gradio after models are loaded (via @modal.enter)."""
        import torch
        import json
        import librosa
        import numpy as np
        import gradio as gr

        hybrid_model = self.hybrid_model
        whisper_asr = self.whisper_asr
        processor = self.processor

        def run_prediction(audio_input):
            if audio_input is None:
                return None, "", "Record or upload audio first."

            sr, audio_data = audio_input

            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float64:
                audio_float = audio_data.astype(np.float32)
            else:
                audio_float = audio_data
            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)

            if sr != 16000:
                audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=16000)

            inputs = processor(audio_float, sampling_rate=16000, return_tensors="pt")
            audio_values = inputs.input_features.to("cuda")

            prediction = hybrid_model.generate(audio_values)

            with torch.no_grad():
                asr_ids = whisper_asr.generate(audio_values, max_length=448)
            asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

            try:
                tool_call = json.loads(prediction)
            except json.JSONDecodeError:
                tool_call = {"raw_output": prediction, "parse_error": True}

            if isinstance(tool_call, dict) and "function" in tool_call:
                func = tool_call["function"]
                args = tool_call.get("arguments", {})
                summary = f"Function: {func}"
                if args:
                    summary += "\nArguments: " + ", ".join(f"{k}={v}" for k, v in args.items())
            else:
                summary = json.dumps(tool_call, indent=2)

            return tool_call, asr_transcript, summary

        with gr.Blocks(title="Speech-to-Tool-Call") as demo:
            gr.Markdown("""
# Speech-to-Tool-Call — Voice to Tool Calls

Speak a command into your microphone and get a structured JSON function call.

**Architecture**: Whisper-small + Qwen3-0.6B (LoRA) | **Size**: 0.6B params, 730MB | **Accuracy**: 77% EM on STOP

**Try saying**: "Set an alarm for 7 AM" | "What's the weather?" | "Play some jazz music" | "Navigate to the airport" | "Send a message to John saying I'll be late"
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="Record or upload audio",
                    )
                    submit_btn = gr.Button(
                        "Convert to Tool Call",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    json_output = gr.JSON(label="Tool Call Output")
                    transcript_output = gr.Textbox(
                        label="Whisper ASR Transcript (for reference)",
                        interactive=False,
                    )
                    summary_output = gr.Textbox(
                        label="Summary",
                        interactive=False,
                    )

            submit_btn.click(
                fn=run_prediction,
                inputs=[audio_input],
                outputs=[json_output, transcript_output, summary_output],
            )

            gr.Markdown("""
---
**Supported functions**: alarm_set, alarm_cancel, alarm_check, alarm_silence, alarm_snooze, alarm_update,
weather_get_current, navigation_start, navigation_traffic, nav_duration, nav_eta, nav_distance,
message_send, message_read, message_react, media_play_music, media_playback_control, music_playlist,
calendar_get_events, timer_set, timer_check, timer_modify, timer_pause, timer_resume, timer_cancel,
reminder_set, reminder_check, reminder_cancel, reminder_update
            """)

        demo.launch(server_name="0.0.0.0", server_port=8000)
