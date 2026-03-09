"""
Speech-to-Tool-Call — Gradio demo for HuggingFace Spaces.

Hybrid Whisper-small + Qwen3-0.6B (LoRA) model that converts speech to JSON tool calls
across 14 domains (68 functions).

Usage:
    python app.py                    # local
    Deployed on HF Spaces with ZeroGPU automatically
"""

import os
import json
import spaces
import gradio as gr
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    WhisperModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model

from asr_override import COLLAPSE_TARGETS, asr_override as apply_asr_override

# ── Constants ──
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")
DEFAULT_INSTRUCTION = "<|audio|>Convert to tool call JSON:\n"

# ── Global state (lazy-loaded inside @spaces.GPU) ──
hybrid_model = None
whisper_asr = None
processor = None


# ── Model definition (v3 only) ──

class HybridQwen(nn.Module):
    """Whisper encoder → Conv1d compression → MLP projector → Qwen3 LLM."""

    def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                 whisper_model_name="openai/whisper-small", function_names=None):
        super().__init__()
        self.whisper = whisper_encoder
        for param in self.whisper.parameters():
            param.requires_grad = False

        self.target_dtype = base_model.get_input_embeddings().weight.dtype
        hidden_size = base_model.config.hidden_size
        whisper_dim = whisper_encoder.config.d_model

        # 2x Conv1d → 4x temporal compression
        self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1)
        self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1)
        self.conv_norm = nn.LayerNorm(whisper_dim)
        # 2-layer MLP projector
        self.audio_proj = nn.Sequential(
            nn.Linear(whisper_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.function_names = function_names or []
        num_funcs = len(self.function_names) if self.function_names else 1
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
        """Load projector weights from checkpoint."""
        projector_path = os.path.join(checkpoint_dir, "projector.pt")
        if not os.path.exists(projector_path):
            print(f"WARNING: No projector at {projector_path}")
            return
        state_dict = torch.load(projector_path, map_location="cpu")
        self.audio_conv1.load_state_dict(state_dict["audio_conv1"])
        self.audio_conv2.load_state_dict(state_dict["audio_conv2"])
        self.conv_norm.load_state_dict(state_dict["conv_norm"])
        self.audio_proj.load_state_dict(state_dict["audio_proj"])
        if "function_classifier" in state_dict and self.function_names:
            self.function_classifier.load_state_dict(state_dict["function_classifier"])
        if "whisper_top_layers" in state_dict:
            for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
        print("Projector weights loaded.")

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

    @torch.no_grad()
    def generate(self, audio_values, max_new_tokens=128):
        audio_embeds = self.encode_audio(audio_values)
        bs = audio_embeds.shape[0]
        embed_tokens = self.llm.get_input_embeddings()

        instr_ids = self.instruction_ids.to(audio_values.device).expand(bs, -1)
        instr_embeds = embed_tokens(instr_ids)
        current_embeds = torch.cat([audio_embeds, instr_embeds], dim=1)

        generated_ids = []
        for _ in range(max_new_tokens):
            outputs = self.llm(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1)
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


# ── Model loading ──

def load_models():
    """Load all models from MODEL_DIR. Called once inside @spaces.GPU."""
    global hybrid_model, whisper_asr, processor

    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
    whisper_model_name = config.get("whisper_model", "openai/whisper-small")
    function_names = config.get("function_names", [])
    base_llm = config.get("base_llm", "unsloth/Qwen2.5-0.5B-Instruct")
    lora_r = config.get("lora_r", 16)

    print(f"Loading base LLM: {base_llm} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
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
    model = get_peft_model(model, lora_config)

    adapter_path = os.path.join(MODEL_DIR, "adapter_model.bin")
    if os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        remapped = {}
        for k, v in adapter_state.items():
            nk = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
            nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
            nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped[nk] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"LoRA loaded: {len(remapped)} keys, {len(missing)} missing (expected), {len(unexpected)} unexpected")

    # Whisper
    print(f"Loading Whisper: {whisper_model_name}")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

    # Assemble hybrid model
    hybrid_model = HybridQwen(
        model, whisper_encoder, tokenizer, instruction_prompt,
        whisper_model_name=whisper_model_name, function_names=function_names,
    )
    hybrid_model.load_pretrained(MODEL_DIR)
    hybrid_model.eval()

    # Full Whisper for ASR override
    whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
    whisper_asr.eval()
    print("All models loaded.")


# ── Inference ──

@spaces.GPU
def predict(audio_path, enable_asr_override):
    """Run inference on an audio file. Returns (tool_call_json, asr_transcript, raw_output)."""
    global hybrid_model, whisper_asr, processor

    if audio_path is None:
        return None, "", "No audio provided."

    # Lazy-load models on first call
    if hybrid_model is None:
        load_models()

    # Load audio
    wav, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    audio_values = inputs.input_features.to("cuda")

    # Generate prediction
    prediction = hybrid_model.generate(audio_values)

    # ASR override
    asr_transcript = ""
    override_applied = False
    if enable_asr_override:
        try:
            pred_parsed = json.loads(prediction)
            pred_func = pred_parsed.get("function", "") if isinstance(pred_parsed, dict) else ""
        except (json.JSONDecodeError, TypeError):
            pred_func = ""

        if pred_func in COLLAPSE_TARGETS:
            with torch.no_grad():
                asr_ids = whisper_asr.generate(audio_values, max_new_tokens=128)
            asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]
            corrected, info = apply_asr_override(pred_func, asr_transcript, pred_parsed)
            if corrected is not None:
                prediction = json.dumps(corrected)
                override_applied = True

    # Always get ASR transcript for display (even if override not needed)
    if not asr_transcript:
        with torch.no_grad():
            asr_ids = whisper_asr.generate(audio_values, max_new_tokens=128)
        asr_transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

    # Parse JSON for structured display
    try:
        tool_call = json.loads(prediction)
    except json.JSONDecodeError:
        tool_call = {"raw_output": prediction, "parse_error": True}

    status = f"ASR override applied ({pred_func} → {tool_call.get('function', '?')})" if override_applied else "Direct model prediction"

    return tool_call, asr_transcript, status


# ── Gradio UI ──

DESCRIPTION = """
# Speech-to-Tool-Call — Voice to Tool Calls

Speak a command and get a structured JSON tool call. Powered by a hybrid **Whisper-small + Qwen3-0.6B** model
fine-tuned on 25K+ samples across **14 domains** and **68 functions**.

**Examples**: "Set a timer for 5 minutes", "Turn on the headlights", "Translate hello to Spanish", "Play some jazz music"
"""

EXAMPLES_DIR = "./examples"

def build_examples():
    """Build examples list from audio files in examples/ directory."""
    examples = []
    if os.path.isdir(EXAMPLES_DIR):
        for fname in sorted(os.listdir(EXAMPLES_DIR)):
            if fname.endswith((".mp3", ".wav", ".flac", ".ogg")):
                examples.append([os.path.join(EXAMPLES_DIR, fname), True])
    return examples if examples else None


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            asr_toggle = gr.Checkbox(
                value=True, label="Enable ASR Override",
                info="Post-processing that corrects known model collapse patterns using keyword matching",
            )
            submit_btn = gr.Button("Convert to Tool Call", variant="primary", size="lg")

        with gr.Column(scale=1):
            json_output = gr.JSON(label="Tool Call")
            transcript_output = gr.Textbox(label="ASR Transcript", interactive=False)
            status_output = gr.Textbox(label="Status", interactive=False)

    examples = build_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[audio_input, asr_toggle],
            outputs=[json_output, transcript_output, status_output],
            fn=predict,
            cache_examples=False,
        )

    submit_btn.click(
        fn=predict,
        inputs=[audio_input, asr_toggle],
        outputs=[json_output, transcript_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
