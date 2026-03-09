"""
MLX port of HybridQwen: Whisper encoder + Conv1d projector + Qwen3-0.6B.

Runs entirely on Apple Silicon — no PyTorch, no CUDA, no bitsandbytes.

Usage:
    from mlx_model import HybridQwenMLX
    model = HybridQwenMLX.load("./model_export/")
    result = model.generate(audio_path="test.wav")
"""
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Audio projector: Conv1d compression + MLP (mirrors PyTorch HybridQwen)
# ---------------------------------------------------------------------------

class AudioProjector(nn.Module):
    """Conv1d 4x compression + MLP projector from Whisper dim to Qwen dim."""

    def __init__(self, whisper_dim: int = 768, hidden_size: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=whisper_dim, out_channels=whisper_dim,
            kernel_size=3, stride=2, padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=whisper_dim, out_channels=whisper_dim,
            kernel_size=3, stride=2, padding=1,
        )
        self.norm = nn.LayerNorm(whisper_dim)
        self.proj_linear1 = nn.Linear(whisper_dim, hidden_size)
        self.proj_linear2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """
        Args:
            encoder_output: (batch, 1500, 768) from Whisper encoder
        Returns:
            (batch, 375, hidden_size) projected audio embeddings
        """
        # Conv1d expects (batch, seq, channels) in MLX
        x = nn.relu(self.conv1(encoder_output))   # (batch, 750, 768)
        x = nn.relu(self.conv2(x))                 # (batch, 375, 768)
        x = self.norm(x)                            # LayerNorm
        x = nn.gelu(self.proj_linear1(x))          # (batch, 375, hidden_size)
        x = self.proj_linear2(x)                    # (batch, 375, hidden_size)
        return x


def _load_projector_weights(projector: AudioProjector, projector_pt_path: str):
    """Load PyTorch projector.pt weights into MLX AudioProjector.

    Handles Conv1d weight transposition: PyTorch (out, in, kernel) → MLX (out, kernel, in).
    """
    import torch

    state = torch.load(projector_pt_path, map_location="cpu", weights_only=True)

    def to_mx(t):
        return mx.array(t.float().numpy())

    # Conv1d weights: PyTorch shape (out_ch, in_ch, kernel) → MLX shape (out_ch, kernel, in_ch)
    projector.conv1.weight = to_mx(state["audio_conv1"]["weight"].permute(0, 2, 1))
    projector.conv1.bias = to_mx(state["audio_conv1"]["bias"])
    projector.conv2.weight = to_mx(state["audio_conv2"]["weight"].permute(0, 2, 1))
    projector.conv2.bias = to_mx(state["audio_conv2"]["bias"])

    # LayerNorm
    projector.norm.weight = to_mx(state["conv_norm"]["weight"])
    projector.norm.bias = to_mx(state["conv_norm"]["bias"])

    # MLP projector (Sequential indices: 0=Linear, 1=GELU, 2=Linear)
    projector.proj_linear1.weight = to_mx(state["audio_proj"]["0.weight"])
    projector.proj_linear1.bias = to_mx(state["audio_proj"]["0.bias"])
    projector.proj_linear2.weight = to_mx(state["audio_proj"]["2.weight"])
    projector.proj_linear2.bias = to_mx(state["audio_proj"]["2.bias"])


def _load_whisper_finetuned_layers(whisper_model, projector_pt_path: str):
    """Load fine-tuned Whisper top-4 layer weights from projector.pt.

    Maps PyTorch HF Whisper keys to mlx-whisper block attributes:
        self_attn.q_proj → attn.query
        self_attn.k_proj → attn.key
        self_attn.v_proj → attn.value
        self_attn.out_proj → attn.out
        self_attn_layer_norm → attn_ln
        fc1 → mlp1
        fc2 → mlp2
        final_layer_norm → mlp_ln
    """
    import torch

    state = torch.load(projector_pt_path, map_location="cpu", weights_only=True)
    top_layers = state.get("whisper_top_layers")
    if not top_layers:
        print("No fine-tuned Whisper layers found in projector.pt")
        return

    encoder = whisper_model.encoder if hasattr(whisper_model, "encoder") else whisper_model
    blocks = encoder.blocks
    num_blocks = len(blocks)

    # PyTorch key prefix → (mlx parent attr, mlx child attr)
    KEY_MAP = {
        "self_attn.q_proj": ("attn", "query"),
        "self_attn.k_proj": ("attn", "key"),
        "self_attn.v_proj": ("attn", "value"),
        "self_attn.out_proj": ("attn", "out"),
        "self_attn_layer_norm": ("attn_ln", None),
        "fc1": ("mlp1", None),
        "fc2": ("mlp2", None),
        "final_layer_norm": ("mlp_ln", None),
    }

    for idx_str, layer_state in top_layers.items():
        idx = int(idx_str)
        if idx >= num_blocks:
            print(f"Warning: layer {idx} out of range (model has {num_blocks} blocks)")
            continue

        block = blocks[idx]

        for pt_key, tensor in layer_state.items():
            arr = mx.array(tensor.float().numpy())

            # Split into prefix (e.g. "self_attn.q_proj") and param (e.g. "weight")
            parts = pt_key.rsplit(".", 1)
            if len(parts) == 2:
                prefix, param_name = parts
            else:
                continue

            if prefix not in KEY_MAP:
                print(f"Warning: unknown key prefix '{prefix}' in whisper layer {idx}")
                continue

            parent_attr, child_attr = KEY_MAP[prefix]
            obj = getattr(block, parent_attr)
            if child_attr is not None:
                obj = getattr(obj, child_attr)

            setattr(obj, param_name, arr)

    mx.eval(whisper_model.parameters())
    print(f"Loaded {len(top_layers)} fine-tuned Whisper layers")


# ---------------------------------------------------------------------------
# HybridQwenMLX: Full E2E model
# ---------------------------------------------------------------------------

class HybridQwenMLX:
    """MLX port of HybridQwen for Apple Silicon inference."""

    def __init__(self, whisper_model, whisper_processor, qwen_model, qwen_tokenizer,
                 projector: AudioProjector, instruction_prompt: str):
        self.whisper_model = whisper_model
        self.whisper_processor = whisper_processor
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.projector = projector
        self.instruction_prompt = instruction_prompt

        # Pre-tokenize instruction
        self.instruction_ids = mx.array(
            qwen_tokenizer.encode(instruction_prompt, add_special_tokens=False)
        )

    @classmethod
    def load(cls, export_dir: str, qwen_mlx_path: str = None, quantized: bool = False):
        """Load all model components.

        Args:
            export_dir: Directory containing projector.pt and runanywhere_config.json
                        (or the best_model_qwen3_finetuned subdir)
            qwen_mlx_path: Path to MLX-converted Qwen3 model. If None, looks for
                          qwen3_mlx/ or qwen3_mlx_4bit/ in export_dir.
            quantized: If True, prefer 4-bit quantized MLX model.
        """
        import mlx_whisper
        from mlx_lm import load as mlx_lm_load
        from transformers import WhisperProcessor

        export_dir = Path(export_dir)

        # --- Find model artifacts ---
        # Look for projector.pt in expected locations
        projector_path = None
        config_path = None
        for candidate in [
            export_dir,
            export_dir / "best_model_qwen3_finetuned",
            export_dir / "model_export" / "best_model_qwen3_finetuned",
        ]:
            if (candidate / "projector.pt").exists():
                projector_path = candidate / "projector.pt"
                # Look for config
                for cfg_name in ["runanywhere_config.json", "config.json"]:
                    if (candidate / cfg_name).exists():
                        config_path = candidate / cfg_name
                        break
                break

        if projector_path is None:
            raise FileNotFoundError(
                f"projector.pt not found in {export_dir} or subdirectories"
            )

        # --- Load config ---
        config = {}
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
        instruction_prompt = config.get("instruction_prompt", "<|audio|>Convert to tool call JSON:\n")
        whisper_model_name = config.get("whisper_model", "openai/whisper-small")

        # --- Load Whisper ---
        print(f"Loading Whisper ({whisper_model_name})...")
        whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        # mlx_whisper loads internally; we access the encoder via its transcribe internals
        # We'll use a lightweight wrapper
        whisper_model = _load_mlx_whisper(whisper_model_name)

        # Load fine-tuned top-4 layers
        _load_whisper_finetuned_layers(whisper_model, str(projector_path))

        # --- Load projector ---
        print("Loading audio projector...")
        whisper_dim = 768  # whisper-small
        hidden_size = config.get("hidden_size", 1024)  # Qwen3-0.6B
        projector = AudioProjector(whisper_dim=whisper_dim, hidden_size=hidden_size)
        _load_projector_weights(projector, str(projector_path))
        mx.eval(projector.parameters())

        # --- Load Qwen3 MLX ---
        if qwen_mlx_path is None:
            if quantized and (export_dir / "qwen3_mlx_4bit").exists():
                qwen_mlx_path = str(export_dir / "qwen3_mlx_4bit")
            elif (export_dir / "qwen3_mlx").exists():
                qwen_mlx_path = str(export_dir / "qwen3_mlx")
            else:
                raise FileNotFoundError(
                    "No qwen3_mlx/ directory found. Run mlx_lm.convert first. See README."
                )

        print(f"Loading Qwen3 MLX from {qwen_mlx_path}...")
        qwen_model, qwen_tokenizer = mlx_lm_load(qwen_mlx_path)

        # Get hidden_size from the loaded Qwen model config
        qwen_hidden = qwen_model.args.hidden_size
        if qwen_hidden != hidden_size:
            print(f"Rebuilding projector: hidden_size={qwen_hidden} (was {hidden_size})")
            hidden_size = qwen_hidden
            projector = AudioProjector(whisper_dim=whisper_dim, hidden_size=hidden_size)
            _load_projector_weights(projector, str(projector_path))
            mx.eval(projector.parameters())

        print("All components loaded!")
        return cls(
            whisper_model=whisper_model,
            whisper_processor=whisper_processor,
            qwen_model=qwen_model,
            qwen_tokenizer=qwen_tokenizer,
            projector=projector,
            instruction_prompt=instruction_prompt,
        )

    def encode_audio(self, audio_array: np.ndarray, sr: int = 16000) -> mx.array:
        """Run Whisper encoder + projector on raw audio.

        Args:
            audio_array: float32 numpy array at 16kHz
            sr: sample rate (must be 16000)

        Returns:
            (1, 375, hidden_size) projected audio embeddings
        """
        # Whisper feature extraction
        inputs = self.whisper_processor(
            audio_array, sampling_rate=sr, return_tensors="np"
        )
        features = mx.array(inputs.input_features)  # (1, 80, 3000) from HF processor
        # mlx-whisper expects channels-last: (1, 3000, 80)
        features = mx.transpose(features, axes=(0, 2, 1))

        # Whisper encoder forward
        encoder_output = _whisper_encode(self.whisper_model, features)  # (1, 1500, 768)

        # Projector
        projected = self.projector(encoder_output)  # (1, 375, 512)
        return projected

    def generate(self, audio_path: str = None, audio_array: np.ndarray = None,
                 sr: int = 16000, max_new_tokens: int = 128) -> str:
        """Run full E2E inference: audio → JSON tool call.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            audio_array: Raw float32 numpy array at 16kHz (alternative to audio_path)
            sr: Sample rate for audio_array
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated JSON string
        """
        if audio_array is None:
            if audio_path is None:
                raise ValueError("Provide either audio_path or audio_array")
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000)

        import time as _time

        # Encode audio through Whisper + projector
        t0 = _time.time()
        audio_embeds = self.encode_audio(audio_array, sr)  # (1, 375, hidden_size)
        mx.eval(audio_embeds)
        t_whisper = _time.time()

        # Get instruction token embeddings from Qwen
        instr_ids = self.instruction_ids  # (num_instr_tokens,)
        embed_layer = self.qwen_model.model.embed_tokens
        instr_embeds = embed_layer(instr_ids[None, :])  # (1, num_instr_tokens, hidden_size)

        # Concatenate: [audio_embeds | instruction_embeds]
        current_embeds = mx.concatenate([audio_embeds, instr_embeds], axis=1)
        mx.eval(current_embeds)

        # Greedy decoding with brace-counting early stop
        # mlx_lm models: cache is passed as list, mutated in-place
        # model() returns logits only (not a tuple)
        generated_ids = []
        dummy_inputs = mx.zeros((1, 1), dtype=mx.int32)

        # Build KV cache
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.qwen_model)

        t_gen_start = _time.time()
        for step in range(max_new_tokens):
            if step == 0:
                # First forward: process full prefix (375 + instr tokens)
                logits = self.qwen_model(
                    dummy_inputs,
                    cache=cache,
                    input_embeddings=current_embeds,
                )
            else:
                # Subsequent forwards: only the new token embedding
                next_embed = embed_layer(mx.array([[generated_ids[-1]]]))
                logits = self.qwen_model(
                    dummy_inputs,
                    cache=cache,
                    input_embeddings=next_embed,
                )

            # Greedy: take argmax of last position
            # .item() forces eval, no need for separate mx.eval
            next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            generated_ids.append(next_token)

            # Check stopping conditions
            if next_token == self.qwen_tokenizer.eos_token_id:
                break

            text = self.qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
            if text.strip().endswith('}'):
                if text.count('{') > 0 and text.count('{') == text.count('}'):
                    break

        t_done = _time.time()
        result = self.qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"[timing] whisper+proj: {(t_whisper-t0)*1000:.0f}ms | "
              f"prefill: {(_time.time()-t_gen_start)*1000 - len(generated_ids)*0:.0f}ms | "
              f"generate: {(t_done-t_gen_start)*1000:.0f}ms ({len(generated_ids)} tokens) | "
              f"total: {(t_done-t0)*1000:.0f}ms | output: {result[:80]}")
        return result

    def transcribe(self, audio_path: str = None, audio_array: np.ndarray = None,
                   sr: int = 16000) -> str:
        """Run Whisper ASR transcription (for reference/comparison)."""
        import mlx_whisper

        if audio_array is None:
            if audio_path is None:
                raise ValueError("Provide either audio_path or audio_array")
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000)

        result = mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo="mlx-community/whisper-small-mlx",
        )
        return result["text"]


# ---------------------------------------------------------------------------
# Whisper encoder helpers (using mlx-whisper internals)
# ---------------------------------------------------------------------------

def _load_mlx_whisper(model_name: str = "openai/whisper-small"):
    """Load Whisper model via mlx-whisper, returning the raw model object."""
    from mlx_whisper import load_models

    # mlx-whisper's load_models returns (model, tokenizer, mel_filters)
    # Map HF model name to mlx-community equivalent
    # openai/whisper-small → mlx-community/whisper-small-mlx
    name = model_name.split("/")[-1]  # e.g. "whisper-small"
    mlx_repo = f"mlx-community/{name}-mlx"
    model = load_models.load_model(mlx_repo)
    return model


def _whisper_encode(whisper_model, mel_features: mx.array) -> mx.array:
    """Run just the Whisper encoder to get hidden states.

    Args:
        whisper_model: MLX Whisper model
        mel_features: (batch, 80, 3000) mel spectrogram

    Returns:
        (batch, 1500, 768) encoder hidden states
    """
    return whisper_model.encoder(mel_features)


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MLX HybridQwen inference")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model-dir", default="./model_export",
                        help="Directory with model artifacts")
    parser.add_argument("--qwen-mlx", default=None,
                        help="Path to MLX-converted Qwen3 model")
    parser.add_argument("--quantized", action="store_true",
                        help="Use 4-bit quantized MLX model")
    args = parser.parse_args()

    model = HybridQwenMLX.load(
        args.model_dir,
        qwen_mlx_path=args.qwen_mlx,
        quantized=args.quantized,
    )

    print(f"\nProcessing: {args.audio}")
    t0 = time.time()
    result = model.generate(audio_path=args.audio)
    elapsed = time.time() - t0

    print(f"\nResult ({elapsed*1000:.0f}ms):")
    print(result)

    # Also show ASR for comparison
    print(f"\nASR transcript:")
    transcript = model.transcribe(audio_path=args.audio)
    print(transcript)
