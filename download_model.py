"""
Download model files from Modal volumes to local model/ directory.

Usage:
    modal run download_model.py

This downloads the trained model checkpoint needed for the Gradio demo (app.py)
and HuggingFace Spaces deployment.
"""
import modal
import os

image = modal.Image.debian_slim(python_version="3.10")
model_vol = modal.Volume.from_name("hybrid-model-storage")
app = modal.App("download-model")

MODEL_NAME = "final_model_qwen3_finetuned"
FILES = ["config.json", "adapter_model.bin", "projector.pt"]


@app.function(
    image=image,
    volumes={"/models": model_vol},
    timeout=120,
)
def get_model_files():
    """Read model files from Modal volume and return as bytes."""
    model_dir = f"/models/{MODEL_NAME}"
    result = {}
    for fname in FILES:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                result[fname] = f.read()
            print(f"Read {fname}: {len(result[fname]):,} bytes")
        else:
            print(f"WARNING: {path} not found")
    return result


@app.local_entrypoint()
def main():
    os.makedirs("model", exist_ok=True)
    files = get_model_files.remote()
    for fname, data in files.items():
        out_path = os.path.join("model", fname)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved {out_path} ({len(data):,} bytes)")
    print(f"\nDone. Model files in ./model/")
