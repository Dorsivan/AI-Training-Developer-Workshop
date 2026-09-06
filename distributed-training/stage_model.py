"""Download a pretrained model from Hugging Face Hub to shared storage.

Run this before training when using a real pretrained model (e.g. Llama).
It fetches only the files needed for training (config, weights, tokenizer)
and places them in a local directory on shared storage so the training pods
can load them without network access.

Required environment variables:
  STAGE_MODEL_ID       – Hugging Face repo id (e.g. "meta-llama/Llama-3.2-1B")
  STAGE_MODEL_REVISION – git revision / tag to pin (e.g. "main")
  STAGE_MODEL_DEST     – local directory to save into (e.g. "/shared/models/llama")
  HF_TOKEN             – Hugging Face access token (optional, for gated models)
"""
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["STAGE_MODEL_ID"],
    revision=os.environ["STAGE_MODEL_REVISION"],
    local_dir=os.environ["STAGE_MODEL_DEST"],
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
)
