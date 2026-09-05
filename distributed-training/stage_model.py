import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["STAGE_MODEL_ID"],
    revision=os.environ["STAGE_MODEL_REVISION"],
    local_dir=os.environ["STAGE_MODEL_DEST"],
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
)
