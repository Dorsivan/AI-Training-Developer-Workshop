"""Pre-tokenize a training text file into fixed-length blocks for FSDP training.

This script reads a plain-text file, tokenizes it using the model's tokenizer,
and saves the result as a NumPy .npy array of shape [num_blocks, seq_len].
Each row is one training sample — a contiguous chunk of token IDs.

The output file is what train_fsdp.py expects when TOKEN_FILE is set.
Run this once on shared storage before launching the distributed training job.
"""
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

# Path to the pretrained model directory (must contain tokenizer files)
model_dir = "/shared/models/llama-approved"
seq_len = 128

# Tokenize the entire training text into a flat list of token IDs
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
ids = tokenizer(Path("training.txt").read_text(encoding="utf-8"),
                add_special_tokens=False)["input_ids"]

# Append EOS token to mark end of document
if tokenizer.eos_token_id is not None:
    ids.append(tokenizer.eos_token_id)

# Truncate to an exact multiple of seq_len so every block is full
usable = len(ids) // seq_len * seq_len
if usable == 0:
    raise ValueError("Provide at least one complete token block")

# Save as a 2D int64 array: each row is one [seq_len] training sample
Path("/shared/data").mkdir(parents=True, exist_ok=True)
np.save("/shared/data/train_tokens.npy",
        np.asarray(ids[:usable], dtype=np.int64).reshape(-1, seq_len))
