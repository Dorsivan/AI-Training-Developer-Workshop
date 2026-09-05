from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

model_dir = "/shared/models/llama-approved"
seq_len = 128
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
ids = tokenizer(Path("training.txt").read_text(encoding="utf-8"),
                add_special_tokens=False)["input_ids"]
if tokenizer.eos_token_id is not None:
    ids.append(tokenizer.eos_token_id)
usable = len(ids) // seq_len * seq_len
if usable == 0:
    raise ValueError("Provide at least one complete token block")
Path("/shared/data").mkdir(parents=True, exist_ok=True)
np.save("/shared/data/train_tokens.npy",
        np.asarray(ids[:usable], dtype=np.int64).reshape(-1, seq_len))
