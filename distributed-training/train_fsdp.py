def train_llm():
    """Distributed fine-tuning of a Llama model using PyTorch FSDP across multiple nodes.

    FSDP (Fully Sharded Data Parallel) splits the model parameters, gradients, and
    optimizer states across all GPUs. Each GPU holds only a shard of the model in
    memory; during forward/backward, FSDP temporarily gathers the full parameters
    via all-gather, computes, then discards them. This trades network bandwidth for
    GPU memory, allowing models larger than a single GPU's memory to be trained.

    Environment variables control the run:
      STEPS          – total training steps
      SEQ_LEN        – sequence length per sample
      SAVE_EVERY     – checkpoint every N steps
      CHECKPOINT_DIR – where to write sharded checkpoints (on shared storage)
      RESUME_FROM    – path to a prior checkpoint to resume from (optional)
      MODEL_DIR      – path to a pretrained model on shared storage (optional)
      TOKEN_FILE     – path to a .npy file of tokenized training data (optional)
    """
    import gc
    import json
    import os
    from datetime import timedelta
    from pathlib import Path

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_state_dict, set_state_dict, set_model_state_dict,
    )
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    # ── 1. Distributed setup ──────────────────────────────────────────────
    # Each GPU worker gets a LOCAL_RANK from torchrun. We bind this process to
    # that GPU, then join the NCCL process group so all workers can communicate.
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This example requires BF16-capable NVIDIA GPUs")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()

        # ── 2. Read configuration from environment ────────────────────────
        seed = int(os.getenv("SEED", "42"))
        steps = int(os.getenv("STEPS", "40"))
        batch_size = int(os.getenv("BATCH_SIZE", "1"))
        seq_len = int(os.getenv("SEQ_LEN", "128"))
        save_every = int(os.getenv("SAVE_EVERY", "20"))
        model_dir = os.getenv("MODEL_DIR", "")
        token_file = os.getenv("TOKEN_FILE", "")
        resume = os.getenv("RESUME_FROM", "")
        run_dir = Path(os.environ["CHECKPOINT_DIR"])
        if min(steps, batch_size, seq_len, save_every) < 1:
            raise ValueError("Training counts must be positive")
        if model_dir and not token_file:
            raise ValueError("Pretrained training requires TOKEN_FILE")
        if not model_dir and token_file:
            raise ValueError("TOKEN_FILE requires the matching MODEL_DIR")

        print(json.dumps({"rank": rank, "local_rank": local_rank,
                          "world_size": world, "host": os.getenv("HOSTNAME")}), flush=True)

        # ── 3. Build model config ─────────────────────────────────────────
        # If MODEL_DIR is set, load a real pretrained Llama config from disk.
        # Otherwise, create a tiny Llama config for smoke-testing the pipeline.
        torch.manual_seed(seed)
        if model_dir:
            config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        else:
            config = LlamaConfig(
                vocab_size=256, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=4, max_position_embeddings=2048,
                tie_word_embeddings=False, attention_dropout=0.0,
            )
        if config.model_type != "llama" or config.tie_word_embeddings:
            raise ValueError("This pinned example supports untied Llama models only")
        if seq_len > config.max_position_embeddings:
            raise ValueError("SEQ_LEN exceeds configured context length")
        config.use_cache = False
        config._attn_implementation = "sdpa"

        # ── 4. Create model on "meta" device, then apply FSDP sharding ───
        # Allocating on meta avoids materializing the full model on every GPU.
        # After FSDP wraps each transformer layer + the outer model, we call
        # to_empty() to allocate real GPU memory for just this rank's shard.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        mesh = init_device_mesh("cuda", (world,))
        # BF16 for forward/backward computation, FP32 for gradient reduction
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        model.to_empty(device=device)

        # Rotary embeddings need real tensors on-device (not meta)
        model.model.rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

        # ── 5. Initialize weights ─────────────────────────────────────────
        # On fresh start (no resume): rank 0 loads the full state dict (either
        # from a pretrained checkpoint or random init), then broadcasts it to
        # all other ranks via set_model_state_dict with broadcast_from_rank0.
        if not resume:
            full_state = {}
            if rank == 0:
                with torch.device("cpu"):
                    if model_dir:
                        source = AutoModelForCausalLM.from_pretrained(
                            model_dir, config=config, torch_dtype=torch.float32,
                            low_cpu_mem_usage=True, local_files_only=True,
                            use_safetensors=True,
                        )
                    else:
                        source = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
                full_state = source.state_dict()
                del source
            set_model_state_dict(
                model, full_state,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
            del full_state
            gc.collect()

        # ── 6. Training mode and optimizer ────────────────────────────────
        model.train()
        # Gradient checkpointing trades compute for memory: instead of storing
        # every activation, it recomputes them during backward.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)

        # ── 7. Checkpointable training state ──────────────────────────────
        # Wraps model + optimizer + step counter so that dcp.save/load can
        # persist and restore the entire training state across all ranks.
        class TrainingState:
            def __init__(self):
                self.step = 0

            def state_dict(self):
                model_state, optimizer_state = get_state_dict(model, optimizer)
                return {"model": model_state, "optimizer": optimizer_state, "step": self.step}

            def load_state_dict(self, state):
                set_state_dict(model, optimizer, model_state_dict=state["model"],
                               optim_state_dict=state["optimizer"])
                self.step = int(state["step"])

        state = TrainingState()
        # The "contract" records the exact topology + data config. Resuming from
        # a checkpoint with a different contract is rejected to avoid silent bugs.
        contract = {"world_size": world, "seed": seed, "batch_size": batch_size,
                    "seq_len": seq_len, "model_dir": model_dir, "token_file": token_file}

        # ── 8. Resume from checkpoint (optional) ──────────────────────────
        if resume:
            marker = Path(resume) / "complete.json"
            if not marker.is_file():
                raise ValueError("Checkpoint lacks complete.json; do not resume a partial save")
            saved = json.loads(marker.read_text())
            if saved["contract"] != contract:
                raise ValueError("Resume requires the same data, paths, topology, seed and batch shape")
            dcp.load({"training": state}, checkpoint_id=resume)

        # ── 9. Dataset ────────────────────────────────────────────────────
        # If TOKEN_FILE is provided, loads pre-tokenized data from a .npy file
        # (see prepare_tokens.py). Otherwise, generates synthetic sequential
        # token IDs for smoke-testing — no real text needed.
        class TokenBlocks(Dataset):
            def __init__(self):
                self.blocks = np.load(token_file, mmap_mode="r") if token_file else None
                if self.blocks is not None:
                    if self.blocks.ndim != 2 or self.blocks.shape[1] != seq_len:
                        raise ValueError("TOKEN_FILE must have shape [samples, SEQ_LEN]")
                    if not np.issubdtype(self.blocks.dtype, np.integer):
                        raise ValueError("Token array must have integer dtype")

            def __len__(self):
                return len(self.blocks) if self.blocks is not None else 4096

            def __getitem__(self, index):
                if self.blocks is None:
                    return (torch.arange(seq_len) + index) % config.vocab_size
                tokens = torch.from_numpy(np.array(self.blocks[index], dtype=np.int64, copy=True))
                if tokens.min() < 0 or tokens.max() >= config.vocab_size:
                    raise ValueError("Token ID outside model vocabulary")
                return tokens

        # DistributedSampler ensures each rank sees a non-overlapping slice of
        # the dataset, so every sample is processed exactly once per epoch.
        dataset = TokenBlocks()
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     seed=seed, shuffle=True, drop_last=True)
        sampler.set_epoch(0)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=0, drop_last=True)
        if steps > len(loader) or state.step >= steps:
            raise ValueError("Require checkpoint step < STEPS <= batches in one epoch")

        # ── 10. Checkpoint saving ─────────────────────────────────────────
        # dcp.save writes one shard file per rank — each rank saves only its
        # portion of the model/optimizer. A complete.json marker is written by
        # rank 0 after all ranks finish, so partial saves are never resumed.
        def save_checkpoint():
            path = run_dir / f"step-{state.step:06d}"
            if path.exists():
                raise FileExistsError(f"Use a new checkpoint directory: {path}")
            dist.barrier()
            dcp.save({"training": state}, checkpoint_id=str(path))
            dist.barrier()
            if rank == 0:
                (path / "complete.json").write_text(json.dumps({"contract": contract, "step": state.step}))
            dist.barrier()

        # ── 11. Training loop ─────────────────────────────────────────────
        for batch_index, tokens in enumerate(loader):
            # Skip batches already completed (when resuming from checkpoint)
            if batch_index < state.step:
                continue
            if batch_index >= steps:
                break
            torch.manual_seed(seed + batch_index * world + rank)
            tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass: labels=tokens means next-token prediction (causal LM)
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss

            # Safety check: all ranks must have finite loss, otherwise abort
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("Non-finite loss on at least one rank")

            # Backward pass: FSDP handles gradient sharding + all-reduce
            loss.backward()
            optimizer.step()

            # Average loss across all ranks for logging
            mean_loss = loss.detach().float()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world
            state.step = batch_index + 1
            if rank == 0:
                print(f"step={state.step} global_mean_loss={mean_loss.item():.5f}", flush=True)

            # Save checkpoint periodically and at the final step
            if state.step % save_every == 0 or state.step == steps:
                save_checkpoint()
        print(f"rank={rank} completed steps={state.step}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    train_llm()
