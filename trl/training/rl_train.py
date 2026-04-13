from __future__ import annotations

import copy
import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from trl.data.vocab import EOS, Vocab
from trl.generation.sampler import sample
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.model.value_head import ValueHead
from trl.objectives.base import Objectives
from trl.replay_buffer import ReplayBuffer
from trl.training.utils import (
    cleanup_ddp,
    get_lr_scheduler,
    is_main,
    save_checkpoint,
    setup_ddp,
)


def _load_objectives(import_path: str) -> Objectives:
    """Dynamically import objectives factory, e.g. 'mtrl.objectives:build'."""
    module_path, func_name = import_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)()


def _sequence_log_probs(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probs for sequences. Returns (batch, seq_len-1)."""
    logits, _ = model(token_ids[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs of actual next tokens
    targets = token_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)


def rl_train(
    checkpoint_path: str,
    vocab_path: str | None,
    objectives_path: str,
    iterations: int = 10000,
    batch_size: int = 512,
    lr: float = 1e-5,
    kl_beta: float = 0.05,
    pareto_lambda: float = 0.1,
    temperature: float = 1.0,
    temperature_final: float = 0.8,
    replay_fraction: float = 0.1,
    checkpoint_dir: str = "checkpoints_rl/",
    wandb_project: str | None = None,
) -> None:
    local_rank = setup_ddp()
    distributed = torch.distributed.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TransformerConfig(**ckpt["config"])

    if vocab_path is not None:
        vocab = Vocab.load(vocab_path)
    elif "vocab" in ckpt:
        vocab = Vocab(ckpt["vocab"])
    else:
        raise ValueError("Checkpoint has no embedded vocab; pass --vocab explicitly")

    objectives = _load_objectives(objectives_path)
    objectives.pareto_lambda = pareto_lambda

    policy = TransformerLM(config).to(device)
    policy.load_state_dict(ckpt["model"])
    if device.type == "cuda":
        policy = policy.to(torch.bfloat16)

    # Frozen reference model for KL penalty
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Value head
    value_head = ValueHead(config.d_model).to(device)
    if device.type == "cuda":
        value_head = value_head.to(torch.bfloat16)

    if distributed:
        policy = DDP(policy, device_ids=[local_rank])
        value_head = DDP(value_head, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, total_steps=iterations)

    replay = ReplayBuffer(capacity=10000)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Optional wandb
    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(project=wandb_project, config={"rl": True, **asdict(config)})

    raw_policy = policy.module if hasattr(policy, "module") else policy

    for step in range(1, iterations + 1):
        # Anneal temperature
        t = temperature + (temperature_final - temperature) * (step / iterations)

        # Sample from policy
        sequences = sample(raw_policy, batch_size, temperature=t, device=device)

        # Decode token IDs back to token strings for scoring
        token_string_seqs = [vocab.decode(seq) for seq in sequences]

        # Score with objectives
        scored = objectives.evaluate(token_string_seqs)
        for i, seq in enumerate(sequences):
            scored[i].token_ids = seq

        rewards = objectives.get_rewards(scored)

        # Add to replay buffer
        replay.add(scored, rewards.tolist())

        # Compute policy loss (REINFORCE with baseline)
        # Pad sequences for batched computation
        max_len = max(len(s) for s in sequences)
        padded = [s + [EOS] * (max_len - len(s)) for s in sequences]
        token_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        policy_log_probs = _sequence_log_probs(policy, token_tensor)
        with torch.no_grad():
            ref_log_probs = _sequence_log_probs(ref_model, token_tensor)

        # KL penalty per token
        kl = policy_log_probs - ref_log_probs

        # Value baseline
        raw_policy_for_hidden = policy.module if hasattr(policy, "module") else policy
        hidden, _ = raw_policy_for_hidden.hidden(token_tensor[:, :-1])
        raw_vh = value_head.module if hasattr(value_head, "module") else value_head
        values = raw_vh(hidden)

        # Reward is per-sequence; broadcast to token level
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        reward_per_token = reward_tensor.unsqueeze(1).expand_as(policy_log_probs)

        advantage = reward_per_token - values.detach()

        # REINFORCE loss + KL penalty + value loss
        policy_loss = -(policy_log_probs * advantage).mean()
        kl_loss = kl_beta * kl.mean()
        value_loss = F.mse_loss(values, reward_per_token)

        loss = policy_loss + kl_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        if is_main() and step % 100 == 0:
            validity = sum(1 for s in scored if s.valid) / len(scored)
            print(
                f"Step {step}/{iterations}  "
                f"reward={rewards.mean():.3f}  "
                f"validity={validity:.2%}  "
                f"kl={kl.mean().item():.4f}  "
                f"loss={loss.item():.4f}"
            )

        if wandb_run and step % 10 == 0:
            wandb_run.log(
                {
                    "reward_mean": rewards.mean(),
                    "validity": sum(1 for s in scored if s.valid) / len(scored),
                    "kl": kl.mean().item(),
                    "loss": loss.item(),
                },
                step=step,
            )

        if step % 1000 == 0:
            save_checkpoint(
                policy, optimizer, step, asdict(config),
                str(Path(checkpoint_dir) / f"rl_step_{step}.pt"),
                vocab=vocab.token_to_id,
            )

    # Save final checkpoint
    save_checkpoint(
        policy, optimizer, iterations, asdict(config),
        str(Path(checkpoint_dir) / "rl_final.pt"),
        vocab=vocab.token_to_id,
    )

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()
