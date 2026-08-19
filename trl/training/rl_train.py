from __future__ import annotations

import copy
import importlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from trl.data.vocab import BOS, PAD, Vocab
from trl.generation.sampler import sample
from trl.model.transformer import TransformerConfig, TransformerLM
from trl.model.value_head import ValueHead
from trl.objectives.base import Objectives, ScoredItem
from trl.training.pretrain import _autocast_context, resolve_precision
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


def _prepare_sequences(
    sequences: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend BOS, pad with PAD, and return the next-token validity mask."""
    with_bos = [[BOS, *sequence] for sequence in sequences]
    max_len = max(len(sequence) for sequence in with_bos)
    padded = [sequence + [PAD] * (max_len - len(sequence)) for sequence in with_bos]
    token_ids = torch.tensor(padded, dtype=torch.long, device=device)
    target_mask = token_ids[:, 1:] != PAD
    return token_ids, target_mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _global_pareto_rewards(
    objectives: Objectives,
    local_scored: list[ScoredItem],
) -> tuple[np.ndarray, list[ScoredItem]]:
    """Pareto-rank one global DDP batch, then return this rank's rewards."""
    stripped = [
        ScoredItem(
            token_ids=[],
            scores=dict(item.scores),
            valid=item.valid,
            rejection_reason=item.rejection_reason,
        )
        for item in local_scored
    ]
    if not torch.distributed.is_initialized():
        return objectives.get_rewards(stripped), stripped

    world_size = torch.distributed.get_world_size()
    gathered: list[list[ScoredItem] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, stripped)
    all_scored = [item for rank_items in gathered if rank_items is not None for item in rank_items]
    all_rewards = objectives.get_rewards(all_scored)
    rank = torch.distributed.get_rank()
    start = sum(len(gathered[i] or []) for i in range(rank))
    stop = start + len(stripped)
    return all_rewards[start:stop], all_scored


def _score_summary(scored: list[ScoredItem], objectives: Objectives) -> dict[str, object]:
    valid = [item for item in scored if item.valid]
    means = {
        objective.name: float(np.mean([item.scores[objective.name] for item in valid]))
        for objective in objectives.objectives
        if valid
    }
    rejections = Counter(item.rejection_reason for item in scored if not item.valid)
    return {
        "validity": len(valid) / max(1, len(scored)),
        "reward_mean": float(objectives.get_rewards(scored).mean()),
        "objective_means": means,
        "rejections": dict(rejections.most_common()),
    }


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
    replay_fraction: float = 0.0,
    precision: str = "auto",
    checkpoint_every: int = 100,
    log_every: int = 10,
    checkpoint_dir: str = "checkpoints_rl/",
    wandb_project: str | None = None,
) -> None:
    local_rank = setup_ddp()
    distributed = torch.distributed.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(precision, device)
    world_size = torch.distributed.get_world_size() if distributed else 1
    if batch_size <= 0 or batch_size % world_size:
        raise ValueError(f"batch_size={batch_size} must be positive and divisible by {world_size}")
    local_batch_size = batch_size // world_size
    if replay_fraction != 0:
        raise ValueError("replay_fraction is not implemented; set it to 0")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")
    if log_every <= 0:
        raise ValueError("log_every must be > 0")

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

    policy: torch.nn.Module = TransformerLM(config).to(device)
    policy.load_state_dict(ckpt["model"])

    # Frozen reference model for KL penalty
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Value head
    value_head: torch.nn.Module = ValueHead(config.d_model).to(device)

    if distributed:
        policy = DDP(policy, device_ids=[local_rank])
        value_head = DDP(value_head, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, total_steps=iterations)
    scaler = torch.amp.GradScaler("cuda", enabled=resolved_precision == "fp16")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Optional wandb
    wandb_run = None
    if wandb_project and is_main():
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "rl": True,
                **asdict(config),
                "batch_size": batch_size,
                "precision": resolved_precision,
            },
        )

    raw_policy = cast(
        TransformerLM,
        policy.module if hasattr(policy, "module") else policy,
    )

    for step in range(1, iterations + 1):
        # Anneal temperature
        t = temperature + (temperature_final - temperature) * (step / iterations)

        # Sample from policy
        sequences = sample(
            raw_policy,
            local_batch_size,
            max_len=config.max_seq_len,
            temperature=t,
            device=device,
        )

        # Decode token IDs back to token strings for scoring
        token_string_seqs = [vocab.decode(seq) for seq in sequences]

        # Score with objectives
        scored = objectives.evaluate(token_string_seqs)
        for i, seq in enumerate(sequences):
            scored[i].token_ids = seq

        rewards, global_scored = _global_pareto_rewards(objectives, scored)
        summary = _score_summary(global_scored, objectives)

        # Compute policy loss (REINFORCE with baseline)
        token_tensor, target_mask = _prepare_sequences(sequences, device)
        # Keep dropout disabled so likelihoods match the sampling distribution.
        policy.eval()
        with _autocast_context(device, resolved_precision):
            policy_log_probs = _sequence_log_probs(policy, token_tensor)
        with torch.no_grad():
            with _autocast_context(device, resolved_precision):
                ref_log_probs = _sequence_log_probs(ref_model, token_tensor)

        # KL penalty per token
        kl = policy_log_probs - ref_log_probs

        # Value baseline
        raw_policy_for_hidden = policy.module if hasattr(policy, "module") else policy
        with torch.no_grad():
            with _autocast_context(device, resolved_precision):
                hidden, _ = raw_policy_for_hidden.hidden(token_tensor[:, :-1])
        with _autocast_context(device, resolved_precision):
            values = value_head(hidden)

        # Reward is per-sequence; broadcast to token level
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        reward_per_token = reward_tensor.unsqueeze(1).expand_as(policy_log_probs)

        advantage = reward_per_token - values.detach().float()

        # REINFORCE loss + KL penalty + value loss
        policy_loss = -_masked_mean(policy_log_probs.float() * advantage, target_mask)
        kl_loss = kl_beta * _masked_mean(kl.float(), target_mask)
        value_loss = _masked_mean(
            F.mse_loss(values.float(), reward_per_token, reduction="none"),
            target_mask,
        )

        loss = policy_loss + kl_loss + value_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_head.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if is_main() and step % log_every == 0:
            objective_text = " ".join(
                f"{name}={value:.3f}"
                for name, value in cast(dict[str, float], summary["objective_means"]).items()
            )
            rejection_text = ", ".join(
                f"{reason}:{count}"
                for reason, count in list(cast(dict[str, int], summary["rejections"]).items())[:3]
            )
            print(
                f"Step {step}/{iterations}  "
                f"reward={cast(float, summary['reward_mean']):.3f}  "
                f"validity={cast(float, summary['validity']):.2%}  "
                f"kl={_masked_mean(kl.float(), target_mask).item():.4f}  "
                f"loss={loss.item():.4f}  {objective_text}  "
                f"rejected=[{rejection_text}]",
                flush=True,
            )

        if wandb_run and step % 10 == 0:
            wandb_run.log(
                {
                    "reward_mean": summary["reward_mean"],
                    "validity": summary["validity"],
                    "kl": _masked_mean(kl.float(), target_mask).item(),
                    "loss": loss.item(),
                    **{
                        f"objective/{name}": value
                        for name, value in cast(
                            dict[str, float], summary["objective_means"]
                        ).items()
                    },
                },
                step=step,
            )

        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(
                policy,
                optimizer,
                step,
                asdict(config),
                str(Path(checkpoint_dir) / f"rl_step_{step}.pt"),
                vocab=vocab.token_to_id,
                scheduler=scheduler,
                scaler=scaler,
            )

    # Save final checkpoint
    save_checkpoint(
        policy,
        optimizer,
        iterations,
        asdict(config),
        str(Path(checkpoint_dir) / "rl_final.pt"),
        vocab=vocab.token_to_id,
        scheduler=scheduler,
        scaler=scaler,
    )

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()
