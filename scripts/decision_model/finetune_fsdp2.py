#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune decision models with LoRA or full decoder updates using FSDP2.

Fine-tuning optimizes hard- or soft-target cross-entropy over option-code logits
at a trailing mask token. Full fine-tuning updates encoder layers and the
final norm while embeddings and the output head stay frozen. LoRA updates
projection adapters and merges them into the exported Hugging Face checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from scripts.decision_model.checkpoint_manifest import (
    file_sha256,
    safetensors_inventory,
    weight_manifest,
)
from scripts.decision_model.runtime import (
    DecisionRows,
    Scorer,
    batched_option_logits,
    enable_bidirectional,
    resolve_prompt_module,
)
from scripts.decision_model.runtime import (
    collate_examples as collate_examples,  # noqa: PLC0414 -- public re-export
)
from scripts.shared.fsdp2_common import (
    cosine_lr as recipe_cosine_lr,
)
from scripts.shared.fsdp2_common import (
    fsdp2_wrap as recipe_fsdp2_wrap,
)
from scripts.shared.fsdp2_common import (
    load_dcp_checkpoint as recipe_load_dcp_checkpoint,
)
from scripts.shared.fsdp2_common import (
    resolve_grad_accum as recipe_resolve_grad_accum,
)
from scripts.shared.fsdp2_common import (
    save_dcp_checkpoint as recipe_save_dcp_checkpoint,
)
from scripts.shared.fsdp2_common import (
    validate_resume_step as recipe_validate_resume_step,
)

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("lora", "full"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt-module", default="", help="Path to the model-supplied prompt module"
    )
    parser.add_argument(
        "--drop-config-key",
        action="append",
        default=[],
        help="Top-level model configuration key to omit from exported checkpoints",
    )
    parser.add_argument("--attn-implementation", default="sdpa")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Examples processed together on each GPU before gradient accumulation",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--canonical-fraction", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-dcp", default="")
    parser.add_argument(
        "--stop-step",
        type=int,
        default=0,
        help="Stop at an absolute completed step while preserving the full-run LR schedule",
    )
    parser.add_argument("--skip-initial-eval", action="store_true")

    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--targets", default=",".join(LORA_TARGETS))
    parser.add_argument("--expected-lora-modules", type=int)
    parser.add_argument("--expected-trainable-parameters", type=int)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--length-ordered",
        action="store_true",
        help="Process longest canonical examples first for batch-capacity testing",
    )
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--no-save-best", action="store_true")
    parser.add_argument("--no-save-last", action="store_true")
    args = parser.parse_args(argv)

    if args.lr is None:
        args.lr = 3e-5 if args.method == "lora" else 1e-6
    if args.batch is None:
        args.batch = 16 if args.method == "lora" else 64
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.per_device_batch_size < 1:
        parser.error("--per-device-batch-size must be positive")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive")
    if not math.isfinite(args.warmup) or not 0.0 <= args.warmup <= 1.0:
        parser.error("--warmup must be in [0, 1]")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        parser.error("--adam-beta1 and --adam-beta2 must be in [0, 1)")
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        parser.error("--adam-eps must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be finite and positive")
    if not math.isfinite(args.kl_weight) or args.kl_weight < 0:
        parser.error("--kl-weight must be finite and non-negative")
    if args.rank < 1:
        parser.error("--rank must be positive")
    if args.method == "lora" and args.alpha < 1:
        parser.error("--alpha must be positive for LoRA")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if not targets:
        parser.error("--targets must contain at least one module name")
    args.targets = ",".join(targets)
    if not 0.0 <= args.canonical_fraction <= 1.0:
        parser.error("--canonical-fraction must be in [0, 1]")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if args.train_limit < 0:
        parser.error("--train-limit cannot be negative")
    if args.eval_limit < 0:
        parser.error("--eval-limit cannot be negative")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.stop_step < 0:
        parser.error("--stop-step cannot be negative")
    if any(not key or "." in key for key in args.drop_config_key):
        parser.error("--drop-config-key must name a non-empty top-level configuration key")
    for option in ("expected_lora_modules", "expected_trainable_parameters"):
        if getattr(args, option) is not None and getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    return args


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LoRALinear(nn.Module):
    """A frozen linear projection with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    targets: set[str],
) -> list[str]:
    """Replace target decoder projections and leave only adapters trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and name.startswith("encoder.layers.")
            and name.rsplit(".", 1)[-1] in targets
        ):
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"No decoder projections matched LoRA targets: {sorted(targets)}")

    for name, base in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(base, rank, alpha, dropout))
    return [name for name, _ in replacements]


def mark_full_trainable(model: nn.Module) -> list[str]:
    """Fine-tune decoder blocks and final norm; freeze embeddings and output head."""
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("encoder.layers.", "encoder.norm."))
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise ValueError("No decision-model decoder parameters were selected for full fine-tuning")
    return names


def shard_scorer(
    scorer: Scorer,
    *,
    cpu_offload: bool,
    device: torch.device,
) -> Scorer:
    """Shard each decoder layer and then the scorer root."""
    if not cpu_offload:
        layer_class = type(scorer.inner.encoder.layers[0]).__name__
        recipe_fsdp2_wrap(scorer, layer_class)
        scorer.to(device)
        scorer._fsdp2_implementation = "shared-fsdp2"  # type: ignore[attr-defined]
        return scorer

    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    options: dict = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "reshard_after_forward": True,
    }
    if cpu_offload:
        options["offload_policy"] = CPUOffloadPolicy()
        for module in scorer.modules():
            for name, buffer in list(module.named_buffers(recurse=False)):
                module.register_buffer(
                    name,
                    buffer.to(device),
                    persistent=name not in module._non_persistent_buffers_set,
                )
    for layer in scorer.inner.encoder.layers:
        fully_shard(layer, **options)
    fully_shard(scorer, **options)
    scorer._fsdp2_implementation = "cpu-offload"  # type: ignore[attr-defined]
    return scorer


def fsdp2_inventory(scorer: Scorer) -> dict[str, int]:
    """Validate root/layer wrapping and parameter distribution."""
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    units = [module for module in scorer.modules() if isinstance(module, FSDPModule)]
    expected_units = len(scorer.inner.encoder.layers) + 1
    if len(units) != expected_units:
        raise RuntimeError(
            f"Expected {expected_units} FSDP2 units (root + layers), found {len(units)}"
        )
    parameters = list(scorer.named_parameters())
    unsharded = [name for name, parameter in parameters if not isinstance(parameter, DTensor)]
    if unsharded:
        raise RuntimeError("FSDP2 left parameters unsharded: " + ", ".join(unsharded[:10]))
    return {
        "fsdp2_units": len(units),
        "sharded_parameter_tensors": len(parameters),
    }


def audit_parameter(scorer: Scorer, method: str) -> tuple[str, nn.Parameter, torch.Tensor]:
    """Snapshot a trainable local shard to measure optimizer updates."""
    parameters = [
        (name, parameter)
        for name, parameter in scorer.named_parameters()
        if parameter.requires_grad
    ]
    if method == "lora":
        parameters = [item for item in parameters if item[0].endswith(".lora_B")]
    if not parameters:
        raise RuntimeError(f"No trainable parameter available to audit for {method}")
    name, parameter = parameters[0]
    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    return name, parameter, local.detach().clone()


def audited_parameter_delta(
    parameter: nn.Parameter,
    baseline: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute the maximum shard update across ranks."""
    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    delta = (
        (local.detach() - baseline).abs().max().float().to(device)
        if local.numel()
        else torch.zeros((), device=device)
    )
    dist.all_reduce(delta, op=dist.ReduceOp.MAX)
    return delta.item()


def row_loss(example: dict, logits: torch.Tensor, kl_weight: float = 0.0) -> torch.Tensor:
    log_probs = logits.log_softmax(-1)
    soft = (
        torch.tensor(example["soft"], device=logits.device, dtype=log_probs.dtype)
        if example["soft"] is not None
        else None
    )
    if kl_weight > 0:
        loss = -log_probs[example["gold"]]
        if soft is not None:
            loss = loss + kl_weight * (soft * (soft.clamp_min(1e-12).log() - log_probs)).sum()
        return loss
    target = soft
    if target is None:
        target = F.one_hot(
            torch.tensor(example["gold"], device=logits.device),
            len(example["cand"]),
        ).to(log_probs.dtype)
    return -(target * log_probs).sum()


def learning_rate(step: int, total: int, peak: float, warmup: float) -> float:
    """Compute linear warmup followed by cosine decay."""
    warm_steps = int(warmup * total)
    return recipe_cosine_lr(step, warm_steps, total, peak, 0.1 * peak)


def merge_lora_state_dict(
    state: dict[str, torch.Tensor], scaling: float
) -> dict[str, torch.Tensor]:
    """Merge adapter weights and restore standard model parameter names."""
    normalized = {key.removeprefix("inner."): value for key, value in state.items()}
    merged: dict[str, torch.Tensor] = {}
    suffix = ".base.weight"
    for key, value in normalized.items():
        if key.endswith((".lora_A", ".lora_B")):
            continue
        if key.endswith(suffix):
            stem = key[: -len(suffix)]
            adapter_a = normalized[f"{stem}.lora_A"].float()
            adapter_b = normalized[f"{stem}.lora_B"].float()
            merged[f"{stem}.weight"] = (value.float() + adapter_b @ adapter_a * scaling).to(
                torch.bfloat16
            )
        elif ".base." in key:
            merged[key.replace(".base.", ".")] = value
        else:
            merged[key] = value
    return merged


def resume_signature(
    args: argparse.Namespace, world: int, total_steps: int, prompt_module: Path
) -> dict:
    """Fingerprint inputs and settings that determine optimizer/data continuity."""
    keys = (
        "method",
        "lr",
        "batch",
        "per_device_batch_size",
        "seed",
        "warmup",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "max_grad_norm",
        "kl_weight",
        "canonical_fraction",
        "max_tokens",
        "train_limit",
        "eval_limit",
        "rank",
        "alpha",
        "dropout",
        "targets",
        "length_ordered",
        "attn_implementation",
        "no_grad_checkpointing",
        "no_save_best",
        "drop_config_key",
        "eval_every",
    )
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        identity = weight_manifest(checkpoint)
        model = {
            "weights": identity["sha256"],
            "index": identity["index_sha256"],
            "assets": {
                file.relative_to(checkpoint).as_posix(): file_sha256(file)
                for file in sorted(checkpoint.rglob("*"))
                if file.is_file() and file.suffix in {".json", ".py", ".model", ".tiktoken", ".txt"}
            },
        }
    else:
        if not args.model_revision:
            raise ValueError("Hub checkpoints require --model-revision for reproducible resume")
        model = {"id": args.checkpoint, "revision": args.model_revision}
    return {
        "settings": {key: getattr(args, key) for key in keys},
        "world_size": world,
        "total_steps": total_steps,
        "train_sha256": file_sha256(Path(args.train)),
        "eval_sha256": file_sha256(Path(args.eval)) if args.eval else None,
        "prompt_module_sha256": file_sha256(prompt_module),
        "model": model,
    }


def read_resume_state(checkpoint: str | Path, expected: dict) -> dict:
    """Reject incompatible resumes before loading model or optimizer state."""
    path = Path(checkpoint) / "decision-training-state.json"
    if not path.is_file():
        raise ValueError(f"Resume checkpoint lacks configuration metadata: {path}")
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or state.get("signature") != expected:
        raise ValueError("Resume inputs/settings differ from the original training run")
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Resume metadata has an invalid completed step")
    if state.get("best_saved"):
        best = Path(checkpoint) / "best"
        receipt = json.loads((best / "pure-pytorch-receipt.json").read_text())
        manifest = weight_manifest(best)
        if manifest["sha256"] != receipt.get("model_sha256"):
            raise ValueError("Saved best checkpoint does not match its weight manifest")
        if "index_sha256" in receipt and manifest["index_sha256"] != receipt["index_sha256"]:
            raise ValueError("Saved best checkpoint index does not match its receipt")
        if (
            receipt.get("prompt_module_sha256") is not None
            and file_sha256(best / "decision_prompt.py") != receipt["prompt_module_sha256"]
        ):
            raise ValueError("Saved best checkpoint prompt module does not match its receipt")
    return state


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def save_dcp_checkpoint(
    scorer: Scorer,
    optimizer: torch.optim.Optimizer,
    step: int,
    target: str,
    rank: int,
    training_state: dict,
    best_checkpoint: Path | None = None,
) -> Path:
    """Save model/optimizer state, training metadata, and per-rank RNG state."""
    if rank == 0 and (Path(target) / f"step_{step}").exists():
        raise FileExistsError(f"Distributed checkpoint already exists: {target}/step_{step}")
    dist.barrier()
    checkpoint = Path(recipe_save_dcp_checkpoint(scorer, optimizer, step, target))
    torch.save(capture_rng_state(), checkpoint / f"rng-rank-{rank:05d}.pt")
    if rank == 0:
        if best_checkpoint is not None:
            shutil.copytree(best_checkpoint, checkpoint / "best")
        payload = {**training_state, "step": step, "best_saved": best_checkpoint is not None}
        temporary = checkpoint / ".decision-training-state.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(checkpoint / "decision-training-state.json")
    dist.barrier()
    if rank == 0:
        print(f"saved distributed checkpoint: {checkpoint}", flush=True)
    return checkpoint


def load_dcp_checkpoint(
    scorer: Scorer,
    optimizer: torch.optim.Optimizer,
    source: str,
    rank: int,
) -> int:
    """Restore model/optimizer state and per-rank RNG state."""
    step = recipe_load_dcp_checkpoint(scorer, optimizer, source)
    rng_path = Path(source) / f"rng-rank-{rank:05d}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Missing rank-local RNG checkpoint: {rng_path}")
    state = torch.load(rng_path, map_location="cpu", weights_only=True)
    restore_rng_state(state)
    dist.barrier()
    if rank == 0:
        print(f"loaded distributed checkpoint: {source}", flush=True)
    return step


def gather_full_export_state(
    scorer: Scorer,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Gather parameters and persistent buffers, then restore sharded storage."""
    from torch.distributed.fsdp import FSDPModule

    units = [*scorer.inner.encoder.layers, scorer]
    if not all(isinstance(unit, FSDPModule) for unit in units):
        raise RuntimeError("Expected every export unit to be an FSDP2 module")
    materialized = []
    try:
        for index, unit in enumerate(units):
            if rank == 0:
                print(
                    f"checkpoint unshard: {index + 1}/{len(units)} start",
                    flush=True,
                )
            unit.unshard()
            materialized.append(unit)
            if rank == 0:
                print(
                    f"checkpoint unshard: {index + 1}/{len(units)} done",
                    flush=True,
                )
        torch.cuda.synchronize()

        state: dict[str, torch.Tensor] = {}
        parameters = list(scorer.named_parameters())
        for index, (name, parameter) in enumerate(parameters):
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            if tuple(local.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"FSDP2 unshard left {name} local: "
                    f"{tuple(local.shape)} != {tuple(parameter.shape)}"
                )
            if rank == 0:
                copy_started = time.monotonic()
                report_copy = index == 0 or (index + 1) % 32 == 0 or index + 1 == len(parameters)
                if report_copy:
                    size_mib = local.numel() * 2 / 2**20
                    print(
                        f"checkpoint host copy: {index + 1}/{len(parameters)} "
                        f"{name} ({size_mib:.1f} MiB) start",
                        flush=True,
                    )
                gpu_tensor = local.detach().to(dtype=torch.bfloat16)
                host_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
                host_tensor.copy_(gpu_tensor, non_blocking=True)
                torch.cuda.synchronize()
                state[name] = host_tensor.contiguous()
                if report_copy:
                    print(
                        f"checkpoint host copy: {index + 1}/{len(parameters)} "
                        f"done in {time.monotonic() - copy_started:.2f}s",
                        flush=True,
                    )
        # Preserve state_dict's persistent buffers as well as parameters.
        for prefix, module in scorer.named_modules():
            for name, buffer in module.named_buffers(recurse=False):
                if name in module._non_persistent_buffers_set:
                    continue
                key = f"{prefix}.{name}" if prefix else name
                value = buffer.full_tensor() if hasattr(buffer, "full_tensor") else buffer
                if rank == 0:
                    state[key] = value.detach().cpu().contiguous()
        dist.barrier()
        return state
    finally:
        for unit in reversed(materialized):
            unit.reshard()


def export_checkpoint(
    scorer: Scorer,
    *,
    tokenizer,
    expected_shapes: dict[str, tuple[int, ...]],
    source: str,
    source_revision: str,
    target: Path,
    method: str,
    lora_scaling: float,
    rank: int,
    prompt_module: Path,
    drop_config_keys: list[str],
) -> None:
    """Gather full weights, verify a staged checkpoint, and publish HF shards."""
    if method == "full":
        state = gather_full_export_state(scorer, rank)
        state_dict_implementation = "layer-first-fsdp2"
    else:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state = get_model_state_dict(
            scorer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        state_dict_implementation = "distributed-state-dict"
    if rank == 0:
        state = (
            merge_lora_state_dict(state, lora_scaling)
            if method == "lora"
            else {key.removeprefix("inner."): value for key, value in state.items()}
        )
        state = {
            key: (
                value.to(dtype=torch.bfloat16, device="cpu").contiguous()
                if value.is_floating_point()
                else value.to(device="cpu").contiguous()
            )
            for key, value in state.items()
        }
        actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
        if actual_shapes != expected_shapes:
            missing = sorted(set(expected_shapes) - set(actual_shapes))
            unexpected = sorted(set(actual_shapes) - set(expected_shapes))
            bad_shapes = sorted(
                key
                for key in set(actual_shapes) & set(expected_shapes)
                if actual_shapes[key] != expected_shapes[key]
            )
            raise RuntimeError(
                "Export state does not match the loaded model; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"bad_shapes={bad_shapes[:10]}"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        scorer.inner.save_pretrained(
            staging,
            state_dict=state,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(staging)
        saved_shapes = safetensors_inventory(staging)
        if saved_shapes != expected_shapes:
            raise RuntimeError(
                "Saved checkpoint validation failed; "
                f"keys_match={set(saved_shapes) == set(expected_shapes)}"
            )
        configuration = json.loads((staging / "config.json").read_text())
        for key in drop_config_keys:
            configuration.pop(key, None)
        (staging / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
        shutil.copy2(prompt_module, staging / "decision_prompt.py")
        manifest = weight_manifest(staging)
        (staging / "SOURCE.json").write_text(
            json.dumps(
                {
                    "model": "decision-model",
                    "derived_from": source,
                    "source_revision": source_revision or None,
                    "method": method,
                    "merged_model_sha256": manifest["sha256"],
                    "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
                },
                indent=2,
            )
            + "\n"
        )
        (staging / "pure-pytorch-receipt.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "source_checkpoint": source,
                    "source_revision": source_revision or None,
                    "trainer": "scripts/decision_model/finetune_fsdp2.py",
                    "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
                    "fsdp2": True,
                    "state_dict_implementation": state_dict_implementation,
                    "tensor_count": len(state),
                    "model_bytes": manifest["bytes"],
                    "model_sha256": manifest["sha256"],
                    "index_sha256": manifest["index_sha256"],
                    "weight_files": manifest["files"],
                    "prompt_module": "decision_prompt.py",
                    "prompt_module_sha256": file_sha256(staging / "decision_prompt.py"),
                    "dropped_config_keys": drop_config_keys,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                indent=2,
            )
            + "\n"
        )
        checksums = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                checksums.append(f"{file_sha256(path)}  {path.relative_to(staging)}")
        (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    del state
    dist.barrier()


def _run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    os.environ.setdefault(
        "HF_MODULES_CACHE",
        f"/tmp/hf-modules-{job_id}-{local_rank}",
    )
    import transformers
    from transformers import AutoModel, AutoTokenizer

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Full-state BF16 conversion and checkpoint writes happen on rank 0
    # while the other ranks wait. Keep the process-group watchdog above the
    # default ten minutes so a healthy checkpoint write is not killed.
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    try:
        full_batch_microsteps = recipe_resolve_grad_accum(
            args.batch,
            args.per_device_batch_size,
            world,
        )
    except ValueError as error:
        message = str(error).replace("--gbs", "--batch")
        message = message.replace("per_device_train_batch_size", "--per-device-batch-size")
        raise SystemExit(message) from error

    def all_sum(value: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(value)
        return value

    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    def log(filename: str, record: dict) -> None:
        if rank == 0:
            with (output / filename).open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(filename.removesuffix(".jsonl"), json.dumps(record), flush=True)

    pretrained_kwargs = {"trust_remote_code": True}
    if args.model_revision:
        pretrained_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **pretrained_kwargs)
    prompt_module = resolve_prompt_module(args.checkpoint, args.prompt_module)
    prepare = DecisionRows(tokenizer, prompt_module, args.max_tokens)
    rows = read_jsonl(args.train)
    source_train_rows = len(rows)
    if args.train_limit:
        rows = rows[: args.train_limit]
    canonical_lengths = [0] * len(rows)
    usable = []
    for index, row in enumerate(rows):
        example = prepare(row)
        if example is not None:
            usable.append(index)
            canonical_lengths[index] = len(example["ids"]) + 1
    eval_rows = read_jsonl(args.eval) if args.eval else []
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
    validation = [example for example in map(prepare, eval_rows) if example is not None]
    if not usable:
        raise SystemExit("No usable training rows remain after prompt-length filtering")
    if len(validation) != len(eval_rows):
        raise SystemExit("Validation prompts exceed --max-tokens; full coverage is required")
    if rank == 0:
        print(
            f"usable rows: train {len(usable)}/{len(rows)}, "
            f"eval {len(validation)}/{len(eval_rows)}",
            flush=True,
        )

    load_dtype = torch.bfloat16 if args.method == "lora" else torch.float32
    model = AutoModel.from_pretrained(
        args.checkpoint,
        dtype=load_dtype,
        attn_implementation=args.attn_implementation,
        **pretrained_kwargs,
    )
    expected_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    mask_id = int(model.mask_token_id)
    enable_bidirectional(model)
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.method == "lora":
        modified = apply_lora(
            model,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            targets=set(args.targets.split(",")),
        )
    else:
        modified = mark_full_trainable(model)

    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if args.expected_lora_modules is not None and len(modified) != args.expected_lora_modules:
        raise RuntimeError(
            f"Expected {args.expected_lora_modules} LoRA modules, found {len(modified)}"
        )
    if (
        args.expected_trainable_parameters is not None
        and trainable_parameter_count != args.expected_trainable_parameters
    ):
        raise RuntimeError(
            "Trainable parameter count mismatch: expected "
            f"{args.expected_trainable_parameters}, found {trainable_parameter_count}"
        )

    scorer = shard_scorer(Scorer(model), cpu_offload=args.cpu_offload, device=device)
    fsdp2 = fsdp2_inventory(scorer)
    parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters remain after sharding")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    def option_logits(example: dict) -> torch.Tensor:
        return batched_option_logits(
            [example],
            scorer,
            mask_id=mask_id,
            device=device,
        )[0]

    def validate(step: int) -> dict | None:
        if not validation:
            return None
        scorer.eval()
        stats = torch.zeros(3, device=device, dtype=torch.float64)
        shard = validation[rank::world]
        forwards = math.ceil(len(validation) / world)
        with torch.no_grad():
            for index in range(forwards):
                example = shard[index] if index < len(shard) else validation[0]
                logits = option_logits(example)
                if index < len(shard):
                    stats += torch.tensor(
                        [
                            float(logits.argmax().item() == example["gold"]),
                            -logits.log_softmax(-1)[example["gold"]].item(),
                            1.0,
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
        all_sum(stats)
        scorer.train()
        record = {
            "step": step,
            "accuracy": (stats[0] / stats[2]).item(),
            "cross_entropy": (stats[1] / stats[2]).item(),
            "rows": int(stats[2].item()),
        }
        log("validation.jsonl", record)
        return record

    steps_per_epoch = math.ceil(len(usable) / args.batch)
    total_steps = args.max_steps or max(1, math.ceil(steps_per_epoch * args.epochs))
    stop_step = args.stop_step or total_steps
    if stop_step > total_steps:
        raise SystemExit(f"--stop-step {stop_step} exceeds total steps {total_steps}")
    signature_payload = [
        resume_signature(args, world, total_steps, prompt_module) if rank == 0 else None
    ]
    dist.broadcast_object_list(signature_payload, src=0)
    signature = signature_payload[0]
    start_step = 0
    resumed = {}
    if args.resume:
        resumed = read_resume_state(args.resume, signature)
        start_step = recipe_validate_resume_step(
            load_dcp_checkpoint(scorer, optimizer, args.resume, rank),
            total_steps,
        )
        if start_step != resumed["step"]:
            raise ValueError("DCP completed step differs from its training metadata")
        if resumed.get("best_saved"):
            if rank == 0:
                shutil.copytree(Path(args.resume) / "best", output / "best")
            dist.barrier()
    if start_step >= stop_step:
        raise SystemExit(f"Resume step {start_step} must be below this job's stop step {stop_step}")
    if start_step > total_steps:
        raise SystemExit(f"Resume step {start_step} exceeds total steps {total_steps}")
    config = {
        **vars(args),
        "world_size": world,
        "total_steps": total_steps,
        "start_step": start_step,
        "job_stop_step": stop_step,
        "source_train_rows": source_train_rows,
        "train_rows": len(rows),
        "usable_train_rows": len(usable),
        "gradient_accumulation_steps": full_batch_microsteps,
        "drop_last": False,
        "steps_per_epoch": steps_per_epoch,
        "eval_rows": len(validation),
        "modified_modules_or_parameters": len(modified),
        "trainable_parameters": trainable_parameter_count,
        "implementation": "fsdp2",
        "fsdp2_implementation": scorer._fsdp2_implementation,
        "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
        "container_image": os.environ.get("CONTAINER_IMAGE", "unknown"),
        "resume_signature": signature,
        **fsdp2,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print("config", json.dumps(config), flush=True)

    scorer.train()
    best = resumed.get("best")
    best_key = (-best["accuracy"], best["cross_entropy"]) if best else None
    best_step = resumed.get("best_step", start_step)
    initial = None if args.skip_initial_eval else validate(start_step)
    initial_key = (-initial["accuracy"], initial["cross_entropy"]) if initial else None
    initial_improved = initial_key is not None and (best_key is None or initial_key < best_key)
    if initial_improved:
        best = initial
        best_key = initial_key
        best_step = start_step
    if initial_improved and not args.no_save_best:
        export_checkpoint(
            scorer,
            tokenizer=tokenizer,
            expected_shapes=expected_shapes,
            source=args.checkpoint,
            source_revision=args.model_revision,
            target=output / "best",
            method=args.method,
            lora_scaling=args.alpha / args.rank,
            rank=rank,
            prompt_module=prompt_module,
            drop_config_keys=args.drop_config_key,
        )
    step = start_step
    epoch = step // steps_per_epoch
    cursor = (step % steps_per_epoch) * args.batch

    def epoch_order(epoch_index: int) -> list[int]:
        order = list(usable)
        if args.length_ordered:
            order.sort(key=lambda index: (-canonical_lengths[index], index))
        else:
            random.Random(args.seed + epoch_index).shuffle(order)
        return order

    order = epoch_order(epoch)
    audited_name, audited_tensor, audited_before = audit_parameter(scorer, args.method)

    while step < stop_step:
        started = time.monotonic()
        if cursor >= len(order):
            epoch += 1
            cursor = 0
            order = epoch_order(epoch)
        batch = order[cursor : cursor + args.batch]
        cursor += len(batch)
        valid_count = len(batch)
        microbatch_capacity = world * args.per_device_batch_size
        microsteps = (
            full_batch_microsteps
            if valid_count == args.batch
            else math.ceil(valid_count / microbatch_capacity)
        )
        padded_count = microsteps * microbatch_capacity
        padded = batch + [batch[-1]] * (padded_count - valid_count)

        optimizer.zero_grad(set_to_none=True)
        losses = []
        valid_flags = []
        for microstep in range(microsteps):
            local_start = microstep * microbatch_capacity + rank * args.per_device_batch_size
            local_slots = range(
                local_start,
                local_start + args.per_device_batch_size,
            )
            local_indices = [padded[slot] for slot in local_slots]
            local_valid = [slot < valid_count for slot in local_slots]
            examples = []
            for row_index in local_indices:
                shuffle_seed = row_index * 1_000_003 + args.seed + epoch
                example = prepare(
                    rows[row_index],
                    None
                    if random.Random(shuffle_seed).random() < args.canonical_fraction
                    else shuffle_seed,
                ) or prepare(rows[row_index])
                examples.append(example)

            synchronize = microstep == microsteps - 1
            scorer.set_requires_gradient_sync(synchronize)
            if hasattr(scorer, "set_is_last_backward"):
                scorer.set_is_last_backward(synchronize)
            logits = batched_option_logits(
                examples,
                scorer,
                mask_id=mask_id,
                device=device,
            )
            micro_losses = [
                row_loss(example, example_logits, args.kl_weight)
                for example, example_logits in zip(
                    examples,
                    logits,
                    strict=True,
                )
            ]
            scaled_loss = torch.stack(
                [
                    loss if valid else loss * 0
                    for loss, valid in zip(
                        micro_losses,
                        local_valid,
                        strict=True,
                    )
                ]
            ).sum() * (world / valid_count)
            scaled_loss.backward()
            losses.extend(loss.detach() for loss in micro_losses)
            valid_flags.extend(local_valid)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, args.max_grad_norm, error_if_nonfinite=True
        )
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        rate = learning_rate(step, total_steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        step += 1

        local_loss_sum = torch.stack(
            [
                loss if valid else torch.zeros_like(loss)
                for loss, valid in zip(losses, valid_flags, strict=True)
            ]
        ).sum()
        loss_stats = torch.stack(
            [
                local_loss_sum,
                local_loss_sum.new_tensor(sum(valid_flags)),
            ]
        )
        all_sum(loss_stats)
        mean_loss = loss_stats[0] / loss_stats[1]
        if step % args.log_every == 0 or step == total_steps:
            runtime_stats = torch.tensor(
                [
                    time.monotonic() - started,
                    torch.cuda.max_memory_allocated() / 2**30,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(runtime_stats, op=dist.ReduceOp.MAX)
            step_seconds = runtime_stats[0].item()
            log(
                "training.jsonl",
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": round(mean_loss.item(), 5),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": rate,
                    "seconds": round(step_seconds, 2),
                    "examples_per_second": round(valid_count / step_seconds, 3),
                    "peak_gib": round(runtime_stats[1].item(), 1),
                },
            )

        if step % args.eval_every == 0 or step == total_steps:
            current = validate(step)
            key = (-current["accuracy"], current["cross_entropy"]) if current is not None else None
            if key is not None and (best_key is None or key < best_key):
                best = current
                best_key = key
                best_step = step
                if not args.no_save_best and (step != total_steps or args.no_save_last):
                    export_checkpoint(
                        scorer,
                        tokenizer=tokenizer,
                        expected_shapes=expected_shapes,
                        source=args.checkpoint,
                        source_revision=args.model_revision,
                        target=output / "best",
                        method=args.method,
                        lora_scaling=args.alpha / args.rank,
                        rank=rank,
                        prompt_module=prompt_module,
                        drop_config_keys=args.drop_config_key,
                    )

    audited_delta = audited_parameter_delta(
        audited_tensor,
        audited_before,
        device,
    )
    if not math.isfinite(audited_delta):
        raise RuntimeError(
            f"Sampled parameter update is not finite: {audited_name} "
            f"(max_abs_delta={audited_delta})"
        )
    log(
        "runtime-audit.jsonl",
        {
            **fsdp2,
            "audited_parameter": audited_name,
            "max_abs_delta": audited_delta,
            "optimizer_steps": step,
        },
    )

    completed = step == total_steps
    if completed and not args.no_save_last:
        export_checkpoint(
            scorer,
            tokenizer=tokenizer,
            expected_shapes=expected_shapes,
            source=args.checkpoint,
            source_revision=args.model_revision,
            target=output / "last",
            method=args.method,
            lora_scaling=args.alpha / args.rank,
            rank=rank,
            prompt_module=prompt_module,
            drop_config_keys=args.drop_config_key,
        )
    if (
        completed
        and not args.no_save_best
        and not args.no_save_last
        and (best is None or best_step == total_steps)
    ):
        if rank == 0:
            if (output / "best").exists():
                shutil.rmtree(output / "best")
            shutil.copytree(output / "last", output / "best")
        dist.barrier()

    if args.save_dcp:
        best_checkpoint = output / "best"
        if args.no_save_best or not best_checkpoint.is_dir():
            best_checkpoint = None
        save_dcp_checkpoint(
            scorer,
            optimizer,
            step,
            args.save_dcp,
            rank,
            training_state={"signature": signature, "best": best, "best_step": best_step},
            best_checkpoint=best_checkpoint,
        )

    if rank == 0:
        status = {
            "start_step": start_step,
            "completed_step": step,
            "job_stop_step": stop_step,
            "total_steps": total_steps,
            "complete": completed,
        }
        (output / "segment.json").write_text(json.dumps(status, indent=2) + "\n")
        if completed:
            selected = {
                "best_step": best_step if best is not None else total_steps,
                "accuracy": -best_key[0] if best_key else None,
                "cross_entropy": best_key[1] if best_key else None,
            }
            (output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
            (output / "SUCCESS").write_text(
                f"completed_step={step}\ncode_commit={os.environ.get('CODE_COMMIT', 'unknown')}\n"
            )


def main(argv: list[str] | None = None) -> None:
    try:
        _run(argv)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
