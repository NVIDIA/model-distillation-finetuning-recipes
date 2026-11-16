# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt preparation and batched readout for decision models.

The model-supplied prompt module exposes candidate_codes(tokenizer) and
prepare_questions(tokenizer, payload, position_ordering=True, permutation=None).
DecisionRows validates the resulting labels, token IDs, and targets.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import random
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


def resolve_prompt_module(checkpoint: str | Path, prompt_module: str | Path | None = None) -> Path:
    """Resolve an explicit prompt module or the module shipped with a snapshot."""
    path = Path(prompt_module) if prompt_module else Path(checkpoint) / "decision_prompt.py"
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(
            f"Missing decision prompt module: {path}. Pass --prompt-module with a reviewed "
            "self-contained Python module supplied with the model."
        )
    return path.resolve()


def load_prompt_module(path: str | Path) -> ModuleType:
    """Load model-specific prompt code using its source hash as the cache key."""
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(f"Decision prompt module is not a Python file: {path}")
    source = path.read_bytes()
    name = "_decision_prompt_" + hashlib.sha256(source).hexdigest()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load decision prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute the reviewed bytes used as the cache key, bypassing stale bytecode.
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
        for attribute in ("candidate_codes", "prepare_questions"):
            if not callable(getattr(module, attribute, None)):
                raise ValueError(f"Prompt module {path} must define callable {attribute}")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class DecisionRows:
    """Validate one-question rows and prepare model-supplied prompt records."""

    def __init__(self, tokenizer, prompt_module: str | Path, max_tokens: int):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.module_path = Path(prompt_module).resolve()
        self.scoring = load_prompt_module(self.module_path)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.codes = self.scoring.candidate_codes(tokenizer)
        if not isinstance(self.codes, (list, tuple)) or any(
            not isinstance(code, (list, tuple)) or len(code) != 2 for code in self.codes
        ):
            raise ValueError("candidate_codes must return (code, token_id) pairs")

    def __call__(self, row: dict, shuffle_seed: int | None = None) -> dict | None:
        if not isinstance(row, dict):
            raise ValueError("Decision row must be an object")
        payload = row.get("input")
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("Decision rows must contain exactly one input question")
        reference = row.get("reference")
        if not isinstance(reference, dict) or "target" not in reference:
            raise ValueError("Decision row lacks reference.target")
        permutation = None
        if shuffle_seed is not None:

            def permutation(_key: str, size: int) -> list[int]:
                order = list(range(size))
                random.Random(shuffle_seed).shuffle(order)
                return order

        prepared = self.scoring.prepare_questions(
            self.tokenizer,
            dict(payload, model="decision-model"),
            position_ordering=True,
            permutation=permutation,
        )
        if not isinstance(prepared, (list, tuple)) or len(prepared) != 1:
            raise ValueError("Prompt module must return exactly one prepared question")
        if not isinstance(prepared[0], (list, tuple)) or len(prepared[0]) != 5:
            raise ValueError("Prepared questions must contain key, kind, labels, descriptions, IDs")
        _, kind, labels, _descriptions, ids = prepared[0]
        if not isinstance(labels, (list, tuple)) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError("Decision labels must be non-empty strings")
        labels = list(labels)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("Decision labels must contain at least two unique labels")
        gold = reference["target"]
        gold = str(gold).lower() if kind == "noul" else str(gold)
        if gold not in labels:
            raise ValueError(f"Unknown decision target: {gold!r}")
        if (
            not isinstance(ids, (list, tuple))
            or not ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids
            )
        ):
            raise ValueError("Prompt module must return non-empty non-negative token IDs")
        ids = list(ids)
        if len(self.codes) < len(labels):
            raise ValueError("Prompt module does not provide enough option-code tokens")
        candidates = [token for _, token in self.codes[: len(labels)]]
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in candidates
        ):
            raise ValueError("Option-code token IDs must be non-negative integers")
        if len(set(candidates)) != len(candidates):
            raise ValueError("Option-code token IDs must be unique")
        soft = row.get("soft_labels")
        if soft is not None:
            if not isinstance(soft, dict) or set(soft) - set(labels):
                raise ValueError("soft_labels must map only known decision labels to weights")
            try:
                weights = [float(soft.get(label, 0.0)) for label in labels]
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("soft_labels weights must be numeric") from error
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("soft_labels weights must be finite and non-negative")
            scale = max(weights)
            if scale <= 0:
                raise ValueError("soft_labels must have positive total weight")
            scaled = [weight / scale for weight in weights]
            total = math.fsum(scaled)
            soft = [weight / total for weight in scaled]
        if len(ids) + 1 > self.max_tokens:
            return None
        return {
            "ids": ids,
            "gold": labels.index(gold),
            "soft": soft,
            "cand": candidates,
            "labels": labels,
        }


def enable_bidirectional(model: nn.Module) -> None:
    """Configure noncausal attention while retaining support for padding masks."""
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        raise ValueError("Checkpoint does not expose encoder.layers")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if (
            attention is None
            or not hasattr(attention, "diffusion_lm")
            or not hasattr(attention, "is_causal")
        ):
            raise ValueError(
                f"{type(layer).__name__} does not expose bidirectional attention flags"
            )
        # The checkpoint's diffusion_lm branch forces attention_mask=None.
        # Use the standard branch with causality disabled so right-padding can
        # be masked without changing the single-example noncausal computation.
        attention.diffusion_lm = False
        attention.is_causal = False


class Scorer(nn.Module):
    """Batched prompt IDs to vocabulary logits at each trailing mask token."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        readout_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder = self.inner.encoder
        hidden = encoder.embed_tokens(input_ids)
        sequence_length = input_ids.shape[1]
        cache_position = torch.arange(sequence_length, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)

        additive_mask = None
        if attention_mask is not None:
            additive_mask = torch.zeros(
                (input_ids.shape[0], 1, 1, sequence_length),
                device=input_ids.device,
                dtype=hidden.dtype,
            )
            additive_mask.masked_fill_(
                ~attention_mask[:, None, None, :].bool(),
                torch.finfo(hidden.dtype).min,
            )

        position_embeddings = encoder.rotary_emb(hidden, position_ids=position_ids)
        for layer in encoder.layers:
            hidden = layer(
                hidden,
                attention_mask=additive_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_training=False,
            )
        hidden = encoder.norm(hidden)

        if readout_positions is None:
            readout_positions = torch.full(
                (input_ids.shape[0],),
                sequence_length - 1,
                device=input_ids.device,
                dtype=torch.long,
            )
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        logits = self.inner.diffusion_head(hidden[batch_indices, readout_positions]).float()
        return logits.clone()


def collate_examples(
    examples: list[dict],
    *,
    mask_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Right-pad decision-model prompts and return their mask-token readout positions."""
    if not examples:
        raise ValueError("Cannot collate an empty example list")
    sequences = [example["ids"] + [mask_id] for example in examples]
    length_values = [len(sequence) for sequence in sequences]
    max_length = max(length_values)
    input_ids_cpu = torch.full(
        (len(sequences), max_length),
        mask_id,
        dtype=torch.long,
    )
    for index, sequence in enumerate(sequences):
        input_ids_cpu[index, : len(sequence)] = torch.tensor(sequence)

    input_ids = input_ids_cpu.to(device)
    lengths = torch.tensor(length_values, device=device, dtype=torch.long)
    attention_mask = None
    if any(length != max_length for length in length_values):
        positions = torch.arange(max_length, device=device).unsqueeze(0)
        attention_mask = positions < lengths.unsqueeze(1)
    return input_ids, attention_mask, lengths - 1


def batched_option_logits(
    examples: list[dict],
    scorer: Scorer,
    *,
    mask_id: int,
    device: torch.device,
) -> list[torch.Tensor]:
    input_ids, attention_mask, readout_positions = collate_examples(
        examples,
        mask_id=mask_id,
        device=device,
    )
    vocabulary_logits = scorer(input_ids, attention_mask, readout_positions)
    return [vocabulary_logits[index, example["cand"]] for index, example in enumerate(examples)]
