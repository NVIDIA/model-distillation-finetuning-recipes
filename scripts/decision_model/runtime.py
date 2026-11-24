# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt preparation and batched readout for decision models.

The model-supplied prompt module exposes candidate_codes(tokenizer) and
prepare_questions(tokenizer, payload, position_ordering=True, permutation=None).
DecisionRows validates the resulting labels, token IDs, and targets.

With packed_read=True the module also exposes state_end_token_id(tokenizer): the
token that closes the state in every prompt. A row's questions are then read in
one sequence, [state][question 1, mask][question 2, mask] ...: the state attends
only to the state, each question to the state and itself, question positions
restart after the state, and each question is read at its own mask token.
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
    """Validate decision rows and prepare model-supplied prompt records.

    By default each row holds exactly one question. With packed_read=True a row
    may hold several questions about one state (reference.targets maps each
    question key to its label) and prepares one packed example.
    """

    def __init__(
        self,
        tokenizer,
        prompt_module: str | Path,
        max_tokens: int,
        packed_read: bool = False,
    ):
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
        self.packed_read = packed_read
        self.state_end_id = None
        if packed_read:
            state_end = getattr(self.scoring, "state_end_token_id", None)
            if not callable(state_end):
                raise ValueError(
                    f"Prompt module {self.module_path} must define callable "
                    "state_end_token_id for packed reads"
                )
            token = state_end(tokenizer)
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("state_end_token_id must return a non-negative token ID")
            self.state_end_id = token

    def __call__(self, row: dict, shuffle_seed: int | None = None) -> dict | None:
        if self.packed_read:
            return self._packed(row, shuffle_seed)
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
        example = self._question(prepared[0], reference["target"], row.get("soft_labels"))
        if len(example["ids"]) + 1 > self.max_tokens:
            return None
        return example

    def _question(self, prepared, gold, soft) -> dict:
        """Validate one prepared question and its target."""
        if not isinstance(prepared, (list, tuple)) or len(prepared) != 5:
            raise ValueError("Prepared questions must contain key, kind, labels, descriptions, IDs")
        _, kind, labels, _descriptions, ids = prepared
        if not isinstance(labels, (list, tuple)) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError("Decision labels must be non-empty strings")
        labels = list(labels)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("Decision labels must contain at least two unique labels")
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
        return {
            "ids": ids,
            "gold": labels.index(gold),
            "soft": soft,
            "cand": candidates,
            "labels": labels,
        }

    def _packed(self, row: dict, shuffle_seed: int | None) -> dict | None:
        """Prepare all questions of one state as a packed example."""
        if not isinstance(row, dict):
            raise ValueError("Decision row must be an object")
        payload = row.get("input")
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Decision rows must contain at least one input question")
        reference = row.get("reference")
        if not isinstance(reference, dict):
            raise ValueError("Decision row lacks reference")
        soft = row.get("soft_labels")
        multi = not (len(questions) == 1 and "target" in reference)
        if not multi:
            targets = {next(iter(questions)): reference["target"]}
            softs = {next(iter(questions)): soft}
        else:
            targets = reference.get("targets")
            if not isinstance(targets, dict) or set(targets) != set(questions):
                raise ValueError("Packed rows need reference.targets for every question key")
            if soft is not None:
                raise ValueError("soft_labels apply only to one-question rows")
            softs = {}
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
        if not isinstance(prepared, (list, tuple)) or len(prepared) != len(questions):
            raise ValueError("Prompt module must return one prepared question per input question")
        keys = [item[0] if isinstance(item, (list, tuple)) and item else None for item in prepared]
        if sorted(map(str, keys)) != sorted(map(str, questions)) or len(set(keys)) != len(keys):
            raise ValueError("Prepared question keys must match the input questions")
        examples = [
            dict(self._question(item, targets[item[0]], softs.get(item[0])), key=item[0])
            for item in prepared
        ]
        state_length = packed_state_length(
            [example["ids"] for example in examples], self.state_end_id
        )
        state = examples[0]["ids"][:state_length]
        packed_questions = []
        for example in examples:
            packed_questions.append(dict(example, ids=example["ids"][state_length:]))
        packed = {"packed": True, "multi": multi, "state": state, "questions": packed_questions}
        if packed_length(packed) > self.max_tokens:
            return None
        return packed


def packed_options(packed_read: bool) -> dict:
    """DecisionRows keyword arguments; empty for single reads to keep that call unchanged."""
    return {"packed_read": True} if packed_read else {}


def packed_state_length(prompts: list[list[int]], state_end_id: int) -> int:
    """Length of the shared state: every prompt up to and including state_end_id."""
    lengths = set()
    for ids in prompts:
        if state_end_id not in ids:
            raise ValueError("Packed reads need the state-end token in every prompt")
        lengths.add(ids.index(state_end_id) + 1)
    if len(lengths) != 1:
        raise ValueError("Packed questions must share one state boundary")
    length = lengths.pop()
    if any(ids[:length] != prompts[0][:length] for ids in prompts[1:]):
        raise ValueError("Packed questions must share the state token IDs")
    return length


def packed_length(example: dict) -> int:
    """Token count of a packed example, including one mask token per question."""
    return len(example["state"]) + sum(
        len(question["ids"]) + 1 for question in example["questions"]
    )


def example_length(example: dict) -> int:
    """Token count read for a prepared example, including mask tokens."""
    return packed_length(example) if example.get("packed") else len(example["ids"]) + 1


def example_questions(example: dict) -> list[dict]:
    """Questions of a prepared example in readout order."""
    return example["questions"] if example.get("packed") else [example]


def packed_layout(example: dict, mask_id: int) -> dict:
    """One packed read: token IDs, positions, attention groups, and mask-token slots.

    groups[i] is -1 for state tokens and k for question k. Token i attends to
    token j when groups[j] is -1, or when both belong to the same question.
    """
    state = list(example["state"])
    ids = list(state)
    positions = list(range(len(state)))
    groups = [-1] * len(state)
    slots = []
    for index, question in enumerate(example["questions"]):
        block = list(question["ids"]) + [mask_id]
        ids += block
        positions += range(len(state), len(state) + len(block))
        groups += [index] * len(block)
        slots.append(len(ids) - 1)
    return {"ids": ids, "positions": positions, "groups": groups, "slots": slots}


def packed_attention(groups: torch.Tensor) -> torch.Tensor:
    """Boolean [batch, query, key] attention for packed groups; -2 marks padding."""
    state_key = groups[:, None, :] == -1
    same_question = (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] >= 0)
    return state_key | same_question


def collate_packed(
    examples: list[dict],
    *,
    mask_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad packed reads; return IDs, positions, attention, and readout rows/slots."""
    if not examples:
        raise ValueError("Cannot collate an empty example list")
    layouts = [packed_layout(example, mask_id) for example in examples]
    max_length = max(len(layout["ids"]) for layout in layouts)
    input_ids = torch.full((len(layouts), max_length), mask_id, dtype=torch.long)
    position_ids = torch.zeros((len(layouts), max_length), dtype=torch.long)
    groups = torch.full((len(layouts), max_length), -2, dtype=torch.long)
    rows, slots = [], []
    for index, layout in enumerate(layouts):
        length = len(layout["ids"])
        input_ids[index, :length] = torch.tensor(layout["ids"])
        position_ids[index, :length] = torch.tensor(layout["positions"])
        groups[index, :length] = torch.tensor(layout["groups"])
        rows += [index] * len(layout["slots"])
        slots += layout["slots"]
    return (
        input_ids.to(device),
        position_ids.to(device),
        packed_attention(groups).to(device),
        torch.tensor(rows, device=device, dtype=torch.long),
        torch.tensor(slots, device=device, dtype=torch.long),
    )


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
        position_ids: torch.Tensor | None = None,
        allowed: torch.Tensor | None = None,
        readout_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return vocabulary logits at the readout positions.

        Packed reads pass position_ids, a boolean [batch, query, key] allowed
        matrix, and readout_rows naming the batch row of each readout position.
        """
        encoder = self.inner.encoder
        hidden = encoder.embed_tokens(input_ids)
        sequence_length = input_ids.shape[1]
        cache_position = torch.arange(sequence_length, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        additive_mask = None
        if allowed is not None:
            additive_mask = torch.zeros(
                (input_ids.shape[0], 1, sequence_length, sequence_length),
                device=input_ids.device,
                dtype=hidden.dtype,
            )
            additive_mask.masked_fill_(~allowed[:, None].bool(), torch.finfo(hidden.dtype).min)
        elif attention_mask is not None:
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
        batch_indices = (
            readout_rows
            if readout_rows is not None
            else torch.arange(input_ids.shape[0], device=input_ids.device)
        )
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


def packed_option_logits(
    examples: list[dict],
    scorer: Scorer,
    *,
    mask_id: int,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    """Option-code logits of every question of each packed example."""
    input_ids, position_ids, allowed, rows, slots = collate_packed(
        examples, mask_id=mask_id, device=device
    )
    vocabulary_logits = scorer(
        input_ids,
        readout_positions=slots,
        position_ids=position_ids,
        allowed=allowed,
        readout_rows=rows,
    )
    output, offset = [], 0
    for example in examples:
        questions = example["questions"]
        output.append(
            [
                vocabulary_logits[offset + index, question["cand"]]
                for index, question in enumerate(questions)
            ]
        )
        offset += len(questions)
    return output


def example_option_logits(
    examples: list[dict],
    scorer: Scorer,
    *,
    mask_id: int,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    """Per-question option-code logits for single or packed examples."""
    if all(example.get("packed") for example in examples):
        return packed_option_logits(examples, scorer, mask_id=mask_id, device=device)
    if any(example.get("packed") for example in examples):
        raise ValueError("Cannot mix packed and single-read examples in one batch")
    logits = batched_option_logits(examples, scorer, mask_id=mask_id, device=device)
    return [[value] for value in logits]
