# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read raw option scores through the Transformers decision interface."""

import random

import torch
from torch import nn
from torch.utils._pytree import tree_map_only

from scripts.decision_model.runtime import example_length, question_supervision
from scripts.decision_model.targets import question_targets


def load_decision_model(checkpoint, *, native, **kwargs):
    if not native:
        from transformers import AutoModel

        return AutoModel.from_pretrained(checkpoint, **kwargs)
    try:
        from transformers import AutoModelForDecision
    except ImportError as error:
        raise ImportError(
            "--native-decision requires Transformers with AutoModelForDecision"
        ) from error
    return AutoModelForDecision.from_pretrained(checkpoint, **kwargs)


class NativeRows:
    def __init__(self, model, tokenizer, max_tokens, packed_read=False):
        self.config = model.decision_config
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        if packed_read:
            if not hasattr(self.config, "decision_attention_layout"):
                raise ValueError(
                    "Checkpoint does not expose decision_attention_layout for --packed-read"
                )
            # Keep preparation, export, and subsequent serving on the same layout.
            self.config.decision_attention_layout = "state_sees_no_questions"

    def __call__(self, row, shuffle_seed=None):
        payload = row["input"]
        questions = payload["questions"]
        targets = question_targets(row, questions)
        multi = "targets" in row["reference"]
        soft = row.get("soft_labels")
        if multi and soft is not None:
            raise ValueError("soft_labels apply only to one-question rows with reference.target")
        if shuffle_seed is not None:
            shuffled = dict(questions)
            rng = random.Random(shuffle_seed)
            for key, question in questions.items():
                if question.get("type") == "choice":
                    items = list(question["criteria"].items())
                    rng.shuffle(items)
                    shuffled[key] = dict(question, criteria=dict(items))
            payload = dict(payload, questions=shuffled)

        seen = set()
        reads, ordered = [], []
        for inputs, prepared in self.config.prepare_inputs(payload, self.tokenizer):
            group = []
            for q in prepared:
                if q.name not in targets or q.name in seen:
                    raise ValueError("Prepared question keys must match the input questions")
                seen.add(q.name)
                supervision = question_supervision(q.options, q.type, targets[q.name], soft)
                group.append(dict(supervision, key=q.name))
            ids = inputs["input_ids"]
            if not group or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError(
                    "Each prepared read must contain one sequence and at least one question"
                )
            reads.append({"inputs": inputs, "questions": group})
            ordered.extend(group)
        if not reads or seen != set(questions):
            raise ValueError("Prepared question keys must match the input questions")
        example = {"reads": reads, "questions": ordered, "multi": multi}
        return example if example_length(example) <= self.max_tokens else None


class NativeScorer(nn.Module):
    """Keep the same FSDP2/export parameter namespace as the prompt-module scorer."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, **inputs):
        return self.inner(**inputs)


def inputs_to_device(inputs, device):
    return tree_map_only(torch.Tensor, lambda tensor: tensor.to(device), dict(inputs))


def read_logits(read, scorer, device):
    scores = scorer(**inputs_to_device(read["inputs"], device))
    sizes = [len(q["labels"]) for q in read["questions"]]
    if not isinstance(scores, torch.Tensor) or scores.shape != (sum(sizes),):
        raise ValueError("Native decision forward must return one flat score per option")
    return scores.float().split(sizes)
