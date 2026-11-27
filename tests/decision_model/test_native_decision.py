# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from scripts.decision_model import finetune_fsdp2 as training
from scripts.decision_model.evaluate import predict_row
from scripts.decision_model.native import NativeRows, NativeScorer, read_logits
from scripts.decision_model.score import score_questions
from scripts.decision_model.split_packed import row_labels
from scripts.decision_model.validate_checkpoint import validate_native_row


class DecisionConfig:
    decision_attention_layout = "state_sees_paired_question"

    def prepare_inputs(self, request, tokenizer):
        assert set(request) == {"state", "questions", "assets"}
        questions = []
        for key, raw in reversed(request["questions"].items()):
            if raw["type"] == "score":
                options = [str(i) for i in range(len(raw["criteria"]))]
            elif raw["type"] == "noul":
                options = ["false", "true"]
            else:
                options = list(raw["criteria"])
            q = SimpleNamespace(
                name=key, type=raw["type"], options=options, labels=list("ABC")[: len(options)]
            )
            questions.append(q)
        groups = (
            [questions]
            if self.decision_attention_layout == "state_sees_no_questions"
            else [[q] for q in questions]
        )
        for group in groups:
            sizes = [len(q.options) for q in group]
            yield (
                {"input_ids": torch.tensor([[*range(sum(sizes))]]), "extra": {"sizes": sizes}},
                group,
            )


class DecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(12, 4)
        block = nn.Module()
        block.q_proj = nn.Linear(4, 4)
        self.model.layers = nn.ModuleList([block])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 1)
        self.decision_config = DecisionConfig()

    def forward(self, input_ids, extra):
        assert input_ids.numel() == sum(extra["sizes"])
        x = self.model.embed_tokens(input_ids)
        x = self.model.layers[0].q_proj(x)
        return self.lm_head(self.model.norm(x)).flatten()


def sample():
    return {
        "id": "sample",
        "input": {
            "state": "A parcel arrived.",
            "assets": {},
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {"local": "a", "regional": "b", "other": "c"},
                },
                "priority": {"type": "score", "criteria": ["low", "high"]},
                "delivered": {"type": "noul"},
            },
        },
        "reference": {"targets": {"category": "regional", "priority": 1, "delivered": True}},
    }


def single_sample():
    row = sample()
    row["input"]["questions"] = {"category": row["input"]["questions"]["category"]}
    row["reference"] = {"target": "regional"}
    row["soft_labels"] = {"other": 6, "regional": 2}
    return row


@pytest.mark.parametrize("packed", [False, True])
def test_native_semantic_options_and_predictions(packed):
    model = DecisionModel()
    row = sample()
    original = deepcopy(row)
    prepare = NativeRows(model, object(), 7, packed)
    example = prepare(row)
    assert len(example["reads"]) == (1 if packed else 3)
    assert training.example_length(example) == (7 if packed else 3)
    assert [(q["key"], q["labels"][q["gold"]]) for q in example["questions"]] == [
        ("delivered", "true"),
        ("priority", "1"),
        ("category", "regional"),
    ]
    shorter = NativeRows(model, object(), training.example_length(example) - 1)
    assert shorter(row) is None
    question = prepare(single_sample(), shuffle_seed=1)["questions"][0]
    assert question["labels"] != ["local", "regional", "other"]
    assert dict(zip(question["labels"], question["soft"])) == {
        "other": 0.75,
        "regional": 0.25,
        "local": 0,
    }
    shuffled = prepare(row, shuffle_seed=1)
    assert shuffled["questions"][1]["labels"] == ["0", "1"]
    assert row == original
    with torch.no_grad():
        prediction = predict_row("sample", example, model, mask_id=None, device="cpu")
        for read in example["reads"]:
            expected = model(**read["inputs"])
            for q, values in zip(
                read["questions"],
                expected.split([len(q["labels"]) for q in read["questions"]]),
                strict=True,
            ):
                assert prediction["questions"][q["key"]]["probs"] == dict(
                    zip(q["labels"], values.softmax(-1).tolist())
                )
    assert score_questions([row], {"sample": prediction})["scored_questions"] == 3
    assert set(row_labels(row)) == {
        ("delivered", "true"),
        ("priority", "1"),
        ("category", "regional"),
    }
    result = validate_native_row(
        model,
        object(),
        row,
        SimpleNamespace(packed_read=packed, max_tokens=20, atol=0, rtol=0),
        "cpu",
    )
    assert all(q["max_logit_difference"] == 0 for q in result.values())


def test_native_rejects_missing_questions_and_wrong_score_shape():
    model = DecisionModel()
    row = sample()
    reads = list(model.decision_config.prepare_inputs(row["input"], None))
    prepare = NativeRows(model, None, 20)
    for invalid in (reads[:-1], reads + reads[:1]):
        model.decision_config.prepare_inputs = lambda *_, result=invalid: result
        with pytest.raises(ValueError, match="question keys"):
            prepare(row)
    read = {"inputs": reads[0][0], "questions": [{"labels": ["no", "yes"]}]}
    for scores in (torch.ones(1, 2), torch.ones(3)):
        with pytest.raises(ValueError, match="flat score"):
            read_logits(read, lambda result=scores, **_: result, "cpu")


@pytest.mark.parametrize("method", ["full", "lora"])
def test_native_backprop_updates_only_selected_parameters(method):
    model = DecisionModel()
    if method == "lora":
        assert training.apply_lora(model, rank=2, alpha=4, dropout=0, targets={"q_proj"}) == [
            "model.layers.0.q_proj"
        ]
    else:
        assert set(training.mark_full_trainable(model)) == {
            "model.layers.0.q_proj.weight",
            "model.layers.0.q_proj.bias",
            "model.norm.weight",
            "model.norm.bias",
        }
    scorer = NativeScorer(model)
    sync = []
    scorer.set_requires_gradient_sync = sync.append
    prepare = NativeRows(model, None, 20)
    rows = [prepare(sample()), prepare(single_sample())]
    loss = training.native_backward(rows, [True, False], scorer, "cpu", 1, 1, 0, True)
    assert sync == [False, False, False, True]
    assert loss[0] > 0 and loss[1] == 0
    assert (
        not model.model.embed_tokens.weight.requires_grad and not model.lm_head.weight.requires_grad
    )
    assert all(
        p.grad is not None and p.grad.isfinite().all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert any(p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def distributed_backward(rank, rendezvous):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=40)
    )
    try:
        torch.manual_seed(42)
        model = DecisionModel()
        reference = deepcopy(model)
        prepare = NativeRows(model, None, 20)
        rows = [prepare(sample()), prepare(single_sample())]
        scorer = nn.parallel.DistributedDataParallel(model)
        sync = []

        def set_sync(flag):
            sync.append(flag)
            scorer.require_backward_grad_sync = flag

        scorer.set_requires_gradient_sync = set_sync
        training.native_backward([rows[rank]], [True], scorer, "cpu", 2, 2, 0, True)
        assert sync == [False, False, True]
        for row in rows:
            losses = [
                training.row_loss(q, values)
                for read in row["reads"]
                for q, values in zip(
                    read["questions"], read_logits(read, reference, "cpu"), strict=True
                )
            ]
            (torch.stack(losses).mean() / 2).backward()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
    finally:
        dist.destroy_process_group()


def test_uneven_native_reads_match_single_process_gradients(tmp_path):
    mp.spawn(distributed_backward, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
