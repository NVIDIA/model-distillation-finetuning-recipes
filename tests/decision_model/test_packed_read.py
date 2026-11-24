# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.decision_model import finetune_fsdp2 as TRAINER
from scripts.decision_model.runtime import (
    DecisionRows,
    Scorer,
    batched_option_logits,
    collate_packed,
    example_length,
    example_option_logits,
    load_prompt_module,
    packed_attention,
    packed_layout,
    packed_option_logits,
    packed_state_length,
)

STATE_END = 101
MASK = 100

# Expected values from the reference packer of an always-packed decision model
# for three prompts that share the state [1, 2, 3, <state end>].
GOLDEN_PROMPTS = [
    [1, 2, 3, STATE_END, 4, 5],
    [1, 2, 3, STATE_END, 6],
    [1, 2, 3, STATE_END, 7, 8, 9],
]
GOLDEN = {
    "state_len": 4,
    "ids": [1, 2, 3, 101, 4, 5, 100, 6, 100, 7, 8, 9, 100],
    "positions": [0, 1, 2, 3, 4, 5, 6, 4, 5, 4, 5, 6, 7],
    "groups": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2],
    "slots": [6, 8, 12],
    "allowed_pairs": 81,
}

PROMPT_SOURCE = """
STATE_END = 101

def candidate_codes(tokenizer):
    return [("A", 10), ("B", 11), ("C", 12)]

def state_end_token_id(tokenizer):
    return STATE_END

def prepare_questions(tokenizer, payload, position_ordering=True, permutation=None):
    prepared = []
    for key, question in payload["questions"].items():
        labels = list(question["criteria"])
        if permutation is not None:
            labels = [labels[index] for index in permutation(key, len(labels))]
        suffix = tokenizer.questions[key]
        prepared.append((key, question["type"], labels, labels, tokenizer.state + suffix))
    return prepared
"""


def prompt_file(tmp_path, source=PROMPT_SOURCE):
    path = tmp_path / "decision_prompt.py"
    path.write_text(source + f"\n# fixture: {tmp_path}\n", encoding="utf-8")
    return path


def tokenizer(state=(1, 2, 3, STATE_END)):
    return SimpleNamespace(
        state=list(state),
        questions={"first": [4, 5], "second": [6], "third": [7, 8, 9]},
    )


def question(*labels):
    return {"type": "choice", "instructions": "Pick one.", "criteria": {x: x for x in labels}}


def packed_row(keys=("first", "second", "third")):
    labels = {"first": ("x", "y"), "second": ("x", "y", "z"), "third": ("y", "z")}
    return {
        "id": "row",
        "input": {"state": "s", "questions": {key: question(*labels[key]) for key in keys}},
        "reference": {"targets": {key: labels[key][-1] for key in keys}},
    }


def test_packed_layout_matches_reference_golden():
    state_length = packed_state_length(GOLDEN_PROMPTS, STATE_END)
    example = {
        "state": GOLDEN_PROMPTS[0][:state_length],
        "questions": [{"ids": prompt[state_length:]} for prompt in GOLDEN_PROMPTS],
    }
    layout = packed_layout(example, MASK)
    allowed = packed_attention(torch.tensor([layout["groups"]]))[0]

    assert state_length == GOLDEN["state_len"]
    for key in ("ids", "positions", "groups", "slots"):
        assert layout[key] == GOLDEN[key]
    assert int(allowed.sum()) == GOLDEN["allowed_pairs"]
    # State attends to state only; each question to the state and itself.
    groups = layout["groups"]
    for query, query_group in enumerate(groups):
        for key, key_group in enumerate(groups):
            expected = key_group == -1 or (query_group >= 0 and key_group == query_group)
            assert bool(allowed[query, key]) is expected


def test_packed_rows_prepare_shared_state_and_question_blocks(tmp_path):
    rows = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=64, packed_read=True)
    example = rows(packed_row())

    assert example["packed"] is True
    assert example["state"] == [1, 2, 3, STATE_END]
    assert [q["ids"] for q in example["questions"]] == [[4, 5], [6], [7, 8, 9]]
    assert [q["key"] for q in example["questions"]] == ["first", "second", "third"]
    assert [q["labels"][q["gold"]] for q in example["questions"]] == ["y", "z", "z"]
    assert [q["cand"] for q in example["questions"]] == [[10, 11], [10, 11, 12], [10, 11]]
    assert example_length(example) == len(GOLDEN["ids"])
    assert (
        DecisionRows(tokenizer(), prompt_file(tmp_path), 12, packed_read=True)(packed_row()) is None
    )


def test_one_question_rows_keep_target_and_soft_labels_when_packed(tmp_path):
    row = packed_row(("second",))
    row["reference"] = {"target": "y"}
    row["soft_labels"] = {"x": 1, "y": 3}
    example = DecisionRows(tokenizer(), prompt_file(tmp_path), 64, packed_read=True)(row)

    (only,) = example["questions"]
    assert example["state"] == [1, 2, 3, STATE_END]
    assert only["ids"] == [6]
    assert only["labels"][only["gold"]] == "y"
    assert only["soft"] == pytest.approx([0.25, 0.75, 0.0])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda row: row["reference"].pop("targets"), "reference.targets"),
        (lambda row: row["reference"]["targets"].pop("third"), "reference.targets"),
        (lambda row: row.update(soft_labels={"x": 1}), "one-question rows"),
        (lambda row: row["input"].update(questions={}), "at least one"),
    ],
)
def test_packed_rows_reject_incomplete_targets(tmp_path, change, message):
    row = packed_row()
    change(row)
    rows = DecisionRows(tokenizer(), prompt_file(tmp_path), 64, packed_read=True)
    with pytest.raises(ValueError, match=message):
        rows(row)


def test_packed_reads_need_a_shared_state_boundary(tmp_path):
    with pytest.raises(ValueError, match="state-end token"):
        DecisionRows(tokenizer((1, 2, 3)), prompt_file(tmp_path), 64, packed_read=True)(
            packed_row()
        )
    with pytest.raises(ValueError, match="share one state boundary"):
        packed_state_length([[1, STATE_END, 2], [1, 1, STATE_END, 2]], STATE_END)
    with pytest.raises(ValueError, match="share the state token IDs"):
        packed_state_length([[1, STATE_END, 2], [2, STATE_END, 2]], STATE_END)
    missing = PROMPT_SOURCE.replace("def state_end_token_id", "def unused_state_end")
    with pytest.raises(ValueError, match="state_end_token_id"):
        DecisionRows(tokenizer(), prompt_file(tmp_path, missing), 64, packed_read=True)


def test_single_read_rejects_multi_question_rows_as_before(tmp_path):
    with pytest.raises(ValueError, match="exactly one input question"):
        DecisionRows(tokenizer(), prompt_file(tmp_path), 64)(packed_row())


class GroupAttentionLayer(nn.Module):
    """Single-head attention that honors additive masks and rotary positions."""

    def __init__(self, width=8):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **_kwargs):
        cos, _ = position_embeddings
        states = hidden_states + cos
        scores = self.query(states) @ self.key(states).transpose(-1, -2) / math.sqrt(8)
        if attention_mask is not None:
            scores = scores + attention_mask[:, 0]
        return hidden_states + scores.softmax(-1) @ self.value(states)


class GroupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, 8)
        self.layers = nn.ModuleList([GroupAttentionLayer(), GroupAttentionLayer()])
        self.norm = nn.LayerNorm(8)

    def rotary_emb(self, hidden, position_ids):
        angles = position_ids[..., None].to(hidden.dtype) * torch.arange(1, 9, dtype=hidden.dtype)
        return angles.cos(), angles.sin()


class GroupDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GroupEncoder()
        self.diffusion_head = nn.Linear(8, 128, bias=False)


def packed_example(prompts, cands):
    state_length = packed_state_length(prompts, STATE_END)
    return {
        "packed": True,
        "state": prompts[0][:state_length],
        "questions": [
            {"ids": prompt[state_length:], "cand": cand, "gold": 0, "soft": None}
            for prompt, cand in zip(prompts, cands, strict=True)
        ],
    }


def test_packed_question_logits_do_not_depend_on_other_questions():
    torch.manual_seed(3)
    scorer = Scorer(GroupDecisionModel()).eval()
    cands = [[10, 11], [12, 13, 14], [15, 16]]
    together = packed_option_logits(
        [packed_example(GOLDEN_PROMPTS, cands)], scorer, mask_id=MASK, device=torch.device("cpu")
    )[0]
    for index, prompt in enumerate(GOLDEN_PROMPTS):
        (alone,) = packed_option_logits(
            [packed_example([prompt], [cands[index]])],
            scorer,
            mask_id=MASK,
            device=torch.device("cpu"),
        )[0]
        torch.testing.assert_close(together[index], alone, atol=1e-6, rtol=1e-6)


def test_packed_read_differs_from_single_read_only_through_state_attention():
    torch.manual_seed(5)
    scorer = Scorer(GroupDecisionModel()).eval()
    prompt = GOLDEN_PROMPTS[2]
    (packed,) = packed_option_logits(
        [packed_example([prompt], [[10, 11]])], scorer, mask_id=MASK, device=torch.device("cpu")
    )[0]
    single = batched_option_logits(
        [{"ids": prompt, "cand": [10, 11]}], scorer, mask_id=MASK, device=torch.device("cpu")
    )[0]
    # Same tokens and positions: with full attention the packed read equals the single read.
    input_ids, position_ids, allowed, rows, slots = collate_packed(
        [packed_example([prompt], [[10, 11]])], mask_id=MASK, device=torch.device("cpu")
    )
    full = scorer(
        input_ids,
        readout_positions=slots,
        position_ids=position_ids,
        allowed=torch.ones_like(allowed),
        readout_rows=rows,
    )[0, [10, 11]]
    torch.testing.assert_close(full, single, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(packed, single)


def test_padded_packed_batches_match_independent_reads():
    torch.manual_seed(11)
    scorer = Scorer(GroupDecisionModel()).eval()
    examples = [
        packed_example(GOLDEN_PROMPTS, [[10, 11], [12, 13], [14, 15]]),
        packed_example([[5, STATE_END, 6]], [[20, 21, 22]]),
    ]
    batched = packed_option_logits(examples, scorer, mask_id=MASK, device=torch.device("cpu"))
    for example, logits in zip(examples, batched, strict=True):
        alone = packed_option_logits([example], scorer, mask_id=MASK, device=torch.device("cpu"))
        for actual, expected in zip(logits, alone[0], strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_single_read_paths_are_unchanged_without_packed_read():
    torch.manual_seed(13)
    scorer = Scorer(GroupDecisionModel()).eval()
    examples = [{"ids": [1, 2], "cand": [3, 4]}, {"ids": [5, 6, 7], "cand": [8, 9]}]
    direct = batched_option_logits(examples, scorer, mask_id=MASK, device=torch.device("cpu"))
    wrapped = example_option_logits(examples, scorer, mask_id=MASK, device=torch.device("cpu"))
    assert all(torch.equal(a, b[0]) for a, b in zip(direct, wrapped, strict=True))
    assert example_length(examples[1]) == 4
    with pytest.raises(ValueError, match="Cannot mix"):
        example_option_logits(
            [examples[0], packed_example(GOLDEN_PROMPTS, [[1, 2]] * 3)],
            scorer,
            mask_id=MASK,
            device=torch.device("cpu"),
        )


def test_packed_row_loss_is_the_mean_of_question_losses():
    first = {"gold": 0, "soft": None, "cand": [1, 2]}
    second = {"gold": 1, "soft": None, "cand": [1, 2, 3]}
    logits = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]
    loss = TRAINER.example_loss({"packed": True, "questions": [first, second]}, logits)
    expected = (TRAINER.row_loss(first, logits[0]) + TRAINER.row_loss(second, logits[1])) / 2
    assert torch.isclose(loss, expected)
    assert torch.equal(TRAINER.example_loss(first, logits[:1]), TRAINER.row_loss(first, logits[0]))


def test_packed_read_flag_is_opt_in_and_recorded_in_resume_signature(tmp_path):
    base = ["--method", "lora", "--checkpoint", "m", "--train", "t", "--output", "o"]
    assert TRAINER.parse_args(base).packed_read is False
    assert TRAINER.parse_args([*base, "--packed-read"]).packed_read is True
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n")
    module = prompt_file(tmp_path)
    args = TRAINER.parse_args(
        ["--method", "lora", "--checkpoint", "org/model", "--model-revision", "abc"]
        + ["--train", str(train), "--output", "o"]
    )
    assert "packed_read" not in TRAINER.resume_signature(args, 1, 1, module)["settings"]
    args.packed_read = True
    assert TRAINER.resume_signature(args, 1, 1, module)["settings"]["packed_read"] is True


class WordTokenizer:
    def __init__(self):
        self.vocab = {"<SEP>": 101}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in text.replace("<SEP>", " <SEP> ").split():
            ids.append(self.vocab.setdefault(part, 200 + len(self.vocab)))
        return ids


def test_example_prompt_module_adds_state_end_only_when_configured():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples/decision_model/prompt_module.py"
    module = load_prompt_module(path)
    payload = {
        "state": "Example Corp acquired Sample Labs.",
        "questions": {"event": question("acquisition", "other")},
    }
    tokenizer = WordTokenizer()
    plain = module.prepare_questions(tokenizer, payload)[0][4]
    assert module.STATE_END_TOKEN is None
    with pytest.raises(ValueError, match="STATE_END_TOKEN"):
        module.state_end_token_id(tokenizer)
    module.STATE_END_TOKEN = "<SEP>"
    try:
        separated = module.prepare_questions(tokenizer, payload)[0][4]
        state_end = module.state_end_token_id(tokenizer)
    finally:
        module.STATE_END_TOKEN = None
    assert state_end == 101
    assert STATE_END not in plain
    assert [token for token in separated if token != state_end] == plain
