# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random

import pytest
import torch
from torch import nn

from scripts.decision_model import finetune_fsdp2 as TRAINER


class Attention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.diffusion_lm = False
        self.is_causal = True


class MLP(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2, bias=False)
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.norm = nn.LayerNorm(8)


class DecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.diffusion_head = nn.Linear(8, 32, bias=False)


class MaskAwareLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **_kwargs):
        if attention_mask is None:
            valid = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            valid = (attention_mask[:, 0, 0] == 0).to(hidden_states.dtype)
        context = (hidden_states * valid.unsqueeze(-1)).sum(1)
        context = context / valid.sum(1, keepdim=True)
        return hidden_states + context.unsqueeze(1)


class MaskAwareEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([MaskAwareLayer(), MaskAwareLayer()])
        self.norm = nn.LayerNorm(8)

    def rotary_emb(self, hidden, position_ids):
        shape = (*position_ids.shape, hidden.shape[-1])
        zeros = torch.zeros(shape, device=hidden.device, dtype=hidden.dtype)
        return zeros, zeros


class MaskAwareDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MaskAwareEncoder()
        self.diffusion_head = nn.Linear(8, 32, bias=False)


def test_native_lora_freezes_base_and_targets_all_projections():
    model = DecisionModel()
    original = model.encoder.layers[0].self_attn.q_proj.weight.detach().clone()
    names = TRAINER.apply_lora(
        model,
        rank=2,
        alpha=4,
        dropout=0,
        targets=set(TRAINER.LORA_TARGETS),
    )

    assert len(names) == 2 * len(TRAINER.LORA_TARGETS)
    assert isinstance(model.encoder.layers[0].self_attn.q_proj, TRAINER.LoRALinear)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    assert torch.equal(original, model.encoder.layers[0].self_attn.q_proj.base.weight)


def test_lora_merge_reconstructs_standard_linear_weight():
    state = {
        "inner.encoder.layers.0.self_attn.q_proj.base.weight": torch.zeros(3, 4),
        "inner.encoder.layers.0.self_attn.q_proj.lora_A": torch.ones(2, 4),
        "inner.encoder.layers.0.self_attn.q_proj.lora_B": torch.ones(3, 2),
        "inner.encoder.norm.weight": torch.ones(4),
    }
    merged = TRAINER.merge_lora_state_dict(state, scaling=0.5)

    assert set(merged) == {
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.norm.weight",
    }
    assert torch.equal(
        merged["encoder.layers.0.self_attn.q_proj.weight"].float(),
        torch.ones(3, 4),
    )


def test_lora_forward_and_adapter_gradients():
    base = nn.Linear(4, 3, bias=False)
    base.weight.data.zero_()
    layer = TRAINER.LoRALinear(base, rank=2, alpha=4, dropout=0)
    layer.lora_A.data.fill_(1)
    layer.lora_B.data.fill_(0.5)
    inputs = torch.ones(2, 4)

    output = layer(inputs)
    assert torch.equal(output, torch.full((2, 3), 8.0))
    output.sum().backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert base.weight.grad is None


def test_batched_padding_matches_independent_noncausal_forwards():
    torch.manual_seed(7)
    scorer = TRAINER.Scorer(MaskAwareDecisionModel())
    examples = [
        {"ids": [1, 2], "cand": [3, 4], "gold": 0, "soft": None},
        {"ids": [5, 6, 7, 8], "cand": [9, 10], "gold": 1, "soft": None},
        {"ids": [11], "cand": [12, 13], "gold": 0, "soft": None},
    ]

    batched = TRAINER.batched_option_logits(
        examples,
        scorer,
        mask_id=31,
        device=torch.device("cpu"),
    )
    independent = [
        TRAINER.batched_option_logits(
            [example],
            scorer,
            mask_id=31,
            device=torch.device("cpu"),
        )[0]
        for example in examples
    ]

    assert all(
        torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        for actual, expected in zip(batched, independent, strict=True)
    )


def test_collate_examples_uses_right_padding_and_mask_readouts():
    examples = [
        {"ids": [1, 2], "cand": [3]},
        {"ids": [4], "cand": [5]},
    ]
    input_ids, attention_mask, readout_positions = TRAINER.collate_examples(
        examples,
        mask_id=9,
        device=torch.device("cpu"),
    )

    assert torch.equal(input_ids, torch.tensor([[1, 2, 9], [4, 9, 9]]))
    assert torch.equal(
        attention_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    assert torch.equal(readout_positions, torch.tensor([2, 1]))


def test_full_finetune_freezes_embeddings_and_diffusion_head():
    model = DecisionModel()
    names = TRAINER.mark_full_trainable(model)

    assert names
    assert not model.encoder.embed_tokens.weight.requires_grad
    assert model.encoder.layers[0].self_attn.q_proj.weight.requires_grad
    assert model.encoder.norm.weight.requires_grad
    assert not model.diffusion_head.weight.requires_grad


def test_bidirectional_flag_and_reference_defaults():
    model = DecisionModel()
    TRAINER.enable_bidirectional(model)
    assert all(not layer.self_attn.diffusion_lm for layer in model.encoder.layers)
    assert all(not layer.self_attn.is_causal for layer in model.encoder.layers)

    lora = TRAINER.parse_args(
        [
            "--method",
            "lora",
            "--checkpoint",
            "checkpoint",
            "--train",
            "train.jsonl",
            "--output",
            "out",
        ]
    )
    full = TRAINER.parse_args(
        [
            "--method",
            "full",
            "--checkpoint",
            "checkpoint",
            "--train",
            "train.jsonl",
            "--output",
            "out",
        ]
    )
    assert (lora.lr, lora.batch, lora.rank, lora.alpha) == (3e-5, 16, 32, 64)
    assert (full.lr, full.batch) == (1e-6, 64)
    assert lora.per_device_batch_size == full.per_device_batch_size == 1
    assert (lora.adam_beta1, lora.adam_beta2, lora.adam_eps) == (
        0.9,
        0.95,
        1e-8,
    )
    assert lora.weight_decay == 0.0
    assert lora.model_revision == ""


def test_adamw_hyperparameters_are_configurable():
    args = TRAINER.parse_args(
        [
            "--method",
            "full",
            "--checkpoint",
            "checkpoint",
            "--train",
            "train.jsonl",
            "--output",
            "out",
            "--lr",
            "2e-6",
            "--weight-decay",
            "0.01",
            "--adam-beta1",
            "0.8",
            "--adam-beta2",
            "0.9",
            "--adam-eps",
            "1e-7",
        ]
    )

    assert args.lr == 2e-6
    assert args.weight_decay == 0.01
    assert (args.adam_beta1, args.adam_beta2, args.adam_eps) == (0.8, 0.9, 1e-7)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--lr", "0"),
        ("--batch", "0"),
        ("--per-device-batch-size", "0"),
        ("--epochs", "0"),
        ("--warmup", "1.1"),
        ("--weight-decay", "-1"),
        ("--adam-beta1", "1"),
        ("--adam-beta2", "-0.1"),
        ("--adam-eps", "0"),
        ("--max-grad-norm", "-1"),
        ("--eval-every", "0"),
        ("--log-every", "0"),
        ("--targets", ",,"),
    ],
)
def test_invalid_training_arguments_fail_early(option, value):
    with pytest.raises(SystemExit):
        TRAINER.parse_args(
            [
                "--method",
                "lora",
                "--checkpoint",
                "checkpoint",
                "--train",
                "train.jsonl",
                "--output",
                "out",
                option,
                value,
            ]
        )


def test_schedule_reaches_peak_and_final_floor():
    total = 11_874
    warm_steps = int(0.05 * total)
    peak = 3e-5
    assert TRAINER.learning_rate(warm_steps - 1, total, peak, 0.05) == peak
    assert TRAINER.learning_rate(total - 1, total, peak, 0.05) > 0.1 * peak
    assert TRAINER.learning_rate(total, total, peak, 0.05) == 0.1 * peak


def test_rng_state_round_trip():
    random.seed(123)
    torch.manual_seed(456)
    state = TRAINER.capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(4)

    random.seed(999)
    torch.manual_seed(999)
    TRAINER.restore_rng_state(state)

    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)


def test_sharded_weight_manifest_and_inventory(tmp_path):
    from safetensors.torch import save_file

    save_file({"a": torch.ones(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b": torch.zeros(3, 4)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        )
    )

    manifest = TRAINER.weight_manifest(tmp_path)
    assert len(manifest["files"]) == 2
    assert manifest["bytes"] > 0
    assert len(manifest["sha256"]) == 64
    assert TRAINER.safetensors_inventory(tmp_path) == {"a": (2,), "b": (3, 4)}


def test_optional_kl_loss_combines_cross_entropy_and_target_divergence():
    example = {"gold": 0, "cand": [2, 3], "soft": [0.75, 0.25]}
    logits = torch.tensor([1.0, -1.0])
    log_probs = logits.log_softmax(-1)
    soft = torch.tensor(example["soft"])
    expected = -log_probs[0] + 0.2 * (soft * (soft.log() - log_probs)).sum()
    assert torch.allclose(TRAINER.row_loss(example, logits, kl_weight=0.2), expected)
