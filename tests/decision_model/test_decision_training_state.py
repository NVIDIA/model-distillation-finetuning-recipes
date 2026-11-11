# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume integrity and checkpoint-selection regressions without model downloads."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from scripts.decision_model import finetune_fsdp2 as training


@pytest.fixture
def training_inputs(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    save_file({"weight": torch.ones(2)}, str(checkpoint / "model.safetensors"))
    (checkpoint / "config.json").write_text('{"model_type": "synthetic"}\n')
    (checkpoint / "tokenizer.json").write_text('{"vocabulary": ["first", "second"]}\n')
    (checkpoint / "vocab.txt").write_text("first\nsecond\n")
    (checkpoint / "model_code").mkdir()
    (checkpoint / "model_code/layers.py").write_text("DIMENSION = 2\n")
    prompt = checkpoint / "decision_prompt.py"
    prompt.write_text('"""Synthetic prompt module for CPU regression tests."""\n')
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps({"id": f"train-{index}"}) + "\n" for index in range(3)))
    validation = tmp_path / "validation.jsonl"
    validation.write_text('{"id": "validation"}\n')
    return SimpleNamespace(checkpoint=checkpoint, prompt=prompt, train=train, validation=validation)


def command(inputs, output, method="lora"):
    return [
        "--method",
        method,
        "--checkpoint",
        str(inputs.checkpoint),
        "--train",
        str(inputs.train),
        "--eval",
        str(inputs.validation),
        "--output",
        str(output),
        "--batch",
        "2",
        "--per-device-batch-size",
        "2",
        "--max-steps",
        "2",
        "--eval-every",
        "1",
        "--lr",
        "0.01",
        "--rank",
        "2",
        "--alpha",
        "4",
        "--skip-initial-eval",
        "--no-grad-checkpointing",
    ]


def signature(inputs, args):
    return training.resume_signature(args, world=1, total_steps=2, prompt_module=inputs.prompt)


@pytest.mark.parametrize(
    "change",
    [
        "data",
        "tokenizer",
        "vocabulary",
        "model_code",
        "weights",
        "prompt",
        "seed",
        "world",
        "eval_every",
    ],
)
def test_resume_signature_detects_continuity_changes(training_inputs, tmp_path, change):
    inputs = training_inputs
    args = training.parse_args(command(inputs, tmp_path / "run"))
    expected = signature(inputs, args)
    world = 1
    if change == "data":
        inputs.train.write_text(inputs.train.read_text() + '{"id": "additional"}\n')
    elif change == "tokenizer":
        (inputs.checkpoint / "tokenizer.json").write_text('{"vocabulary": ["changed"]}\n')
    elif change == "vocabulary":
        (inputs.checkpoint / "vocab.txt").write_text("first\nchanged\n")
    elif change == "model_code":
        (inputs.checkpoint / "model_code/layers.py").write_text("DIMENSION = 4\n")
    elif change == "weights":
        save_file({"weight": torch.zeros(2)}, str(inputs.checkpoint / "model.safetensors"))
    elif change == "prompt":
        inputs.prompt.write_text(inputs.prompt.read_text() + "# changed prompt\n")
    elif change == "seed":
        args.seed += 1
    elif change == "world":
        world = 2
    elif change == "eval_every":
        args.eval_every += 1
    actual = training.resume_signature(args, world, 2, inputs.prompt)
    assert actual != expected


def test_resume_signature_allows_new_segment_paths(training_inputs, tmp_path):
    inputs = training_inputs
    args = training.parse_args(command(inputs, tmp_path / "first"))
    expected = signature(inputs, args)
    relocated = tmp_path / "same-model-in-another-location"
    shutil.copytree(inputs.checkpoint, relocated)
    args.checkpoint = str(relocated)
    args.output = str(tmp_path / "second")
    args.resume = str(tmp_path / "step_1")
    args.stop_step = 2
    assert signature(inputs, args) == expected


def write_training_state(directory, expected, **overrides):
    directory.mkdir(exist_ok=True)
    state = {
        "signature": expected,
        "step": 1,
        "best": None,
        "best_step": 0,
        "best_saved": False,
        **overrides,
    }
    (directory / "decision-training-state.json").write_text(json.dumps(state))
    return state


def test_resume_requires_configuration_metadata(tmp_path):
    with pytest.raises(ValueError, match="configuration metadata"):
        training.read_resume_state(tmp_path, {"expected": "signature"})


@pytest.mark.parametrize("step", [None, True, -1, "1", 1.5])
def test_resume_rejects_invalid_completed_step(tmp_path, step):
    expected = {"expected": "signature"}
    write_training_state(tmp_path, expected, step=step)
    with pytest.raises(ValueError, match="invalid completed step"):
        training.read_resume_state(tmp_path, expected)


def test_resume_rejects_changed_inputs(tmp_path):
    expected = {"train_sha256": "original", "settings": {"seed": 101}}
    write_training_state(tmp_path, expected)
    changed = copy.deepcopy(expected)
    changed["settings"]["seed"] = 102
    with pytest.raises(ValueError, match="differ from the original"):
        training.read_resume_state(tmp_path, changed)


def test_resume_checks_preserved_best_weights(tmp_path):
    expected = {"expected": "signature"}
    state = write_training_state(
        tmp_path,
        expected,
        best_saved=True,
        best={"step": 1, "accuracy": 1.0, "cross_entropy": 0.1, "rows": 1},
        best_step=1,
    )
    best = tmp_path / "best"
    best.mkdir()
    weights = best / "model.safetensors"
    save_file({"weight": torch.ones(2)}, str(weights))
    (best / "pure-pytorch-receipt.json").write_text(
        json.dumps(
            {
                "model_sha256": training.weight_manifest(best)["sha256"],
            }
        )
    )
    assert training.read_resume_state(tmp_path, expected) == state
    save_file({"weight": torch.zeros(2)}, str(weights))
    with pytest.raises(ValueError, match="best checkpoint does not match"):
        training.read_resume_state(tmp_path, expected)


@pytest.mark.parametrize("asset", ["index", "prompt"])
def test_resume_checks_recorded_best_index_and_prompt(tmp_path, asset):
    expected = {"expected": "signature"}
    state = write_training_state(
        tmp_path,
        expected,
        best_saved=True,
        best={"step": 1, "accuracy": 1.0, "cross_entropy": 0.1, "rows": 1},
        best_step=1,
    )
    best = tmp_path / "best"
    best.mkdir()
    save_file({"weight": torch.ones(2)}, str(best / "model.safetensors"))
    index = best / "model.safetensors.index.json"
    index_payload = {"weight_map": {"weight": "model.safetensors"}}
    index.write_text(json.dumps(index_payload))
    prompt = best / "decision_prompt.py"
    prompt.write_text('"""Reviewed synthetic prompt."""\n')
    manifest = training.weight_manifest(best)
    (best / "pure-pytorch-receipt.json").write_text(
        json.dumps(
            {
                "model_sha256": manifest["sha256"],
                "index_sha256": manifest["index_sha256"],
                "prompt_module_sha256": training.file_sha256(prompt),
            }
        )
    )
    assert training.read_resume_state(tmp_path, expected) == state
    if asset == "index":
        index_payload["metadata"] = {"changed": True}
        index.write_text(json.dumps(index_payload))
        error = "index does not match"
    else:
        prompt.write_text(prompt.read_text() + "# changed prompt\n")
        error = "prompt module does not match"
    # The weight shards are untouched: the extra receipt digests must detect
    # these changes independently of the aggregate weight digest.
    assert training.weight_manifest(best)["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match=error):
        training.read_resume_state(tmp_path, expected)


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(8, 2)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.diffusion_lm = True
        layer.self_attn.is_causal = True
        for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, target, nn.Linear(2, 2, bias=False))
        layer.mlp = nn.Module()
        for target in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, target, nn.Linear(2, 2, bias=False))
        self.encoder.layers = nn.ModuleList([layer])
        self.encoder.norm = nn.LayerNorm(2)
        self.diffusion_head = nn.Linear(2, 8, bias=False)
        self.mask_token_id = 7


class SyntheticRows:
    def __init__(self, tokenizer, prompt_module, max_tokens):
        self.module_path = Path(prompt_module)

    def __call__(self, row, shuffle_seed=None):
        return {"ids": [1], "cand": [0, 1], "gold": 0, "soft": None}


class CPUTorch:
    """Keep trainer device allocations on CPU without patching torch globally."""

    cuda = SimpleNamespace(
        set_device=lambda *_: None, max_memory_allocated=lambda: 0, is_available=lambda: False
    )

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(*_args, **_kwargs):
        return torch.device("cpu")


class SyntheticFSDPUnit(nn.Module):
    def __init__(self, name, events):
        super().__init__()
        self.unit_name = name
        self.events = events
        self.materialized = False
        self.reshard_calls = 0

    def unshard(self):
        self.materialized = True
        self.events.append(("unshard", self.unit_name))

    def reshard(self):
        self.materialized = False
        self.reshard_calls += 1
        self.events.append(("reshard", self.unit_name))


class HostCopyFailureTensor(torch.Tensor):
    def copy_(self, source, non_blocking=False):
        raise RuntimeError("injected host-copy failure")


@pytest.mark.parametrize("fail_copy", [False, True])
def test_full_export_preserves_buffers_and_reshards_on_host_copy_failure(monkeypatch, fail_copy):
    from torch.distributed import fsdp

    events = []
    scorer = SyntheticFSDPUnit("root", events)
    scorer.inner = nn.Module()
    scorer.inner.encoder = nn.Module()
    layers = [SyntheticFSDPUnit(f"layer-{index}", events) for index in range(2)]
    for index, layer in enumerate(layers):
        layer.weight = nn.Parameter(torch.full((2,), index + 0.5))
    scorer.inner.encoder.layers = nn.ModuleList(layers)
    scorer.register_buffer("root_counter", torch.tensor([3], dtype=torch.int64))
    layers[0].register_buffer("scale", torch.tensor([0.5]))
    layers[1].register_buffer("cache", torch.tensor([99.0]), persistent=False)
    expected = {name: value.clone() for name, value in scorer.state_dict().items()}
    units = [*layers, scorer]
    allocations = []

    class ExportTorch(CPUTorch):
        cuda = SimpleNamespace(synchronize=lambda: None)

        @staticmethod
        def empty_like(value, **kwargs):
            kwargs.pop("pin_memory", None)
            host = torch.empty_like(value, **kwargs)
            allocations.append(host)
            if fail_copy and len(allocations) == 2:
                return host.as_subclass(HostCopyFailureTensor)
            return host

    monkeypatch.setattr(fsdp, "FSDPModule", SyntheticFSDPUnit)
    monkeypatch.setattr(training, "torch", ExportTorch())
    monkeypatch.setattr(training, "dist", SimpleNamespace(barrier=lambda: None))
    if fail_copy:
        with pytest.raises(RuntimeError, match="injected host-copy failure"):
            training.gather_full_export_state(scorer, rank=0)
    else:
        state = training.gather_full_export_state(scorer, rank=0)
        assert set(state) == set(expected)
        assert not any(name.endswith("cache") for name in state)
        for name, value in expected.items():
            torch.testing.assert_close(state[name], value, check_dtype=False)
        assert state["root_counter"].dtype == torch.int64
        assert state["inner.encoder.layers.0.scale"].dtype == torch.float32
    assert len(allocations) == 2
    assert events == [
        ("unshard", "layer-0"),
        ("unshard", "layer-1"),
        ("unshard", "root"),
        ("reshard", "root"),
        ("reshard", "layer-1"),
        ("reshard", "layer-0"),
    ]
    assert all(not unit.materialized and unit.reshard_calls == 1 for unit in units)


@pytest.fixture
def cpu_training(monkeypatch):
    import transformers

    model = SyntheticModel()
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *_a, **_k: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: object())
    monkeypatch.setattr(training, "torch", CPUTorch())
    monkeypatch.setattr(training, "DecisionRows", SyntheticRows)
    model.distributed_calls = []
    distributed_state = {"initialized": False}

    def initialize(*_args, **_kwargs):
        distributed_state["initialized"] = True
        model.distributed_calls.append("initialize")

    def destroy():
        distributed_state["initialized"] = False
        model.distributed_calls.append("destroy")

    monkeypatch.setattr(
        training,
        "dist",
        SimpleNamespace(
            init_process_group=initialize,
            is_initialized=lambda: distributed_state["initialized"],
            destroy_process_group=destroy,
            barrier=lambda: None,
            all_reduce=lambda *_a, **_k: None,
            broadcast_object_list=lambda *_a, **_k: None,
            ReduceOp=torch.distributed.ReduceOp,
        ),
    )
    for key in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def shard(scorer, **_kwargs):
        scorer._fsdp2_implementation = "synthetic-cpu"
        scorer.set_requires_gradient_sync = lambda _synchronize: None
        return scorer

    monkeypatch.setattr(training, "shard_scorer", shard)
    monkeypatch.setattr(training, "fsdp2_inventory", lambda _s: {"fsdp2_units": 2})
    calls = {"validation": 0}

    def logits(examples, scorer, **_kwargs):
        if scorer.training:
            value = (
                sum(parameter.sum() for parameter in scorer.parameters() if parameter.requires_grad)
                * 0.001
            )
            return [torch.stack((value, value.new_zeros(()))) for _ in examples]
        calls["validation"] += 1
        values = [4.0, -4.0] if calls["validation"] == 1 else [-4.0, 4.0]
        return [torch.tensor(values) for _ in examples]

    monkeypatch.setattr(training, "batched_option_logits", logits)

    def export(_scorer, *, target, **_kwargs):
        records = (target.parent / "training.jsonl").read_text().splitlines()
        step = json.loads(records[-1])["step"]
        target.mkdir(exist_ok=True)
        (target / "exported-step.txt").write_text(str(step))

    monkeypatch.setattr(training, "export_checkpoint", export)
    return model


def test_main_cleans_up_after_batch_configuration_failure(training_inputs, cpu_training, tmp_path):
    argv = command(training_inputs, tmp_path / "invalid-batch")
    argv[argv.index("--batch") + 1] = "3"

    with pytest.raises(SystemExit) as caught:
        training.main(argv)

    message = str(caught.value)
    assert "--batch 3" in message
    assert "--per-device-batch-size" in message
    assert "--gbs" not in message
    assert "per_device_train_batch_size" not in message
    assert cpu_training.distributed_calls == ["initialize", "destroy"]


@pytest.mark.parametrize("argv", [[], ["--help"]])
def test_main_does_not_destroy_an_uninitialized_group(cpu_training, argv):
    with pytest.raises(SystemExit):
        training.main(argv)

    assert cpu_training.distributed_calls == []


@pytest.mark.parametrize("method", ["lora", "full"])
def test_skipped_initial_eval_preserves_earlier_best(
    training_inputs,
    cpu_training,
    tmp_path,
    method,
):
    output = tmp_path / method
    training.main(command(training_inputs, output, method))
    assert cpu_training.distributed_calls == ["initialize", "destroy"]
    assert (output / "best/exported-step.txt").read_text() == "1"
    assert (output / "last/exported-step.txt").read_text() == "2"
    selected = json.loads((output / "selected.json").read_text())
    assert selected["best_step"] == 1
    assert selected["accuracy"] == 1.0
    audit = json.loads((output / "runtime-audit.jsonl").read_text())
    assert audit["max_abs_delta"] > 0
    assert audit["optimizer_steps"] == 2


@pytest.mark.parametrize("method", ["lora", "full"])
def test_unchanged_sample_does_not_reject_successful_training(
    training_inputs,
    cpu_training,
    tmp_path,
    monkeypatch,
    method,
):
    monkeypatch.setattr(training, "audited_parameter_delta", lambda *_args: 0.0)
    output = tmp_path / f"unchanged-sample-{method}"
    training.main(command(training_inputs, output, method))

    records = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert any(record["grad_norm"] > 0 for record in records)
    audit = json.loads((output / "runtime-audit.jsonl").read_text())
    assert audit["max_abs_delta"] == 0.0
    assert audit["optimizer_steps"] == 2
    assert (output / "SUCCESS").is_file()


def test_main_rejects_resume_mismatch_before_loading_dcp(
    training_inputs,
    cpu_training,
    tmp_path,
    monkeypatch,
):
    argv = command(training_inputs, tmp_path / "resumed")
    args = training.parse_args(argv)
    expected = signature(training_inputs, args)
    changed = copy.deepcopy(expected)
    changed["settings"]["seed"] += 1
    checkpoint = tmp_path / "step_1"
    write_training_state(checkpoint, changed)
    load_dcp = Mock(side_effect=AssertionError("DCP must not be loaded"))
    monkeypatch.setattr(training, "load_dcp_checkpoint", load_dcp)
    with pytest.raises(ValueError, match="differ from the original"):
        training.main([*argv, "--resume", str(checkpoint)])
    load_dcp.assert_not_called()


@pytest.mark.parametrize("method", ["lora", "full"])
def test_resumed_run_preserves_best_from_previous_segment(
    training_inputs,
    cpu_training,
    tmp_path,
    monkeypatch,
    method,
):
    output = tmp_path / f"resumed-{method}"
    argv = command(training_inputs, output, method)
    expected = signature(training_inputs, training.parse_args(argv))
    checkpoint = tmp_path / "step_1"
    write_training_state(
        checkpoint,
        expected,
        best_saved=True,
        best_step=0,
        best={"step": 0, "accuracy": 1.0, "cross_entropy": 1e-6, "rows": 1},
    )
    best = checkpoint / "best"
    best.mkdir()
    save_file({"weight": torch.ones(2)}, str(best / "model.safetensors"))
    (best / "pure-pytorch-receipt.json").write_text(
        json.dumps(
            {
                "model_sha256": training.weight_manifest(best)["sha256"],
            }
        )
    )
    (best / "exported-step.txt").write_text("0")
    monkeypatch.setattr(training, "load_dcp_checkpoint", lambda *_a, **_k: 1)
    training.main([*argv, "--resume", str(checkpoint)])
    assert (output / "best/exported-step.txt").read_text() == "0"
    assert (output / "last/exported-step.txt").read_text() == "2"
    selected = json.loads((output / "selected.json").read_text())
    assert selected["best_step"] == 0
    assert selected["cross_entropy"] == 1e-6
