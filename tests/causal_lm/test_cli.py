# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.causal_lm import distill_fsdp2, lora_fsdp2


def test_distillation_defaults_to_kd_only():
    args = distill_fsdp2.get_args(
        [
            "--student_path",
            "student",
            "--teacher_path",
            "teacher",
            "--data_path",
            "data.jsonl",
            "--output_dir",
            "output",
            "--max_steps",
            "1",
        ]
    )
    assert args.use_lm_loss is False
    assert args.blended_loss_weight is None
    assert args.attn_implementation == "sdpa"
    assert args.seed == 1234


def test_lora_accepts_model_specific_launch_options():
    args = lora_fsdp2.get_args(
        [
            "--model_path",
            "model",
            "--data_path",
            "data.jsonl",
            "--output_dir",
            "output",
            "--max_steps",
            "1",
            "--attn_implementation",
            "sdpa",
            "--lora_target_modules",
            "in_proj",
            "out_proj",
            "--dcp_resume_path",
            "checkpoint",
        ]
    )
    assert args.attn_implementation == "sdpa"
    assert args.lora_target_modules == ["in_proj", "out_proj"]
    assert args.dcp_resume_path == "checkpoint"
    assert args.seed == 1234


@pytest.mark.parametrize("module", [distill_fsdp2, lora_fsdp2])
def test_cli_rejects_invalid_blended_loss_weight(module):
    required = [
        "--data_path",
        "data.jsonl",
        "--output_dir",
        "output",
        "--max_steps",
        "1",
        "--blended_loss_weight",
        "1.5",
    ]
    if module is distill_fsdp2:
        required = [
            "--student_path",
            "student",
            "--teacher_path",
            "teacher",
            *required,
        ]
    else:
        required = ["--model_path", "model", *required]

    with pytest.raises(SystemExit):
        module.get_args(required)


@pytest.mark.parametrize("module", [distill_fsdp2, lora_fsdp2])
def test_cli_rejects_non_positive_save_steps(module):
    required = [
        "--data_path",
        "data.jsonl",
        "--output_dir",
        "output",
        "--max_steps",
        "1",
        "--save_steps",
        "0",
    ]
    if module is distill_fsdp2:
        required = [
            "--student_path",
            "student",
            "--teacher_path",
            "teacher",
            *required,
        ]
    else:
        required = ["--model_path", "model", *required]

    with pytest.raises(SystemExit):
        module.get_args(required)


@pytest.mark.parametrize(
    ("module", "retired_flag"),
    [
        (distill_fsdp2, "--lam"),
        (distill_fsdp2, "--skip_lm_loss"),
        (distill_fsdp2, "--intermediate_loss"),
        (lora_fsdp2, "--lam"),
    ],
)
def test_cli_rejects_retired_flags(module, retired_flag):
    required = [
        "--data_path",
        "data.jsonl",
        "--output_dir",
        "output",
        "--max_steps",
        "1",
        retired_flag,
    ]
    if module is distill_fsdp2:
        required = [
            "--student_path",
            "student",
            "--teacher_path",
            "teacher",
            *required,
        ]
    else:
        required = ["--model_path", "model", *required]

    with pytest.raises(SystemExit):
        module.get_args(required)


@pytest.mark.parametrize("module", [distill_fsdp2, lora_fsdp2])
def test_cli_rejects_abbreviated_flags(module):
    required = [
        "--data_path",
        "data.jsonl",
        "--output_dir",
        "output",
        "--max_steps",
        "1",
        "--attn",
        "sdpa",
    ]
    if module is distill_fsdp2:
        required = [
            "--student_path",
            "student",
            "--teacher_path",
            "teacher",
            *required,
        ]
    else:
        required = ["--model_path", "model", *required]

    with pytest.raises(SystemExit):
        module.get_args(required)


@pytest.mark.parametrize("module", [distill_fsdp2, lora_fsdp2])
@pytest.mark.parametrize("flag", ["--max_steps", "--logging_steps"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_step_counts(module, flag, value):
    model_args = (
        ["--student_path", "student", "--teacher_path", "teacher"]
        if module is distill_fsdp2
        else ["--model_path", "model"]
    )
    with pytest.raises(SystemExit):
        module.get_args(
            [
                *model_args,
                "--data_path",
                "data.jsonl",
                "--output_dir",
                "output",
                "--max_steps",
                "1",
                flag,
                value,
            ]
        )


@pytest.mark.parametrize("temperature", ["0", "-1", "nan", "inf", "-inf"])
def test_distillation_rejects_invalid_temperature(temperature):
    with pytest.raises(SystemExit):
        distill_fsdp2.get_args(
            [
                "--student_path",
                "student",
                "--teacher_path",
                "teacher",
                "--data_path",
                "data.jsonl",
                "--output_dir",
                "output",
                "--max_steps",
                "1",
                f"--kd_temperature={temperature}",
            ]
        )
