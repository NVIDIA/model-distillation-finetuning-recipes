# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke test against a user-supplied local Transformers decision checkpoint."""

import json
import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.requires_model
@pytest.mark.parametrize(("method", "packed"), [("full", False), ("lora", True)])
def test_native_checkpoint_finetune_and_reload(tmp_path, method, packed):
    checkpoint = os.environ.get("DECISION_CHECKPOINT")
    if not checkpoint or not torch.cuda.is_available():
        pytest.skip("Set DECISION_CHECKPOINT and provide an NVIDIA GPU")
    world = min(2, torch.cuda.device_count())
    category = {
        "type": "choice",
        "instructions": "Choose the topic.",
        "criteria": {"billing": "Payment", "access": "Login"},
    }
    review = {"type": "noul", "instructions": "Does the payment need review?"}
    rows = []
    for index, state in enumerate(
        ["A payment was duplicated.", "A user cannot sign in.", "A card was charged twice."]
    ):
        questions = {"category": category}
        targets = {"category": "access" if index == 1 else "billing"}
        if index != 1:
            questions["review"] = review
            targets["review"] = True
        rows.append(
            {
                "id": str(index),
                "input": {"state": state, "questions": questions},
                "reference": {"targets": targets},
            }
        )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    validation.write_text(json.dumps(rows[2]) + "\n")
    run = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world}",
            "--module",
            "scripts.decision_model.finetune_fsdp2",
            "--native-decision",
            "--checkpoint",
            checkpoint,
            "--train",
            str(train),
            "--eval",
            str(validation),
            "--method",
            method,
            "--batch",
            "2",
            "--max-steps",
            "1",
            "--no-save-best",
            "--output",
            str(run),
            *(["--packed-read"] if packed else []),
        ],
        check=True,
    )
    audit = json.loads((run / "runtime-audit.jsonl").read_text())
    assert audit["optimizer_steps"] == 1 and audit["max_abs_delta"] > 0
    exported = run / "last"
    prefix = [sys.executable, "-m"]
    shared = ["--native-decision", "--checkpoint", str(exported)]
    subprocess.run(
        [
            *prefix,
            "scripts.decision_model.validate_checkpoint",
            *shared,
            "--data",
            str(validation),
            "--method",
            method,
        ],
        check=True,
    )
    predictions = tmp_path / "evaluation"
    predictions.mkdir()
    # Keep inference inside the GPU allocation inherited from the caller.
    subprocess.run(
        [
            *prefix,
            "scripts.decision_model.evaluate",
            "--worker",
            *shared,
            "--input",
            str(validation),
            "--output-dir",
            str(predictions),
        ],
        check=True,
    )
    result = json.loads((predictions / "predictions.jsonl").read_text())
    assert result["id"] == rows[2]["id"]
    assert set(result["questions"]) == set(rows[2]["reference"]["targets"])
    metadata = json.loads((predictions / "summary.json").read_text())
    assert metadata["failures"] == 0
    if packed:
        assert metadata["decision_attention_layout"] == "state_sees_no_questions"
