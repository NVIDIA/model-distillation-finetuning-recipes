# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.decision_model import finetune_fsdp2 as TRAINER


def prepare_row(row):
    if row.get("failure"):
        raise ValueError(row["failure"])
    if row.get("invalid"):
        return None
    return {"ids": list(range(row["length"] - 1))}


def digest(values):
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("rows", "expected_lengths", "expected_usable"),
    [
        ([], [], []),
        (
            [{"length": 4}, {"invalid": True}, {"length": 4}, {"length": 2}],
            [4, 0, 4, 2],
            [0, 2, 3],
        ),
        ([{"invalid": True}, {"invalid": True}], [0, 0], []),
    ],
)
def test_sequential_canonical_metadata(rows, expected_lengths, expected_usable):
    lengths, usable = TRAINER.canonical_training_metadata(rows, prepare_row)

    assert lengths == expected_lengths
    assert usable == expected_usable


def test_sequential_canonical_metadata_respects_caller_train_limit():
    rows = [{"length": 2}, {"invalid": True}, {"length": 5}, {"length": 9}]

    lengths, usable = TRAINER.canonical_training_metadata(
        rows[:3],
        prepare_row,
    )

    assert lengths == [2, 0, 5]
    assert usable == [0, 2]


def _distributed_worker(
    rank,
    world,
    rendezvous,
    result_dir,
    rows,
    batch,
    failure,
    rows_by_rank,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:
        try:
            lengths, usable = TRAINER.canonical_training_metadata(
                rows[rank] if rows_by_rank else rows,
                prepare_row,
                device=torch.device("cpu"),
            )
            order = sorted(usable, key=lambda index: (-lengths[index], index))
            padded = order[:batch]
            local_assignment = padded[rank::world]
            result = {
                "lengths": lengths,
                "usable": usable,
                "metadata_digest": digest(
                    [
                        value
                        for pair in zip(
                            usable,
                            (lengths[index] for index in usable),
                        )
                        for value in pair
                    ]
                ),
                "order_digest": digest(order),
                "local_assignment": local_assignment,
            }
        except RuntimeError as error:
            if not failure:
                raise
            result = {"error": str(error)}
        Path(result_dir, f"{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def run_distributed(
    tmp_path,
    rows,
    *,
    world=4,
    batch=8,
    failure=False,
    rows_by_rank=False,
):
    rendezvous = tmp_path / "rendezvous"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _distributed_worker,
        args=(world, rendezvous, result_dir, rows, batch, failure, rows_by_rank),
        nprocs=world,
        join=True,
    )
    return [json.loads((result_dir / f"{rank}.json").read_text()) for rank in range(world)]


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"length": 5},
            {"invalid": True},
            {"length": 3},
            {"length": 8},
            {"length": 5},
            {"length": 2},
            {"length": 6},
        ],
        [{"length": 5}, {"length": 2}],
    ],
)
def test_four_process_metadata_and_assignments_match_sequential(tmp_path, rows):
    expected_lengths, expected_usable = TRAINER.canonical_training_metadata(
        rows,
        prepare_row,
    )
    expected_order = sorted(
        expected_usable,
        key=lambda index: (-expected_lengths[index], index),
    )
    results = run_distributed(tmp_path, rows)

    assert all(result["lengths"] == expected_lengths for result in results)
    assert all(result["usable"] == expected_usable for result in results)
    assert len({result["metadata_digest"] for result in results}) == 1
    assert len({result["order_digest"] for result in results}) == 1
    assert [result["local_assignment"] for result in results] == [
        expected_order[:8][rank::4] for rank in range(4)
    ]


def test_distributed_failures_report_globally_earliest_row_without_hanging(tmp_path):
    rows = [
        {"length": 2},
        {"failure": "earliest"},
        {"failure": "later"},
        {"length": 3},
    ]

    results = run_distributed(tmp_path, rows, failure=True)

    messages = {result["error"] for result in results}
    assert messages == {"Training row 1 preprocessing failed with builtins.ValueError: earliest"}


def test_distributed_row_count_mismatch_fails_before_metadata_exchange(tmp_path):
    rows = [
        [{"length": 2}],
        [{"length": 2}, {"length": 3}],
    ]

    results = run_distributed(
        tmp_path,
        rows,
        world=2,
        failure=True,
        rows_by_rank=True,
    )

    messages = {result["error"] for result in results}
    assert messages == {"Training row count differs across ranks: min=1, max=2"}
