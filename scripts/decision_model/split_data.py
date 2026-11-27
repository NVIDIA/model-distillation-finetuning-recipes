#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic, stratified train/validation/test splits from decision JSONL."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO, TypeVar

from scripts.decision_model import split_packed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_id(row: object, line_number: int) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: row must be an object")
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"line {line_number}: id must be a non-empty string")
    return row_id


def row_identity(row: object, *, line_number: int) -> tuple[str, str]:
    row_id = _row_id(row, line_number)
    reference = row.get("reference")
    target = reference.get("target") if isinstance(reference, dict) else None
    if not isinstance(target, str) or not target:
        raise ValueError(f"line {line_number}: reference.target must be a non-empty string")
    return row_id, target


def selection_key(seed: int, row_id: str) -> int:
    payload = f"{seed}\0{row_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_rows(path: Path):
    with path.open() as handle:
        for line_number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            yield line_number, text, row


def _count_labels(input_path: Path) -> tuple[Counter[str], int] | None:
    """Count scalar labels, or switch to joint stratification on the first mapped reference."""
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    rows = 0
    scalar_error = None
    for line_number, _, row in read_rows(input_path):
        reference = row.get("reference") if isinstance(row, dict) else None
        if isinstance(reference, dict) and "targets" in reference:
            return None
        # A later mapped reference selects packed parsing for the whole file,
        # including earlier scalar targets (e.g. boolean answers).
        try:
            row_id, target = row_identity(row, line_number=line_number)
        except ValueError as error:
            scalar_error = scalar_error or str(error)
            continue
        if row_id in seen_ids:
            raise ValueError(f"{input_path}:{line_number}: duplicate id {row_id!r}")
        seen_ids.add(row_id)
        counts[target] += 1
        rows += 1
    if scalar_error:
        raise ValueError(scalar_error)
    if rows < 3:
        raise ValueError("input must contain at least three rows")
    return counts, rows


def _split_quotas(
    label_counts: Counter[str], validation_fraction: float, test_fraction: float
) -> dict[str, dict[str, int]]:
    quotas: dict[str, dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
    for label, count in label_counts.items():
        if count < 3:
            raise ValueError(
                f"label {label!r} has {count} examples; at least three are required "
                "for stratification across train, validation, and test"
            )
        validation = min(count - 2, max(1, round(count * validation_fraction)))
        test = min(count - validation - 1, max(1, round(count * test_fraction)))
        quotas["train"][label] = count - validation - test
        quotas["validation"][label] = validation
        quotas["test"][label] = test
    return quotas


def _select_holdout_ids(
    input_path: Path, quotas: dict[str, dict[str, int]], seed: int
) -> tuple[set[str], set[str]]:
    # Each heap keeps the rows with the smallest stable hashes for one label.
    # Negating the hash makes heap[0] the currently largest selected hash.
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for line_number, _, row in read_rows(input_path):
        row_id, target = row_identity(row, line_number=line_number)
        key = selection_key(seed, row_id)
        heap = heaps[target]
        entry = (-key, row_id)
        holdout_count = quotas["validation"][target] + quotas["test"][target]
        if len(heap) < holdout_count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    for label, heap in heaps.items():
        ordered = [row_id for _, row_id in sorted(heap, reverse=True)]
        validation_count = quotas["validation"][label]
        validation_ids.update(ordered[:validation_count])
        test_ids.update(ordered[validation_count:])
    return validation_ids, test_ids


Label = TypeVar("Label", str, tuple[str, str])


def _check_stratification(
    quotas: dict[str, dict[Label, int]], counts: dict[str, Counter[Label]]
) -> None:
    for split, expected in quotas.items():
        if dict(counts[split]) != expected:
            raise ValueError(
                f"{split} split failed stratification: expected per-label counts "
                f"{expected}, got {dict(counts[split])}"
            )


def _open_temporary(path: Path) -> tuple[Path, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- caller owns this handle
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        delete=False,
    )
    return Path(handle.name), handle


def split_file(
    input_path: Path,
    train_output: Path,
    validation_output: Path,
    test_output: Path,
    *,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 101,
) -> dict[str, object]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    metadata_path = validation_output.with_suffix(validation_output.suffix + ".metadata.json")
    outputs = {"train": train_output, "validation": validation_output, "test": test_output}
    resolved = {path.resolve() for path in (input_path, *outputs.values(), metadata_path)}
    if len(resolved) != 5:
        raise ValueError(
            "input, train/validation/test outputs, and metadata output must be distinct"
        )

    source_sha = file_sha256(input_path)
    scalar_counts = _count_labels(input_path)
    packed = scalar_counts is None
    if scalar_counts is None:
        rows = ((_row_id(row, n), row) for n, _, row in read_rows(input_path))
        validation_ids, test_ids, quotas, input_rows = split_packed.plan(
            rows, validation_fraction, test_fraction, seed
        )
        label_counts = sum((Counter(counts) for counts in quotas.values()), Counter())
    else:
        label_counts, input_rows = scalar_counts
        quotas = _split_quotas(label_counts, validation_fraction, test_fraction)
        validation_ids, test_ids = _select_holdout_ids(input_path, quotas, seed)
    if validation_ids & test_ids:
        raise ValueError("validation and test splits must not overlap")

    counts = {split: Counter() for split in outputs}
    row_counts = Counter()
    staged: dict[str, Path] = {}
    temporaries: list[Path] = []
    try:
        with ExitStack() as stack:
            handles = {}
            for split, path in outputs.items():
                temporary, handle = _open_temporary(path)
                staged[split] = temporary
                temporaries.append(temporary)
                handles[split] = stack.enter_context(handle)
            for line_number, text, row in read_rows(input_path):
                if packed:
                    row_id = _row_id(row, line_number)
                    labels = split_packed.row_labels(row)
                else:
                    row_id, target = row_identity(row, line_number=line_number)
                    labels = [target]
                if row_id in validation_ids:
                    split = "validation"
                elif row_id in test_ids:
                    split = "test"
                else:
                    split = "train"
                handles[split].write(text if text.endswith("\n") else text + "\n")
                counts[split].update(labels)
                row_counts[split] += 1
        if file_sha256(input_path) != source_sha:
            raise ValueError("input changed during splitting; outputs were not accepted")
        _check_stratification(quotas, counts)
        count_field = "question_label_counts" if packed else "label_counts"
        summary: dict[str, object] = {
            "input": str(input_path),
            "input_sha256": source_sha,
            "input_rows": input_rows,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "stratified": True,
            count_field: split_packed.question_counts(label_counts)
            if packed
            else dict(sorted(label_counts.items())),
        }
        for split, path in outputs.items():
            summary[f"{split}_output"] = str(path)
            summary[f"{split}_sha256"] = file_sha256(staged[split])
            summary[f"{split}_rows"] = row_counts[split]
            summary[f"{split}_{count_field}"] = (
                split_packed.question_counts(counts[split])
                if packed
                else dict(sorted(counts[split].items()))
            )
        metadata_tmp, metadata_handle = _open_temporary(metadata_path)
        temporaries.append(metadata_tmp)
        with metadata_handle:
            metadata_handle.write(json.dumps(summary, indent=2) + "\n")
        for split, path in outputs.items():
            staged[split].replace(path)
        metadata_tmp.replace(metadata_path)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    summary = split_file(
        args.input,
        args.train_output,
        args.validation_output,
        args.test_output,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
