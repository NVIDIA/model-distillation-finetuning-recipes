#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert labeled chat JSONL into one-question decision-model rows.

The last user message supplies the state unless --state-field selects a
source field. The final assistant turn supplies a [[label]] target from the
explicit taxonomy. Optional soft labels are normalized in taxonomy order.
Each output file is replaced atomically.

See docs/decision_model_finetuning.md, section 1, for input examples and commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

LABEL_RE = re.compile(r"^\[\[([^\[\]]+?)\]\]$")


class _DuplicateIdError(ValueError):
    """Duplicate IDs remain fatal even when invalid rows may be skipped."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wrapped_label(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    match = LABEL_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"{field} is not a double-bracketed label: {value!r}")
    label = match.group(1).strip()
    if not label:
        raise ValueError(f"{field} contains an empty label")
    return label


def reference_label(
    source: dict[str, Any],
    *,
    raw_label_field: str | None = None,
) -> str:
    if not isinstance(source, dict):
        raise ValueError("each input row must be a JSON object")
    messages = source.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    assistant = messages[-1]
    if not isinstance(assistant, dict):
        raise ValueError("the final message must be an object")
    if assistant.get("role") != "assistant":
        raise ValueError("the final message must be the assistant label")
    message_label = parse_wrapped_label(assistant.get("content"), field="assistant content")
    if raw_label_field:
        if raw_label_field not in source:
            raise ValueError(f"missing {raw_label_field!r}")
        raw_label = parse_wrapped_label(
            source.get(raw_label_field),
            field=raw_label_field,
        )
        if message_label != raw_label:
            raise ValueError(
                f"assistant label {message_label!r} disagrees with {raw_label_field} {raw_label!r}"
            )
    return message_label


def convert_row(
    source: dict[str, Any],
    *,
    question: str,
    labels: dict[str, str],
    state_field: str | None = None,
    state_key: str | None = None,
    id_field: str = "id",
    raw_label_field: str | None = None,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("each input row must be a JSON object")
    row_id = source.get(id_field, fallback_id)
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("id must be a non-empty string")
    if state_field:
        state_value = source.get(state_field)
        if state_value is None or (isinstance(state_value, str) and not state_value.strip()):
            raise ValueError(f"{state_field} must be present and non-empty")
        state: Any = {state_key or state_field: state_value}
    else:
        messages = source.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages must contain a user turn before the assistant label")
        user_message = next(
            (
                message
                for message in reversed(messages[:-1])
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        if user_message is None or not isinstance(user_message.get("content"), str):
            raise ValueError("no user message was found before the assistant label")
        state = user_message["content"]
        if not state.strip():
            raise ValueError("the final user message must not be empty")
    target = reference_label(source, raw_label_field=raw_label_field)
    if target not in labels:
        raise ValueError(f"unknown target label {target!r}")

    row = {
        "id": row_id,
        "input": {
            "state": state,
            "questions": {
                "event_type": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": labels,
                }
            },
        },
        "reference": {"target": target},
    }
    soft_labels = source.get("soft_labels")
    if soft_labels is not None:
        if not isinstance(soft_labels, dict):
            raise ValueError("soft_labels must be an object keyed by taxonomy label")
        weights: dict[str, float] = {}
        for raw_label, raw_weight in soft_labels.items():
            soft_label = str(raw_label)
            if soft_label not in labels:
                raise ValueError(f"soft_labels contains unknown label {soft_label!r}")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"soft_labels[{soft_label!r}] must be numeric") from error
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"soft_labels[{soft_label!r}] must be finite and non-negative")
            weights[soft_label] = weight
        scale = max(weights.values(), default=0.0)
        if scale <= 0:
            raise ValueError("soft_labels must have positive total weight")
        # Scaling first avoids overflow when individually finite weights have
        # a total larger than the floating-point range.
        scaled = {label: weight / scale for label, weight in weights.items()}
        total = math.fsum(scaled.values())
        row["soft_labels"] = {label: scaled.get(label, 0.0) / total for label in labels}
    if "source_index" in source:
        row["source_index"] = source["source_index"]
    return row


def prepare_file(
    input_path: Path,
    manifest_path: Path | None,
    taxonomy_path: Path,
    output_path: Path,
    *,
    split_name: str | None = None,
    state_field: str | None = None,
    state_key: str | None = None,
    id_field: str = "id",
    raw_label_field: str | None = None,
    on_error: str = "fail",
) -> dict[str, Any]:
    if on_error not in {"fail", "skip"}:
        raise ValueError("on_error must be 'fail' or 'skip'")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    rejected_path = output_path.with_suffix(output_path.suffix + ".rejected.jsonl")
    outputs = {path.resolve() for path in (output_path, metadata_path, rejected_path)}
    inputs = {path.resolve() for path in (input_path, taxonomy_path)}
    if manifest_path is not None:
        inputs.add(manifest_path.resolve())
    if len(outputs) != 3 or inputs & outputs:
        raise ValueError("input files, outputs, and output sidecars must be distinct")
    taxonomy = json.loads(taxonomy_path.read_text())
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be a JSON object")
    question = taxonomy.get("question")
    labels = taxonomy.get("labels")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("taxonomy.question must be a non-empty string")
    if (
        not isinstance(labels, dict)
        or len(labels) < 2
        or any(
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(description, str)
            or not description.strip()
            for label, description in labels.items()
        )
    ):
        raise ValueError(
            "taxonomy.labels must contain at least two non-empty string labels and descriptions"
        )

    source_sha = file_sha256(input_path)
    expected_rows = None
    if manifest_path:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        manifest_labels = manifest.get("valid_labels")
        if manifest_labels != list(labels):
            raise ValueError("taxonomy labels or order do not exactly match manifest.valid_labels")
        selected_split = split_name or input_path.stem
        splits = manifest.get("splits")
        split = splits.get(selected_split) if isinstance(splits, dict) else None
        if not isinstance(split, dict):
            raise ValueError(f"manifest does not describe split {selected_split!r}")
        expected_rows = split.get("completed_rows")
        if type(expected_rows) is not int or expected_rows < 0:
            raise ValueError("manifest completed_rows must be a non-negative integer")
        expected_sha = split.get("sha256")
        if source_sha != expected_sha:
            raise ValueError(f"source checksum mismatch: expected {expected_sha}, got {source_sha}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    rejected_temporary = rejected_path.with_name(f".{rejected_path.name}.tmp-{os.getpid()}")
    metadata_temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    input_count = 0
    row_count = 0
    rejected_count = 0
    try:
        with (
            input_path.open() as source_handle,
            temporary.open("w") as output_handle,
            rejected_temporary.open("w") as rejected_handle,
        ):
            for line_number, text in enumerate(source_handle, 1):
                if not text.strip():
                    continue
                input_count += 1
                source = None
                try:
                    source = json.loads(text)
                    source_id = (
                        source.get(id_field, f"row-{line_number}")
                        if isinstance(source, dict)
                        else None
                    )
                    if isinstance(source_id, str) and source_id:
                        if source_id in seen_ids:
                            raise _DuplicateIdError(
                                f"{input_path}:{line_number}: duplicate id {source_id!r}"
                            )
                        seen_ids.add(source_id)
                    converted = convert_row(
                        source,
                        question=question,
                        labels=labels,
                        state_field=state_field,
                        state_key=state_key,
                        id_field=id_field,
                        raw_label_field=raw_label_field,
                        fallback_id=f"row-{line_number}",
                    )
                except _DuplicateIdError:
                    raise
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    if on_error == "fail":
                        raise ValueError(f"{input_path}:{line_number}: {error}") from error
                    rejected_count += 1
                    rejected_handle.write(
                        json.dumps(
                            {
                                "line": line_number,
                                "id": source.get(id_field) if isinstance(source, dict) else None,
                                "error": str(error),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                target = converted["reference"]["target"]
                label_counts[target] += 1
                row_count += 1
                output_handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
        if expected_rows is not None and input_count != expected_rows:
            raise ValueError(
                f"input row count mismatch: expected {expected_rows}, got {input_count}"
            )
        if file_sha256(input_path) != source_sha:
            raise ValueError("input changed during conversion; outputs were not accepted")
        if row_count == 0:
            raise ValueError("conversion produced no valid rows")
        summary = {
            "input": str(input_path),
            "input_sha256": source_sha,
            "manifest": str(manifest_path) if manifest_path else None,
            "taxonomy": str(taxonomy_path),
            "output": str(output_path),
            "output_sha256": file_sha256(temporary),
            "rows": row_count,
            "input_rows": input_count,
            "skipped_rows": rejected_count,
            "rejected_output": str(rejected_path) if rejected_count else None,
            "labels": len(labels),
            "state_source": state_field or "last user message",
            "label_distribution": dict(label_counts),
        }
        metadata_temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output_path)
        if rejected_count:
            rejected_temporary.replace(rejected_path)
        else:
            rejected_path.unlink(missing_ok=True)
        metadata_temporary.replace(metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        rejected_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
    return summary


def main(description: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional manifest with valid_labels and split row count/checksum",
    )
    parser.add_argument(
        "--split",
        default="",
        help="manifest split name (default: input filename stem)",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-field",
        default="",
        help="top-level field to use as state; default uses the last user message",
    )
    parser.add_argument(
        "--state-key",
        default="",
        help="key wrapping --state-field in the decision model state (default: field name)",
    )
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--raw-label-field",
        default="",
        help="optional top-level field that must match the final assistant label",
    )
    parser.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="fail",
        help="fail immediately or skip invalid rows and write a rejected JSONL",
    )
    args = parser.parse_args()
    if args.state_key and not args.state_field:
        parser.error("--state-key requires --state-field")
    summary = prepare_file(
        args.input,
        args.manifest,
        args.taxonomy,
        args.output,
        split_name=args.split or None,
        state_field=args.state_field or None,
        state_key=args.state_key or None,
        id_field=args.id_field,
        raw_label_field=args.raw_label_field or None,
        on_error=args.on_error,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
