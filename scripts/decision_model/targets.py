# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-label conventions shared by decision-model data tools and readers."""

from typing import Any


def question_definitions(row: dict) -> dict[str, dict]:
    """Validate the question keys and label sets used by splitting and scoring."""
    payload = row.get("input")
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"reference {row.get('id')!r} has invalid input.questions")
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError(f"reference {row.get('id')!r} has invalid input.questions")
        criteria = question.get("criteria")
        if (
            not isinstance(criteria, dict)
            or len(criteria) < 2
            or any(not isinstance(label, str) or not label for label in criteria)
        ):
            raise ValueError(
                f"reference {row.get('id')!r} question {key!r} "
                "needs at least two unique non-empty criteria labels"
            )
    return questions


def question_targets(row: dict, questions: dict) -> dict[str, Any]:
    """Resolve exactly one reference representation against the input question keys."""
    reference = row.get("reference")
    if not isinstance(reference, dict):
        raise ValueError(f"row {row.get('id')!r} lacks reference")
    if "targets" in reference:
        if "target" in reference:
            raise ValueError(
                f"row {row.get('id')!r} has both reference.target and reference.targets"
            )
        targets = reference["targets"]
        if not isinstance(targets, dict) or not questions or set(targets) != set(questions):
            raise ValueError(f"row {row.get('id')!r} has invalid reference.targets")
        return targets
    if len(questions) == 1 and "target" in reference:
        return {next(iter(questions)): reference["target"]}
    raise ValueError(f"row {row.get('id')!r} needs reference.targets for every question")


def target_label(value: Any, kind: str | None) -> str:
    return str(value).lower() if kind == "noul" else str(value)
