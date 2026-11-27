# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stratify whole states by the observed (question, target) pairs."""

import hashlib
import json
from collections import Counter, defaultdict

from scripts.decision_model.targets import (
    question_definitions,
    question_options,
    question_targets,
    target_label,
)


def row_labels(row: dict) -> list[tuple[str, str]]:
    questions = question_definitions(row)
    if "state" not in row["input"]:
        raise ValueError(f"row {row.get('id')!r} needs input.state")
    targets = question_targets(row, questions)
    labels = []
    for key, value in targets.items():
        question = questions[key]
        label = target_label(value, question.get("type"))
        if label not in question_options(question):
            raise ValueError(f"row {row.get('id')!r} question {key!r} has unknown target {label!r}")
        labels.append((key, label))
    return labels


def _allocate_profiles(profiles, sizes, fractions):
    """Allocate integer numbers of equivalent states, enforcing coverage in each split."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import block_diag, coo_matrix, eye, hstack

    labels = sorted({label for _, counts in profiles for label, _ in counts})
    label_index = {label: index for index, label in enumerate(labels, start=1)}
    # The first marginal counts rows. The others count each question/label pair.
    entries = []
    for column, (rows, counts) in enumerate(profiles):
        entries.append((0, column, rows))
        entries.extend((label_index[label], column, count) for label, count in counts)
    rows, cols, weights = zip(*entries)
    marginals = coo_matrix((weights, (rows, cols)), shape=(len(labels) + 1, len(profiles))).tocsr()
    supports = (marginals[1:] > 0) @ sizes
    for label, support in zip(labels, supports):
        if support < 3:
            raise ValueError(
                f"question/label {label!r} occurs in {support} distinct states; at least three are required"
            )

    # Variables are state counts per profile and split, followed by nonnegative
    # shortfalls from the requested row and label fractions in each split.
    counts = block_diag([marginals] * 3, format="csr")
    conservation = hstack([eye(len(profiles))] * 3)
    slack = eye(counts.shape[0])
    totals = marginals @ sizes
    targets = [fraction * total for fraction in fractions for total in totals]
    # Normalize so common labels cannot dominate rare labels. Give row balance
    # the same total weight as label balance.
    penalties = 1.0 / totals
    penalties[0] *= len(labels)
    num_allocations = counts.shape[1]
    num_shortfalls = counts.shape[0]
    no_shortfalls = coo_matrix((len(profiles), num_shortfalls))
    infinity = float("inf")
    result = milp(
        [0] * num_allocations + list(penalties) * 3,
        integrality=[1] * num_allocations + [0] * num_shortfalls,
        bounds=Bounds(0, infinity),
        constraints=[
            LinearConstraint(hstack([conservation, no_shortfalls]), sizes, sizes),
            LinearConstraint(hstack([counts, 0 * slack]), 1, infinity),
            LinearConstraint(hstack([counts, slack]), targets, infinity),
        ],
    )
    if result.status == 2:
        raise ValueError(
            "No whole-state split can cover every question/label in train, validation, and test"
        )
    if not result.success:
        raise RuntimeError(f"Packed stratification failed: {result.message}")
    allocation = result.x[:num_allocations].round().astype(int).reshape(3, -1)
    if (allocation < 0).any() or (allocation.sum(axis=0) != sizes).any():
        raise RuntimeError("Packed stratification returned invalid state counts")
    expected = {}
    for split, state_counts in zip(("train", "validation", "test"), allocation):
        counts_per_label = (marginals @ state_counts)[1:].tolist()
        if min(counts_per_label) < 1:
            raise RuntimeError(f"{split} split failed question/label coverage")
        expected[split] = dict(zip(labels, counts_per_label))
    return allocation, expected


def plan(rows, validation_fraction: float, test_fraction: float, seed: int):
    state_ids = defaultdict(list)
    state_counts = defaultdict(Counter)
    seen_ids = set()
    for row_id, row in rows:
        if row_id in seen_ids:
            raise ValueError(f"duplicate id {row_id!r}")
        seen_ids.add(row_id)
        labels = row_labels(row)
        state = json.dumps(row["input"]["state"], sort_keys=True, separators=(",", ":")).encode()
        state_key = hashlib.sha256(state).hexdigest()
        state_ids[state_key].append(row_id)
        state_counts[state_key].update(labels)
    # Group interchangeable states so the solver size depends on the number of
    # distinct label profiles rather than the number of examples.
    by_profile = defaultdict(list)
    for state_key, ids in state_ids.items():
        profile = (len(ids), tuple(sorted(state_counts[state_key].items())))
        by_profile[profile].append(state_key)
    profiles = sorted(by_profile)
    fractions = [1 - validation_fraction - test_fraction, validation_fraction, test_fraction]
    allocation, expected = _allocate_profiles(
        profiles, [len(by_profile[p]) for p in profiles], fractions
    )
    validation_ids, test_ids = set(), set()
    for profile, (_, validation, test) in zip(profiles, allocation.T):
        states = sorted(
            by_profile[profile], key=lambda key: hashlib.sha256(f"{seed}/{key}".encode()).digest()
        )
        for state in states[:validation]:
            validation_ids.update(state_ids[state])
        for state in states[validation : validation + test]:
            test_ids.update(state_ids[state])
    return validation_ids, test_ids, expected, len(seen_ids)


def question_counts(counts: dict) -> dict:
    result = {}
    for (question, label), count in sorted(counts.items()):
        result.setdefault(question, {})[label] = count
    return result
