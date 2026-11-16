# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math
import os
import random
import sys
from types import SimpleNamespace

import pytest

from scripts.decision_model.runtime import (
    DecisionRows,
    load_prompt_module,
    resolve_prompt_module,
)

PROMPT_SOURCE = """
import os

IMPORTED_PROMPT_SETTING = os.environ.get("USER_PROMPT_SETTING")

def candidate_codes(tokenizer):
    return tokenizer.codes

def prepare_questions(tokenizer, payload, position_ordering=True, permutation=None):
    assert payload["model"] == "decision-model"
    assert position_ordering is True
    key, question = next(iter(payload["questions"].items()))
    labels = list(question["criteria"])
    descriptions = list(question["criteria"].values())
    if permutation is not None:
        order = permutation(key, len(labels))
        labels = [labels[index] for index in order]
        descriptions = [descriptions[index] for index in order]
    return [(key, question["type"], labels, descriptions, tokenizer.ids)]
"""


def prompt_file(tmp_path, source=PROMPT_SOURCE):
    path = tmp_path / "decision_prompt.py"
    # Keep independent fixture modules out of each other's content cache.
    path.write_text(source + f"\n# fixture: {tmp_path}\n", encoding="utf-8")
    return path


def tokenizer():
    return SimpleNamespace(ids=[1, 2], codes=[("A", 10), ("B", 11), ("C", 12)])


def row():
    return {
        "id": "example",
        "input": {
            "state": "Synthetic example.",
            "questions": {
                "event": {
                    "type": "choice",
                    "criteria": {"alpha": "First", "beta": "Second", "gamma": "Third"},
                }
            },
        },
        "reference": {"target": "gamma"},
    }


def prepared_record(labels=None, ids=None):
    return [
        (
            "event",
            "choice",
            ["alpha", "beta", "gamma"] if labels is None else labels,
            ["First", "Second", "Third"],
            [1, 2] if ids is None else ids,
        )
    ]


def test_resolves_snapshot_prompt_module_and_explicit_override(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    default = prompt_file(snapshot)
    explicit = tmp_path / "custom.py"
    explicit.write_text(PROMPT_SOURCE)

    assert resolve_prompt_module(snapshot) == default.resolve()
    assert resolve_prompt_module(snapshot, explicit) == explicit.resolve()
    assert resolve_prompt_module("model/identifier", explicit) == explicit.resolve()


@pytest.mark.parametrize("name", ["missing.py", "module.txt", "directory.py"])
def test_rejects_missing_or_non_python_prompt_files(tmp_path, name):
    path = tmp_path / name
    if name.endswith(".txt"):
        path.write_text(PROMPT_SOURCE)
    elif name.startswith("directory"):
        path.mkdir()

    with pytest.raises(FileNotFoundError, match="prompt module"):
        resolve_prompt_module(tmp_path, path)
    with pytest.raises(FileNotFoundError, match="prompt module"):
        load_prompt_module(path)


def test_loads_module_under_source_digest_and_reuses_unchanged_code(tmp_path):
    path = prompt_file(tmp_path)
    expected = "_decision_prompt_" + hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_prompt_module(path)

    assert loaded.__name__ == expected
    assert load_prompt_module(path) is loaded


def test_updated_source_is_loaded_even_when_python_bytecode_metadata_matches(tmp_path):
    first_source = PROMPT_SOURCE + "\nMARKER = 1\n"
    second_source = PROMPT_SOURCE + "\nMARKER = 2\n"
    assert len(first_source) == len(second_source)
    path = tmp_path / "decision_prompt.py"
    path.write_text(first_source)
    original_stat = path.stat()
    first = load_prompt_module(path)
    path.write_text(second_source)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = load_prompt_module(path)

    assert first.MARKER == 1
    assert second.MARKER == 2
    assert first is not second


@pytest.mark.parametrize(
    "source, missing",
    [
        ("candidate_codes = None\ndef prepare_questions(*args, **kwargs): pass", "candidate_codes"),
        ("def candidate_codes(tokenizer): return []", "prepare_questions"),
    ],
)
def test_enforces_module_callables_and_removes_failed_cache_entries(tmp_path, source, missing):
    path = prompt_file(tmp_path, source)
    name = "_decision_prompt_" + hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=f"callable {missing}"):
        load_prompt_module(path)

    assert name not in sys.modules


def test_failed_module_execution_does_not_poison_content_cache(tmp_path):
    path = prompt_file(tmp_path, 'raise RuntimeError("synthetic import failure")')
    name = "_decision_prompt_" + hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="synthetic import failure"):
        load_prompt_module(path)

    assert name not in sys.modules


@pytest.mark.parametrize("setting", [None, "user-value"])
def test_prompt_loading_preserves_caller_environment(tmp_path, monkeypatch, setting):
    caller_environment = {} if setting is None else {"USER_PROMPT_SETTING": setting}
    monkeypatch.setattr(os, "environ", caller_environment.copy())

    path = prompt_file(tmp_path)
    loaded = load_prompt_module(path)

    assert loaded.IMPORTED_PROMPT_SETTING == setting
    assert dict(os.environ) == caller_environment
    assert load_prompt_module(path) is loaded
    assert dict(os.environ) == caller_environment


def test_prepares_hard_target_without_mutating_source_row(tmp_path):
    source = row()
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)

    assert prepare(source) == {
        "ids": [1, 2],
        "gold": 2,
        "soft": None,
        "cand": [10, 11, 12],
        "labels": ["alpha", "beta", "gamma"],
    }
    assert "model" not in source["input"]


def test_option_permutation_aligns_hard_and_soft_targets(tmp_path):
    source = row()
    source["soft_labels"] = {"alpha": 1, "beta": 3, "gamma": 6}
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)
    expected = ["alpha", "beta", "gamma"]
    random.Random(7).shuffle(expected)
    assert expected != ["alpha", "beta", "gamma"]

    example = prepare(source, shuffle_seed=7)

    assert example["labels"] == expected
    assert example["gold"] == expected.index("gamma")
    assert example["soft"] == pytest.approx(
        [source["soft_labels"][label] / 10 for label in expected]
    )
    assert example["cand"] == [10, 11, 12]
    assert prepare(source, shuffle_seed=7) == example


def test_soft_targets_normalize_without_overflow(tmp_path):
    source = row()
    source["soft_labels"] = {label: 1e308 for label in ("alpha", "beta", "gamma")}
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)

    soft = prepare(source)["soft"]

    assert all(math.isfinite(weight) and weight > 0 for weight in soft)
    assert soft == pytest.approx([1 / 3] * 3)
    assert math.fsum(soft) == pytest.approx(1)


@pytest.mark.parametrize(
    "soft",
    [
        [],
        {"unknown": 1},
        {"alpha": -1},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {},
        {"alpha": 0},
        {"alpha": "not-a-number"},
        {"alpha": None},
    ],
)
def test_rejects_invalid_soft_targets(tmp_path, soft):
    source = row()
    source["soft_labels"] = soft
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)

    with pytest.raises(ValueError):
        prepare(source)


@pytest.mark.parametrize("source", [None, [], {}, {"input": {}}, {"input": {"questions": {}}}])
def test_rejects_missing_or_invalid_row_shape(tmp_path, source):
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)

    with pytest.raises(ValueError):
        prepare(source)


def test_rejects_multiple_questions_missing_target_and_unknown_target(tmp_path):
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)
    multiple = row()
    multiple["input"]["questions"]["second"] = {}
    missing = row()
    missing["reference"] = {}
    unknown = row()
    unknown["reference"]["target"] = "unknown"

    for source in [multiple, missing, unknown]:
        with pytest.raises(ValueError):
            prepare(source)


@pytest.mark.parametrize("labels", [["alpha"], ["alpha", "alpha"], ["alpha", ""], [[], "alpha"]])
def test_rejects_invalid_prepared_labels(tmp_path, monkeypatch, labels):
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)
    monkeypatch.setattr(
        prepare.scoring, "prepare_questions", lambda *a, **kw: prepared_record(labels)
    )

    with pytest.raises(ValueError):
        prepare(row())


@pytest.mark.parametrize("ids", [[], [-1], [True], [1.0]])
def test_rejects_invalid_prompt_token_ids(tmp_path, monkeypatch, ids):
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)
    monkeypatch.setattr(
        prepare.scoring, "prepare_questions", lambda *a, **kw: prepared_record(ids=ids)
    )

    with pytest.raises(ValueError, match="token IDs"):
        prepare(row())


@pytest.mark.parametrize("prepared", [None, [], [(), ()]])
def test_rejects_invalid_prepared_question_count(tmp_path, monkeypatch, prepared):
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)
    monkeypatch.setattr(prepare.scoring, "prepare_questions", lambda *a, **kw: prepared)

    with pytest.raises(ValueError):
        prepare(row())


@pytest.mark.parametrize(
    "codes",
    [
        None,
        [1, 2, 3],
        [("A", 10)],
        [("A", 10), ("B", 10), ("C", 12)],
        [("A", True), ("B", 11), ("C", 12)],
        [("A", -1), ("B", 11), ("C", 12)],
    ],
)
def test_rejects_invalid_option_code_tables(tmp_path, codes):
    tok = tokenizer()
    tok.codes = codes

    with pytest.raises(ValueError):
        prepare = DecisionRows(tok, prompt_file(tmp_path), max_tokens=3)
        prepare(row())


def test_boolean_question_targets_are_normalized(tmp_path):
    source = row()
    source["input"]["questions"]["event"] = {
        "type": "noul",
        "criteria": {"false": "No", "true": "Yes"},
    }
    source["reference"]["target"] = True
    prepare = DecisionRows(tokenizer(), prompt_file(tmp_path), max_tokens=3)

    assert prepare(source)["gold"] == 1


def test_length_limit_includes_appended_mask_token(tmp_path):
    path = prompt_file(tmp_path)

    assert DecisionRows(tokenizer(), path, max_tokens=3)(row()) is not None
    assert DecisionRows(tokenizer(), path, max_tokens=2)(row()) is None
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        DecisionRows(tokenizer(), path, max_tokens=0)
