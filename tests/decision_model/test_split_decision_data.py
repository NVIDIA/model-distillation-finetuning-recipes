# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.decision_model import split_data as split_decision_data

REPO = Path(__file__).resolve().parents[2]


class SplitDecisionDataTest(unittest.TestCase):
    def write_rows(self, path: Path) -> None:
        rows = []
        for label, count in (("common", 40), ("rare", 3)):
            for index in range(count):
                rows.append(
                    {
                        "id": f"{label}-{index}",
                        "input": {"state": {"headline": f"{label} {index}"}},
                        "reference": {"target": label},
                    }
                )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_split_is_disjoint_stratified_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)

            first_train = root / "first-train.jsonl"
            first_validation = root / "first-validation.jsonl"
            first_test = root / "first-test.jsonl"
            summary = split_decision_data.split_file(
                source,
                first_train,
                first_validation,
                first_test,
                validation_fraction=0.2,
                test_fraction=0.1,
                seed=101,
            )
            second_train = root / "second-train.jsonl"
            second_validation = root / "second-validation.jsonl"
            second_test = root / "second-test.jsonl"
            split_decision_data.split_file(
                source,
                second_train,
                second_validation,
                second_test,
                validation_fraction=0.2,
                test_fraction=0.1,
                seed=101,
            )

            train_rows = [json.loads(line) for line in first_train.read_text().splitlines()]
            validation_rows = [
                json.loads(line) for line in first_validation.read_text().splitlines()
            ]
            test_rows = [json.loads(line) for line in first_test.read_text().splitlines()]
            train_ids = {row["id"] for row in train_rows}
            validation_ids = {row["id"] for row in validation_rows}
            test_ids = {row["id"] for row in test_rows}

            self.assertFalse(train_ids & validation_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(validation_ids & test_ids)
            self.assertEqual(len(train_ids | validation_ids | test_ids), 43)
            expected_counts = {
                "train": {"common": 28, "rare": 1},
                "validation": {"common": 8, "rare": 1},
                "test": {"common": 4, "rare": 1},
            }
            for split, rows in (
                ("train", train_rows),
                ("validation", validation_rows),
                ("test", test_rows),
            ):
                self.assertEqual({row["reference"]["target"] for row in rows}, {"common", "rare"})
                self.assertEqual(summary[f"{split}_label_counts"], expected_counts[split])
                self.assertEqual(summary[f"{split}_rows"], len(rows))
            self.assertTrue(summary["stratified"])
            self.assertEqual(first_train.read_bytes(), second_train.read_bytes())
            self.assertEqual(first_validation.read_bytes(), second_validation.read_bytes())
            self.assertEqual(first_test.read_bytes(), second_test.read_bytes())

    def test_cli_writes_three_splits_and_verified_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "decisions.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.decision_model.convert_chat",
                    "--input",
                    str(REPO / "examples/decision_model/chat.jsonl"),
                    "--taxonomy",
                    str(REPO / "examples/decision_model/taxonomy.json"),
                    "--output",
                    str(source),
                ],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test")}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.decision_model.split_data",
                    "--input",
                    str(source),
                    "--train-output",
                    str(paths["train"]),
                    "--validation-output",
                    str(paths["validation"]),
                    "--test-output",
                    str(paths["test"]),
                    "--validation-fraction",
                    "0.05",
                    "--test-fraction",
                    "0.05",
                    "--seed",
                    "101",
                ],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertTrue(summary["stratified"])
            for split, expected_rows in (("train", 12), ("validation", 4), ("test", 4)):
                self.assertEqual(summary[f"{split}_rows"], expected_rows)
                self.assertEqual(
                    set(summary[f"{split}_label_counts"]), set(summary["label_counts"])
                )
                self.assertEqual(
                    summary[f"{split}_sha256"],
                    hashlib.sha256(paths[split].read_bytes()).hexdigest(),
                )
            metadata = paths["validation"].with_suffix(".jsonl.metadata.json")
            self.assertEqual(json.loads(metadata.read_text()), summary)

    def test_rejects_labels_with_too_few_examples_for_three_splits(self):
        for count in (1, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.jsonl"
                self.write_rows(source)
                with source.open("a") as handle:
                    for index in range(count):
                        handle.write(
                            json.dumps({"id": f"scarce-{index}", "reference": {"target": "scarce"}})
                            + "\n"
                        )
                with self.assertRaisesRegex(ValueError, "'scarce'.*at least three"):
                    split_decision_data.split_file(
                        source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                    )
                self.assertEqual(list(root.iterdir()), [source])

    def test_rejects_invalid_holdout_fractions(self):
        for validation, test in ((0, 0.05), (0.05, 0), (0.6, 0.4), (0.9, 0.2)):
            with (
                self.subTest(validation=validation, test=test),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source.jsonl"
                self.write_rows(source)
                with self.assertRaisesRegex(ValueError, "fraction"):
                    split_decision_data.split_file(
                        source,
                        root / "train.jsonl",
                        root / "validation.jsonl",
                        root / "test.jsonl",
                        validation_fraction=validation,
                        test_fraction=test,
                    )
                self.assertEqual(list(root.iterdir()), [source])

    def test_stratification_failure_rejects_outputs_even_when_all_labels_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            with (
                patch.object(
                    split_decision_data,
                    "_select_holdout_ids",
                    return_value=({"common-0", "rare-0"}, {"common-1", "rare-1"}),
                ),
                self.assertRaisesRegex(ValueError, "failed stratification"),
            ):
                split_decision_data.split_file(
                    source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                )
            self.assertEqual(list(root.iterdir()), [source])

    def test_overlapping_holdouts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            with (
                patch.object(
                    split_decision_data,
                    "_select_holdout_ids",
                    return_value=({"common-0"}, {"common-0"}),
                ),
                self.assertRaisesRegex(ValueError, "must not overlap"),
            ):
                split_decision_data.split_file(
                    source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                )
            self.assertEqual(list(root.iterdir()), [source])

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "same", "reference": {"target": "first"}}),
                        json.dumps({"id": "same", "reference": {"target": "second"}}),
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    root / "test.jsonl",
                )

    def test_metadata_output_cannot_overwrite_input_or_split_outputs(self):
        for protected in ("input", "train", "test"):
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                validation = root / "validation.jsonl"
                metadata = validation.with_suffix(".jsonl.metadata.json")
                source = metadata if protected == "input" else root / "input.jsonl"
                train = metadata if protected == "train" else root / "train.jsonl"
                test = metadata if protected == "test" else root / "test.jsonl"
                source.write_text("protected source")
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    split_decision_data.split_file(source, train, validation, test)
                self.assertEqual(source.read_text(), "protected source")
                self.assertFalse(validation.exists())

    def test_source_change_between_passes_rejects_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            select = split_decision_data._select_holdout_ids

            def change_after_selection(*args):
                selected = select(*args)
                source.write_text(source.read_text() + "\n")
                return selected

            with (
                patch.object(split_decision_data, "_select_holdout_ids", change_after_selection),
                self.assertRaisesRegex(ValueError, "input changed during splitting"),
            ):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    root / "test.jsonl",
                )
            self.assertFalse((root / "train.jsonl").exists())
            self.assertFalse((root / "validation.jsonl").exists())
            self.assertFalse((root / "test.jsonl").exists())
            self.assertFalse(list(root.glob(".*.tmp-*")))

    def test_output_creation_failure_cleans_existing_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            invalid_parent = root / "not-a-directory"
            invalid_parent.write_text("keep")
            with self.assertRaises(FileExistsError):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    invalid_parent / "test.jsonl",
                )
            self.assertFalse((root / "train.jsonl").exists())
            self.assertFalse(list(root.glob(".*.tmp-*")))


def packed_row(row_id, state, targets):
    criteria = {
        "event": {"acquisition": "Acquisition", "other": "Other"},
        "sector": {"finance": "Finance", "technology": "Technology"},
    }
    questions = {
        key: {"type": "choice", "instructions": "Pick one.", "criteria": criteria[key]}
        for key in targets
    }
    return {
        "id": row_id,
        "input": {"state": state, "questions": questions},
        "reference": {"targets": targets},
    }


def test_packed_split_keeps_states_together_and_covers_each_question_label(tmp_path):
    rows = []
    for index in range(8):
        state = {"headline": f"Company {index} news", "day": 1}
        event = ["acquisition", "other"][index % 2]
        sector = ["finance", "technology"][(index // 2) % 2]
        rows.append(packed_row(f"news-{index}", state, {"event": event, "sector": sector}))
        single = packed_row(f"again-{index}", dict(reversed(state.items())), {"event": event})
        single["reference"] = {"target": event}
        rows.append(single)
    source = tmp_path / "source.jsonl"
    paths = {name: tmp_path / name for name in ("train", "validation", "test")}
    assignments = []
    for order in (rows, reversed(rows)):
        source.write_text("".join(json.dumps(row) + "\n" for row in order))
        summary = split_decision_data.split_file(
            source, *paths.values(), validation_fraction=0.25, test_fraction=0.25
        )
        states = {}
        seen_ids = set()
        for name, path in paths.items():
            selected = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(selected) == summary[f"{name}_rows"] == (8 if name == "train" else 4)
            observed = {"event": Counter(), "sector": Counter()}
            for row in selected:
                assert row["id"] not in seen_ids
                seen_ids.add(row["id"])
                state = row["input"]["state"]["headline"]
                assert states.setdefault(state, name) == name
                reference = row["reference"]
                targets = reference.get("targets") or {"event": reference["target"]}
                for key, label in targets.items():
                    observed[key][label] += 1
            assert set(observed["event"]) == {"acquisition", "other"}
            assert set(observed["sector"]) == {"finance", "technology"}
            assert summary[f"{name}_question_label_counts"] == observed
        assert seen_ids == {row["id"] for row in rows}
        assert summary["question_label_counts"] == {
            "event": {"acquisition": 8, "other": 8},
            "sector": {"finance": 4, "technology": 4},
        }
        assignments.append(states)
    assert assignments[0] == assignments[1]


@pytest.mark.parametrize("impossible_overlap", [False, True])
def test_packed_split_refuses_infeasible_coverage_before_writing(tmp_path, impossible_overlap):
    source = tmp_path / "source.jsonl"
    if impossible_overlap:
        # Every three-state subset would need three different split assignments.
        rows = []
        for state in range(4):
            keys = [str(n) for n in range(4) if n != state]
            questions = {
                key: {"type": "choice", "criteria": {"yes": "Y", "no": "N"}} for key in keys
            }
            rows.append(
                {
                    "id": str(state),
                    "input": {"state": state, "questions": questions},
                    "reference": {"targets": dict.fromkeys(keys, "yes")},
                }
            )
        message = "No whole-state split"
    else:
        rows = [packed_row(str(i), i % 2, {"event": "other"}) for i in range(6)]
        message = "2 distinct states"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        split_decision_data.split_file(
            source, *(tmp_path / p for p in ("train", "validation", "test"))
        )
    assert list(tmp_path.iterdir()) == [source]


def test_mixed_boolean_targets_share_a_stratum_regardless_of_row_order(tmp_path):
    source = tmp_path / "source.jsonl"
    paths = [tmp_path / name for name in ("train", "validation", "test")]
    question = {"type": "noul", "criteria": {"true": "Yes", "false": "No"}}
    references = [{"target": True}, {"targets": {"deal": "TRUE"}}, {"target": "true"}]
    rows = [
        {"id": str(i), "input": {"state": i, "questions": {"deal": question}}, "reference": ref}
        for i, ref in enumerate(references)
    ]
    for order in (rows, rows[1:] + rows[:1]):
        source.write_text("".join(json.dumps(row) + "\n" for row in order))
        summary = split_decision_data.split_file(source, *paths)
        for path in paths:
            assert summary[f"{path.name}_question_label_counts"] == {"deal": {"true": 1}}


if __name__ == "__main__":
    unittest.main()
