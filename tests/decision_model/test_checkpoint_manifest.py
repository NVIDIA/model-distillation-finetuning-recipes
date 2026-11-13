# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.decision_model.checkpoint_manifest import (
    file_sha256,
    safetensors_inventory,
    weight_files,
    weight_manifest,
)


def write_index(checkpoint, mapping, **metadata):
    path = checkpoint / "model.safetensors.index.json"
    path.write_text(json.dumps({"metadata": metadata, "weight_map": mapping}), encoding="utf-8")
    return path


def test_single_weight_manifest_matches_file_and_legacy_digest(tmp_path):
    weights = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, 3)}, str(weights))

    manifest = weight_manifest(tmp_path)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    aggregate = hashlib.sha256(f"model.safetensors\0{digest}\n".encode()).hexdigest()

    assert manifest == {
        "sha256": aggregate,
        "bytes": weights.stat().st_size,
        "files": [{"file": weights.name, "bytes": weights.stat().st_size, "sha256": digest}],
        "index_sha256": None,
    }
    assert safetensors_inventory(tmp_path) == {"weight": (2, 3)}


def test_index_hash_detects_changes_without_changing_legacy_weight_digest(tmp_path):
    weights = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"weight": torch.ones(2)}, str(weights))
    index = write_index(tmp_path, {"weight": weights.name}, total_size=8)
    before = weight_manifest(tmp_path)
    assert before["index_sha256"] == file_sha256(index)

    write_index(tmp_path, {"weight": weights.name}, total_size=9)
    after = weight_manifest(tmp_path)

    assert before["sha256"] == after["sha256"]
    assert before["index_sha256"] != after["index_sha256"]


@pytest.mark.parametrize(
    "payload",
    [[], None, {}, {"weight_map": {}}, {"weight_map": []}, {"weight_map": "model.safetensors"}],
)
def test_rejects_malformed_weight_map(tmp_path, payload):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="weight_map must be a non-empty object"):
        weight_files(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.safetensors",
        "/outside.safetensors",
        "nested/../../outside.safetensors",
        "C:/outside.safetensors",
        "C:outside.safetensors",
        "nested\\outside.safetensors",
        "./model.safetensors",
        "nested//model.safetensors",
        "model.bin",
        "",
        "model\0.safetensors",
        None,
        ["model.safetensors"],
    ],
)
def test_rejects_unsafe_or_invalid_indexed_shard_paths(tmp_path, name):
    write_index(tmp_path, {"weight": name})

    with pytest.raises(ValueError, match="safetensors shard path"):
        weight_manifest(tmp_path)


def test_rejects_empty_tensor_key(tmp_path):
    write_index(tmp_path, {"": "model.safetensors"})

    with pytest.raises(ValueError, match="tensor key"):
        weight_files(tmp_path)


def test_rejects_duplicate_index_tensor_keys(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"first.safetensors","a":"second.safetensors"}}'
    )

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        weight_files(tmp_path)


def test_reports_missing_indexed_shard(tmp_path):
    write_index(tmp_path, {"weight": "missing.safetensors"})

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        weight_manifest(tmp_path)


def test_rejects_directory_in_place_of_index(tmp_path):
    save_file({"weight": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").mkdir()

    with pytest.raises(ValueError, match="index is not a readable file"):
        weight_files(tmp_path)


@pytest.mark.parametrize(
    "mapping",
    [
        {"a": "second.safetensors", "b": "first.safetensors"},
        {"missing": "first.safetensors", "b": "second.safetensors"},
    ],
)
def test_inventory_requires_exact_index_tensor_locations(tmp_path, mapping):
    save_file({"a": torch.ones(2)}, str(tmp_path / "first.safetensors"))
    save_file({"b": torch.ones(3)}, str(tmp_path / "second.safetensors"))
    write_index(tmp_path, mapping)

    with pytest.raises(ValueError, match="index does not match checkpoint tensors"):
        safetensors_inventory(tmp_path)


def test_inventory_rejects_duplicate_tensors_across_shards(tmp_path):
    save_file({"a": torch.ones(2)}, str(tmp_path / "first.safetensors"))
    save_file({"a": torch.ones(3)}, str(tmp_path / "second.safetensors"))
    write_index(tmp_path, {"a": "first.safetensors", "other": "second.safetensors"})

    with pytest.raises(ValueError, match="Duplicate tensor key"):
        safetensors_inventory(tmp_path)


def test_supports_nested_shards_and_hf_cache_symlinks(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    weights = cache / "blob"
    save_file({"weight": torch.ones(2)}, str(weights))
    checkpoint = tmp_path / "snapshot"
    (checkpoint / "nested").mkdir(parents=True)
    linked = checkpoint / "nested/model.safetensors"
    linked.symlink_to(weights)
    write_index(checkpoint, {"weight": "nested/model.safetensors"})

    assert weight_files(checkpoint) == [linked]
    assert safetensors_inventory(checkpoint) == {"weight": (2,)}
    assert weight_manifest(checkpoint)["files"][0]["sha256"] == file_sha256(weights)


def test_unindexed_shards_are_sorted_and_missing_weights_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors files"):
        weight_files(tmp_path)
    for name in ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]:
        save_file({name: torch.ones(1)}, str(tmp_path / name))

    assert [path.name for path in weight_files(tmp_path)] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
