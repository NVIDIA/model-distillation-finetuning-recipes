#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable inventory and hashing for single-file or sharded HF safetensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in safetensors index: {key!r}")
        result[key] = value
    return result


def _weight_map(checkpoint: Path) -> dict[str, str] | None:
    index = checkpoint / "model.safetensors.index.json"
    if not index.exists() and not index.is_symlink():
        return None
    if not index.is_file():
        raise ValueError(f"Safetensors index is not a readable file: {index}")
    payload = json.loads(index.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    mapping = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Safetensors weight_map must be a non-empty object: {index}")
    for key, name in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid safetensors tensor key in {index}: {key!r}")
        if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
            raise ValueError(f"Invalid safetensors shard path in {index}: {name!r}")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or PureWindowsPath(name).drive
            or ".." in relative.parts
            or relative.as_posix() != name
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"Unsafe safetensors shard path in {index}: {name!r}")
    return mapping


def weight_files(checkpoint: Path) -> list[Path]:
    mapping = _weight_map(checkpoint)
    if mapping is not None:
        names = sorted(set(mapping.values()))
        paths = [checkpoint / name for name in names]
    else:
        single = checkpoint / "model.safetensors"
        paths = [single] if single.is_file() else sorted(checkpoint.glob("model-*.safetensors"))
    missing = [str(path) for path in paths if not path.is_file()]
    if not paths or missing:
        raise FileNotFoundError(
            f"Missing checkpoint weights under {checkpoint}: {missing or 'no safetensors files'}"
        )
    return paths


def weight_manifest(checkpoint: Path) -> dict:
    files = weight_files(checkpoint)
    entries = [
        {
            "file": path.relative_to(checkpoint).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["file"].encode())
        digest.update(b"\0")
        digest.update(entry["sha256"].encode())
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        # The index digest covers tensor-to-shard assignments separately from shard contents.
        "index_sha256": (
            file_sha256(checkpoint / "model.safetensors.index.json")
            if (checkpoint / "model.safetensors.index.json").is_file()
            else None
        ),
    }


def safetensors_inventory(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    inventory: dict[str, tuple[int, ...]] = {}
    locations: dict[str, str] = {}
    for path in weight_files(checkpoint):
        with safe_open(path, framework="pt", device="cpu") as handle:
            # safe_open is a file reader, not an iterable mapping.
            for key in handle.keys():  # noqa: SIM118
                if key in inventory:
                    raise ValueError(f"Duplicate tensor key {key!r} in {checkpoint}")
                inventory[key] = tuple(handle.get_slice(key).get_shape())
                locations[key] = path.relative_to(checkpoint).as_posix()
    mapping = _weight_map(checkpoint)
    if mapping is not None and locations != mapping:
        missing = sorted(set(mapping) - set(locations))
        unexpected = sorted(set(locations) - set(mapping))
        misplaced = sorted(
            key for key in locations.keys() & mapping.keys() if locations[key] != mapping[key]
        )
        raise ValueError(
            "Safetensors index does not match checkpoint tensors; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, misplaced={misplaced[:10]}"
        )
    return inventory
