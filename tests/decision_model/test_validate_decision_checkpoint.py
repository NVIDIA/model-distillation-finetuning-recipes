# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.decision_model import validate_checkpoint as validator
from scripts.decision_model.checkpoint_manifest import file_sha256, weight_manifest


def checkpoint_fixture(tmp_path):
    weights = tmp_path / 'model.safetensors'
    save_file({'weight': torch.ones(2)}, str(weights))
    (tmp_path / 'config.json').write_text('{}')
    prompt = tmp_path / 'decision_prompt.py'
    prompt.write_text('# synthetic prompt module')
    manifest = weight_manifest(tmp_path)
    receipt = {
        'fsdp2': True,
        'method': 'lora',
        'tensor_count': 1,
        'model_sha256': manifest['sha256'],
        'model_bytes': manifest['bytes'],
        'index_sha256': manifest['index_sha256'],
        'prompt_module_sha256': file_sha256(prompt),
    }
    (tmp_path / 'pure-pytorch-receipt.json').write_text(json.dumps(receipt))
    return receipt, prompt


def test_receipt_checks_weights_tensor_count_and_prompt_identity(tmp_path):
    receipt, prompt = checkpoint_fixture(tmp_path)
    actual, manifest = validator.verify_receipt(tmp_path, method='lora', prompt_module=prompt)
    assert actual == receipt
    assert manifest['sha256'] == receipt['model_sha256']
    with pytest.raises(ValueError, match='method'):
        validator.verify_receipt(tmp_path, method='full', prompt_module=prompt)
    prompt.write_text('# changed prompt')
    with pytest.raises(ValueError, match='prompt module SHA-256'):
        validator.verify_receipt(tmp_path, prompt_module=prompt)


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('tensor_count', 2, 'tensor count'),
        ('tensor_count', True, 'tensor count'),
        ('model_sha256', 'bad', 'model_sha256'),
        ('model_bytes', 1, 'model_bytes'),
        ('index_sha256', 'bad', 'index SHA-256'),
        ('fsdp2', 1, 'FSDP2'),
    ],
)
def test_receipt_rejects_mismatched_export_metadata(tmp_path, field, value, match):
    receipt, _ = checkpoint_fixture(tmp_path)
    receipt[field] = value
    (tmp_path / 'pure-pytorch-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        validator.verify_receipt(tmp_path)


def test_export_checksums_detect_configuration_tampering(tmp_path):
    checkpoint_fixture(tmp_path)
    config = tmp_path / 'config.json'
    (tmp_path / 'SHA256SUMS').write_text(f'{file_sha256(config)}  config.json\n')
    validator.verify_export_checksums(tmp_path)
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        validator.verify_export_checksums(tmp_path)


@pytest.mark.parametrize(
    'name', ['../outside', '/outside', 'C:outside', 'nested\\outside', './config.json']
)
def test_export_checksums_reject_unsafe_or_ambiguous_paths(tmp_path, name):
    (tmp_path / 'SHA256SUMS').write_text(f'{"a" * 64}  {name}\n')
    with pytest.raises(ValueError, match='invalid checksum entry'):
        validator.verify_export_checksums(tmp_path)
