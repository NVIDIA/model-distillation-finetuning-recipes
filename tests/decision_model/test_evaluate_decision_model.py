# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from pathlib import Path

import pytest

from scripts.decision_model import evaluate as evaluator

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('value', ['', '0,', '-1', '0,0', '0,00', 'a'])
def test_gpu_selection_rejects_invalid_and_duplicate_indices(value):
    with pytest.raises(ValueError):
        evaluator.parse_gpus(value)


def test_gpu_selection_canonicalizes_whitespace_and_indices():
    assert evaluator.parse_gpus(' 00, 2 ') == ['0', '2']


def test_merge_requires_coverage_and_preserves_source_order(tmp_path):
    source = [{'id': 'first'}, {'id': 'second'}]
    path = tmp_path / 'predictions.jsonl'
    evaluator.write_jsonl(path, [{'id': 'second'}])
    with pytest.raises(ValueError, match='missing predictions'):
        evaluator.merge_predictions(source, [path])
    assert evaluator.merge_predictions(source, [path], allow_partial=True) == [{'id': 'second'}]
    evaluator.write_jsonl(path, [{'id': 'second'}, {'id': 'first'}])
    assert evaluator.merge_predictions(source, [path]) == source


@pytest.mark.parametrize('rows', [[{'id': ''}], [{'id': None}], [{'id': 'same'}, {'id': 'same'}]])
def test_input_ids_must_be_unique_nonempty_strings(rows):
    with pytest.raises(ValueError):
        evaluator.validate_ids(rows, 'input')


def test_merge_rejects_extra_and_cross_worker_duplicate_predictions(tmp_path):
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    evaluator.write_jsonl(first, [{'id': 'unknown'}])
    with pytest.raises(ValueError, match='do not correspond'):
        evaluator.merge_predictions([{'id': 'first'}], [first], allow_partial=True)
    evaluator.write_jsonl(first, [{'id': 'first'}])
    evaluator.write_jsonl(second, [{'id': 'first'}])
    with pytest.raises(ValueError, match='duplicate prediction ID'):
        evaluator.merge_predictions([{'id': 'first'}], [first, second])


def evaluation_inputs(tmp_path, gpus='0,1'):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    (checkpoint / 'model.safetensors').write_bytes(b'fixture weights for CPU worker mocks')
    (checkpoint / 'decision_prompt.py').write_text('# model-supplied prompt fixture')
    source = tmp_path / 'input.jsonl'
    evaluator.write_jsonl(source, [{'id': 'first'}, {'id': 'second'}])
    args = evaluator.parse_args(
        [
            '--checkpoint',
            str(checkpoint),
            '--input',
            str(source),
            '--output-dir',
            str(tmp_path / 'output'),
            '--gpus',
            gpus,
        ]
    )
    return args


@pytest.mark.parametrize('setting', [None, 'user-value'])
def test_native_workers_preserve_caller_environment_and_record_complete_coverage(
    tmp_path, monkeypatch, setting
):
    args = evaluation_inputs(tmp_path)
    # Initialize shared dependencies before isolating the caller environment.
    importlib.import_module('scripts.decision_model.runtime')
    caller_environment = {}
    if setting is not None:
        caller_environment = {
            'USER_PROMPT_SETTING': setting,
            'TOKENIZERS_PARALLELISM': 'true',
        }
    monkeypatch.setattr(evaluator.os, 'environ', caller_environment.copy())
    launched = []

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def launch(command, *, cwd, env, stdout, stderr):
        assert command[:3] == [sys.executable, '-m', 'scripts.decision_model.evaluate']
        assert Path(cwd) == ROOT
        assert '--worker' in command
        expected_environment = dict(
            caller_environment, CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES']
        )
        expected_environment.setdefault('TOKENIZERS_PARALLELISM', 'false')
        assert env == expected_environment
        assert stdout and stderr
        launched.append(env['CUDA_VISIBLE_DEVICES'])
        worker_dir = Path(command[command.index('--output-dir') + 1])
        shard = Path(command[command.index('--input') + 1])
        records = [dict(row, probs={'A': 0.8, 'B': 0.2}) for row in evaluator.read_jsonl(shard)]
        evaluator.write_jsonl(worker_dir / 'predictions.jsonl', records)
        (worker_dir / 'summary.json').write_text(
            json.dumps(
                {
                    'input_rows': len(records),
                    'predicted_rows': len(records),
                    'failures': 0,
                    'peak_allocated_gib': 1.0,
                }
            )
        )
        return Process()

    monkeypatch.setattr(evaluator.subprocess, 'Popen', launch)
    metadata = evaluator.run_evaluation(args)
    assert launched == ['0', '1']
    assert metadata['predicted_rows'] == metadata['input_rows'] == 2
    assert metadata['evaluator_failures'] == 0
    assert 'scoring_environment' not in metadata
    assert dict(evaluator.os.environ) == caller_environment
    assert len(metadata['prompt_module_sha256']) == 64
    assert [row['id'] for row in evaluator.read_jsonl(Path(metadata['predictions']))] == [
        'first',
        'second',
    ]


@pytest.mark.parametrize('failed_worker', [0, 1])
def test_worker_failure_terminates_other_workers_and_does_not_publish_predictions(
    tmp_path, monkeypatch, failed_worker
):
    args = evaluation_inputs(tmp_path)
    processes = []

    class Process:
        def __init__(self, worker):
            self.code = 1 if worker == failed_worker else None
            self.terminated = False

        def wait(self, timeout=None):
            assert self.code is not None or self.terminated, 'waited for a running worker'
            self.code = -15 if self.terminated else self.code
            return self.code

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated = True

    def launch(*_args, **_kwargs):
        process = Process(worker=len(processes))
        processes.append(process)
        return process

    monkeypatch.setattr(evaluator.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match=f'worker {failed_worker}/GPU {failed_worker} failed'):
        evaluator.run_evaluation(args)
    assert processes[1 - failed_worker].terminated
    assert not (args.output_dir / 'predictions.jsonl').exists()
    assert not (args.output_dir / 'run.json').exists()


@pytest.mark.parametrize('changed_file', ['input', 'prompt module'])
def test_changed_evaluation_inputs_do_not_publish_results(tmp_path, monkeypatch, changed_file):
    args = evaluation_inputs(tmp_path)
    prompt = args.checkpoint / 'decision_prompt.py'

    class Process:
        def poll(self):
            return 0

    def launch(command, **_kwargs):
        worker_dir = Path(command[command.index('--output-dir') + 1])
        shard = Path(command[command.index('--input') + 1])
        records = [dict(row, probs={'A': 0.8, 'B': 0.2}) for row in evaluator.read_jsonl(shard)]
        evaluator.write_jsonl(worker_dir / 'predictions.jsonl', records)
        (worker_dir / 'summary.json').write_text(
            json.dumps({'failures': 0, 'peak_allocated_gib': 1.0})
        )
        if changed_file == 'input':
            evaluator.write_jsonl(args.input, [{'id': 'first', 'changed': True}, {'id': 'second'}])
        else:
            prompt.write_text('# changed prompt implementation')
        return Process()

    monkeypatch.setattr(evaluator.subprocess, 'Popen', launch)

    with pytest.raises(ValueError, match=f'{changed_file} changed during evaluation'):
        evaluator.run_evaluation(args)

    assert not (args.output_dir / 'predictions.jsonl').exists()
    assert not (args.output_dir / 'run.json').exists()
