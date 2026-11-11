# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starter prompt module for a user-supplied decision model.

Copy this file to your own decision_prompt.py, then:
1. Set OPTION_CODES to your model's serving option codes, in positional order.
2. Implement _prompt_token_ids with your model's serving format and tokenization.

Keep prompt helpers in this file so it can load from an exported checkpoint.
See examples/decision_model/prompt_module.py for a synthetic implementation.
"""

OPTION_CODES = ()  # TODO: supply one code per supported option position.


def candidate_codes(tokenizer):
    """Return ordered (code, token_id) pairs for your model's option codes."""
    if len(OPTION_CODES) < 2:
        raise NotImplementedError("Set OPTION_CODES to your model's serving option codes")
    codes = []
    for code in OPTION_CODES:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"Option code {code!r} must be one known token")
        codes.append((code, ids[0]))
    if len({token_id for _, token_id in codes}) != len(codes):
        raise ValueError("Option codes must have distinct token IDs")
    return codes


def _prompt_token_ids(tokenizer, state, instructions, options):
    """TODO: reproduce your model's serving prompt and return its token IDs.

    state is text or a JSON value; instructions is the question text.
    options contains ordered (code, label, description) tuples.
    Preserve this option order when formatting the prompt. Return token IDs
    before the mask token; the trainer appends that token.
    """
    raise NotImplementedError("Implement your model's serving prompt and tokenization")


def prepare_questions(tokenizer, payload, position_ordering=True, permutation=None):
    """Prepare (key, kind, labels, descriptions, prompt_token_ids) records."""
    if not position_ordering:
        raise ValueError("This template requires positional option ordering")
    codes = candidate_codes(tokenizer)
    prepared = []
    for key, question in payload["questions"].items():
        if question["type"] != "choice":
            raise ValueError("This template supports only choice questions")
        options = list(question["criteria"].items())
        if not 2 <= len(options) <= len(codes):
            raise ValueError("Provide at least two choices and enough option codes")
        if permutation is not None:
            order = permutation(key, len(options))
            if sorted(order) != list(range(len(options))):
                raise ValueError("Option permutation must contain each index once")
            options = [options[index] for index in order]
        labels = [label for label, _ in options]
        descriptions = [description for _, description in options]
        prompt_options = [
            (code, label, description) for (code, _), (label, description) in zip(codes, options)
        ]
        ids = _prompt_token_ids(
            tokenizer, payload["state"], question["instructions"], prompt_options
        )
        prepared.append((key, question["type"], labels, descriptions, ids))
    return prepared
