# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic example of the decision-model prompt interface.

Adapt the text format and option codes to your own model's serving convention
before fine-tuning. This example supports two to four choices and needs only the
standard library and the tokenizer supplied by the caller.
"""

import json

OPTION_CODES = ("A", "B", "C", "D")


def candidate_codes(tokenizer):
    """Return the option codes and their individual vocabulary token IDs."""
    codes = []
    for code in OPTION_CODES:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"Example option code {code!r} must be one known token")
        codes.append((code, ids[0]))
    if len({token_id for _, token_id in codes}) != len(codes):
        raise ValueError("Example option codes must have distinct token IDs")
    return codes


def prepare_questions(tokenizer, payload, position_ordering=True, permutation=None):
    """Return (key, kind, labels, descriptions, prompt_token_ids) records.

    ``permutation(key, size)`` supplies option indices in the desired order.
    Labels, descriptions, and prompt options must all use that same order.
    The trainer appends the mask token after these prompt token IDs.
    """
    if not position_ordering:
        raise ValueError("This example requires positional option ordering")
    state = payload["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    codes = candidate_codes(tokenizer)
    prepared = []
    for key, question in payload["questions"].items():
        if question["type"] != "choice":
            raise ValueError("This example supports only choice questions")
        options = list(question["criteria"].items())
        if not 2 <= len(options) <= len(codes):
            raise ValueError("This example requires two to four choices")
        if permutation is not None:
            order = permutation(key, len(options))
            if sorted(order) != list(range(len(options))):
                raise ValueError("Option permutation must contain each index once")
            options = [options[index] for index in order]
        labels = [label for label, _ in options]
        descriptions = [description for _, description in options]
        option_text = "\n".join(
            f"{code}. {label}: {description}"
            for (code, _), (label, description) in zip(codes, options)
        )
        prompt = (
            f"State:\n{state}\n\nQuestion:\n{question['instructions']}\n\n"
            f"Options:\n{option_text}\n\nAnswer:"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prepared.append((key, question["type"], labels, descriptions, ids))
    return prepared
