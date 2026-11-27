# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI validation shared by the decision-model readout commands."""

import argparse


def check_packed_attention(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.packed_read and args.attn_implementation.startswith("flash_attention_"):
        parser.error(
            "--packed-read needs a query-by-key attention mask; use --attn-implementation sdpa"
        )
