# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI validation shared by the decision-model readout commands."""

import argparse


def add_native_decision_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--native-decision",
        action="store_true",
        help="Use the Transformers decision interface supplied by the checkpoint",
    )


def check_packed_attention(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.native_decision:
        if args.prompt_module:
            parser.error(
                "--native-decision uses checkpoint preparation and conflicts with --prompt-module"
            )
        if getattr(args, "packed_attention", "dense") != "dense":
            parser.error("--native-decision delegates packed attention to the checkpoint")
        return
    if args.packed_read and args.attn_implementation.startswith("flash_attention_"):
        parser.error(
            "--packed-read needs a query-by-key attention mask; use --attn-implementation sdpa"
        )
