# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reference FSDP2 recipes for LoRA supervised fine-tuning and logit-level
  knowledge distillation.
- Assistant-only masking with Qwen3 and NemotronH chat templates.
- CPU validation tests and a configurable container build path with explicit
  reproducibility requirements.
- Public documentation for container setup, distributed execution, input data,
  generated artifacts, and reproducibility responsibilities.
- User-supplied decision-model fine-tuning, synthetic prompt/data examples,
  stratified train/validation/test splits, checkpoint validation, and metrics.

### Changed

- Project/package name, container working directory, and image examples now use
  `model-distillation-finetuning-recipes`.
- Scripts and tests are grouped into `causal_lm`, `decision_model`, and `shared`.
  Launch recipes from the repository root with `python -m` or `torchrun --module`.

### Removed

- The old top-level LoRA, distillation, and dataloader-verifier script entry points.

The first public release will be version 0.1.0.
