# Decision model fine-tuning with PyTorch FSDP2

This recipe fine-tunes a user-supplied decision model with an explicit PyTorch loop,
using financial-news event classification as an example. It supports native
LoRA and decoder fine-tuning (`--method full`), uses shared FSDP2 helpers for
both methods, and keeps the serving-time decision objective:

1. Build the prompt with the serving prompt builder.
2. Append one mask token.
3. Run the encoder bidirectionally.
4. Select the option-code logits from the diffusion head.
5. Optimize hard- or soft-target cross-entropy.

This repository provides the fine-tuning workflow and synthetic data examples.
You supply the decision model checkpoint, tokenizer, custom model code, and
serving prompt module.

Run the commands below from the repository root.

## Prerequisites

Use the shared [environment setup](../README.md#requirements) and
[container](../README.md#container) from the repository README.

Provide NVIDIA GPUs and your own compatible decision model checkpoint,
tokenizer, and custom model code. The script expects `encoder.layers`,
`encoder.norm`, a diffusion head, and a mask-token ID. Model loading uses
`trust_remote_code=True`; use the reviewed code from an immutable checkpoint
snapshot.

The encoder also needs `embed_tokens` and `rotary_emb`; each layer's `self_attn`
must expose `diffusion_lm` and `is_causal`. Decoder calls must match the
[shared readout](../scripts/decision_model/runtime.py). By default, the model
must support `gradient_checkpointing_enable`; use `--no-grad-checkpointing`
otherwise.

Supply a self-contained prompt module with `--prompt-module`: one Python file
with its prompt helpers, loadable without the original source checkout. Exports
include it as `decision_prompt.py` for validation and evaluation.

Keep prompt-specific defaults and configuration handling in your supplied
module, and record any overrides with the run.

Copy the [template](../templates/decision_prompt.py) to a writable directory:

```bash
cp -n templates/decision_prompt.py /path/to/your/decision_prompt.py
```

Set `OPTION_CODES` and implement `_prompt_token_ids` to match your decision
model's serving format and tokenization. The template provides `candidate_codes`
and `prepare_questions`, including option ordering; the script appends the mask
token. See the [synthetic example](../examples/decision_model/prompt_module.py)
for a completed implementation. Set `PROMPT_MODULE` below to your adapted file.

Replace `CHECKPOINT` and `PROMPT_MODULE` below with paths to your own decision
model checkpoint and prompt module. Set writable artifact directories outside
the source checkout. Output directories for fine-tuning and inference must be new:

```bash
export CHECKPOINT=/models/decision-model
export PROMPT_MODULE="$CHECKPOINT/decision_prompt.py"
export CHAT_JSONL="$PWD/examples/decision_model/chat.jsonl"
export TAXONOMY="$PWD/examples/decision_model/taxonomy.json"
export DATA_ROOT=/outputs/decision/data
export TRAIN_JSONL="$DATA_ROOT/train.jsonl"
export VALIDATION_JSONL="$DATA_ROOT/validation.jsonl"
export TEST_JSONL="$DATA_ROOT/test.jsonl"
export RUN_ROOT=/outputs/decision/runs
export RESULTS_ROOT=/outputs/decision/results
export CODE_COMMIT="$(git rev-parse HEAD)"
export CONTAINER_IMAGE="<image-name>@sha256:<digest>"
```

Set `CONTAINER_IMAGE` to the actual runtime image digest and save the working-tree
status (`git status --short`) with the run record.

Use a fixed version of the model and tokenizer. For Hugging Face models,
specify the version with `--model-revision <commit>`. Save your fine-tuning command
and settings so you can repeat the run. See the
[reproducibility guide](reproducibility.md)
for details.

## Files

- `examples/decision_model/`: synthetic chat data, taxonomy, and prompt example.
- `templates/decision_prompt.py`: starter module to adapt to your decision model.
- `scripts/decision_model/convert_chat.py`: chat examples with category labels to
  decision rows.
- `scripts/decision_model/split_data.py`: stratified train/validation/test splits.
- `scripts/decision_model/finetune_fsdp2.py`: LoRA and full fine-tuning.
- `scripts/decision_model/runtime.py`: prompt-module loading, input validation,
  batching, and bidirectional option-code readout shared by fine-tuning/evaluation.
- `scripts/shared/fsdp2_common.py`: shared FSDP2, scheduling, and checkpoint helpers.
- `scripts/decision_model/checkpoint_manifest.py`: checkpoint inventory and checksums.
- `scripts/decision_model/validate_checkpoint.py`: manifest verification, CUDA reload,
  and comparison of the encoder and shared readout paths.
- `scripts/decision_model/evaluate.py`: multi-GPU checkpoint inference.
- `scripts/decision_model/score.py`: classification metrics.

## 1. Convert chat data

Store one JSON object per line (**JSONL**). Each row contains the text as a
`user` message and its category as the final `assistant` message, wrapped in
double brackets:

```jsonl
{"id":"finance-001","messages":[{"role":"user","content":"Example Corp acquired Sample Labs."},{"role":"assistant","content":"[[acquisition]]"}]}
{"id":"finance-002","messages":[{"role":"user","content":"Example Bank and Sample Pay announced a payments partnership."},{"role":"assistant","content":"[[partnership]]"}]}
```

The converter uses the last user message. IDs are optional, but must be unique
when provided.

The **taxonomy** defines the question and descriptions of at least two
categories:

```json
{
  "question": "Which event best describes this financial news headline?",
  "labels": {
    "acquisition": "One company buys another company or business.",
    "partnership": "Companies announce an agreement to work together.",
    "earnings": "A company reports its financial results.",
    "other": "The headline describes none of the listed events."
  }
}
```

Assistant labels must exactly match taxonomy keys, including case. The paths
above select the supplied [chat data](../examples/decision_model/chat.jsonl) and
[taxonomy](../examples/decision_model/taxonomy.json).

Convert the data:

```bash
python -m scripts.decision_model.convert_chat \
  --input "$CHAT_JSONL" \
  --taxonomy "$TAXONOMY" \
  --output "$DATA_ROOT/decisions.jsonl"
```

For a separate text field, add `--state-field headline`. Optional `soft_labels`
map taxonomy keys to non-negative weights with a positive total; weights are
normalized, and the assistant label is still required. Use the converter's
`--help` for other options.

Create train, validation, and test splits. Use validation for checkpoint selection
and reserve test for the final evaluation:

```bash
python -m scripts.decision_model.split_data \
  --input "$DATA_ROOT/decisions.jsonl" \
  --train-output "$TRAIN_JSONL" \
  --validation-output "$VALIDATION_JSONL" \
  --test-output "$TEST_JSONL" \
  --validation-fraction 0.05 \
  --test-fraction 0.05 \
  --seed 101
```

Fractions target 90% train, 5% validation, and 5% test. Each category needs at
least three examples; the script reserves one per split and verifies all
per-category counts. Rounding gives 12/4/4 rows for the supplied examples.
Split metadata records counts, checksums, and the stratification result.

## 2. Fine-tune

The examples start with four GPUs, per-device batch 1, and global batch 4.
Increase the batch sizes to fit your model, dataset, and GPUs.

`--epochs 1` runs one pass over the train data. The script calculates the
optimizer-step count from the usable dataset size and global batch. Change
`--epochs` to run more passes, or use `--max-steps` for a fixed step budget.

Prompts exceeding `--max-tokens` (default: 8192) are excluded from fine-tuning.
`usable_train_rows` in `config.json` records the remaining count.

### LoRA

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  --method lora \
  --checkpoint "$CHECKPOINT" \
  --prompt-module "$PROMPT_MODULE" \
  --train "$TRAIN_JSONL" \
  --output "$RUN_ROOT/lora" \
  --lr 3e-5 \
  --batch 4 \
  --per-device-batch-size 1 \
  --rank 32 \
  --alpha 64 \
  --epochs 1 \
  --no-save-best
```

### Full fine-tuning

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  --method full \
  --checkpoint "$CHECKPOINT" \
  --prompt-module "$PROMPT_MODULE" \
  --train "$TRAIN_JSONL" \
  --output "$RUN_ROOT/full" \
  --lr 1e-6 \
  --batch 4 \
  --per-device-batch-size 1 \
  --epochs 1 \
  --no-save-best
```

Other optimizer, schedule, loss, and module-target settings use the
[script's defaults](../scripts/decision_model/finetune_fsdp2.py). To list all options:

```bash
python -m scripts.decision_model.finetune_fsdp2 --help
```

LoRA fine-tunes projection adapters. Full fine-tuning updates the encoder layers
and final norm while keeping embeddings and the output head frozen.

`--batch` is the global batch and must be divisible by the per-device batch
multiplied by the GPU count. The script uses gradient accumulation when
needed. Keep batch sizes equal across methods when comparing throughput.

The commands export `last/` for evaluation in step 5. To select a checkpoint
during fine-tuning, add `--eval "$VALIDATION_JSONL" --eval-every <steps>` and omit
`--no-save-best`.

## 3. Resume fine-tuning

To split a run across jobs, save a distributed checkpoint and resume it in a
new output directory. Keep the same model, data, GPU count, fine-tuning settings,
and full epoch count (or `--max-steps` budget). Carry over any overrides from
section 2.

Choose `lora` or `full`. Set `STOP_STEP` to the absolute optimizer-step number
at which to pause, below the run's total steps:

```bash
export METHOD=lora  # or full
export STOP_STEP="<pause-step>"
```

Run the first part and save its fine-tuning state:

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  --method "$METHOD" \
  --checkpoint "$CHECKPOINT" \
  --prompt-module "$PROMPT_MODULE" \
  --train "$TRAIN_JSONL" \
  --output "$RUN_ROOT/${METHOD}-part-1" \
  --batch 4 \
  --per-device-batch-size 1 \
  --epochs 1 \
  --stop-step "$STOP_STEP" \
  --save-dcp "$RUN_ROOT/${METHOD}-dcp" \
  --no-save-best
```

Resume from the saved `step_<n>` directory and complete the run:

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  --method "$METHOD" \
  --checkpoint "$CHECKPOINT" \
  --prompt-module "$PROMPT_MODULE" \
  --train "$TRAIN_JSONL" \
  --output "$RUN_ROOT/${METHOD}-part-2" \
  --batch 4 \
  --per-device-batch-size 1 \
  --epochs 1 \
  --resume "$RUN_ROOT/${METHOD}-dcp/step_${STOP_STEP}" \
  --no-save-best
```

## 4. Export and validate

Fine-tuning automatically exports a complete Hugging Face checkpoint in `last/`.
LoRA adapters are merged into the base weights.

Select the method and exported checkpoint:

```bash
export METHOD=lora  # or full
export FINETUNED_CHECKPOINT="$RUN_ROOT/$METHOD/last"
```

For a resumed run, set `FINETUNED_CHECKPOINT` to
`$RUN_ROOT/${METHOD}-part-2/last`.

If you enabled validation-based checkpoint selection, use the corresponding
`best/` directory instead.

Validate the checkpoint on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.decision_model.validate_checkpoint \
  --checkpoint "$FINETUNED_CHECKPOINT" \
  --data "$VALIDATION_JSONL" \
  --method "$METHOD"
```

The validator checks exported files, reloads the model, and compares the
encoder and shared scorer outputs on one validation example.

## 5. Evaluate the test split

Finalize checkpoint choices using validation first. Evaluate both methods on
the same held-out test data and taxonomy, using a new directory for each run:

```bash
export EVAL_DIR="$RESULTS_ROOT/${METHOD}-test"

python -m scripts.decision_model.evaluate \
  --checkpoint "$FINETUNED_CHECKPOINT" \
  --input "$TEST_JSONL" \
  --output-dir "$EVAL_DIR" \
  --gpus 0,1,2,3
```

Score the predictions and save JSON and Markdown reports:

```bash
python -m scripts.decision_model.score \
  --references "$TEST_JSONL" \
  --predictions "$EVAL_DIR/predictions.jsonl" \
  --taxonomy "$TAXONOMY" \
  --run-metadata "$EVAL_DIR/run.json" \
  --output-json "$EVAL_DIR/score.json" \
  --output-markdown "$EVAL_DIR/score.md"
```

Reports include accuracy, F1, log loss, and calibration metrics. Evaluation
requires complete coverage; scoring also verifies the supplied run metadata
against the input and prediction files. Use each script's `--help` for
additional options.

## CPU checks

Tests use synthetic fixtures and require no model downloads:

```bash
python -m pytest -m "not gpu and not distributed and not slow and not requires_model"
```

GPU-backed validation remains required before publishing the recipe.
