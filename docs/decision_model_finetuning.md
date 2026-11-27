# Decision model fine-tuning with PyTorch FSDP2

This recipe fine-tunes a user-supplied decision model with an explicit PyTorch loop,
using financial-news event classification as an example. Both LoRA and full
fine-tuning minimize cross-entropy over each question's options. You supply the
checkpoint, tokenizer, and model code.

Run the commands below from the repository root.

## Prerequisites

Use the shared [environment setup](../README.md#requirements) and
[container](../README.md#container) from the repository README.

Use NVIDIA GPUs and a reviewed, immutable checkpoint snapshot. Model loading uses
`trust_remote_code=True`. Choose the interface your checkpoint implements.

### Transformers decision interface

Use `--native-decision` in place of `--prompt-module "$PROMPT_MODULE"` in the
fine-tuning, validation, and evaluation commands below. Install the Transformers
version required by your checkpoint, with `AutoModelForDecision` support. The
recipe passes each row’s `input` to the checkpoint and groups its raw scores by
question in the returned option order. For CPU inputs and a model on the same device, the call is

```python
for inputs, questions in model.decision_config.prepare_inputs(row["input"], tokenizer):
    scores = model(**inputs)
    per_question = scores.split([len(q.options) for q in questions])
```

The checkpoint selects its read layout and builds its attention masks. If it
exposes `decision_attention_layout`, `--packed-read` selects
`state_sees_no_questions` and saves that layout with the export. Each native
forward processes one prepared sequence. A per-device batch can include several
rows, and `--max-tokens` applies to each sequence. Standard backbones expose
`model.layers` and `model.norm`. The checkpoint’s `save_pretrained` must
preserve the decision configuration and template for reload.

### Prompt-module interface

The script expects `encoder.layers`, `encoder.norm`, a diffusion head, and a mask-token ID.

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
  batching, packed reads, and bidirectional option-code readout shared by
  fine-tuning/evaluation.
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

LoRA fine-tunes projection adapters. Full fine-tuning updates the backbone layers
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

## 6. State-blind packing with a prompt module

Some decision models serve several questions about one state in a single
sequence: the state once, then each question followed by its own mask token.
To fine-tune such a model, add `--packed-read` to train in the same layout:

```text
[state][question 1 ... mask][question 2 ... mask] ... [question n ... mask]
```

- The state attends only to the state, so it is encoded without seeing any
  question (state-blind).
- Each question attends to the state and to itself, never to another question.
- Question positions restart after the state.
- Each question is read at its own mask token, with the same loss as a single
  read. A row's loss is the mean over its questions.

A question's answer therefore does not depend on which or how many other
questions share its sequence. The flag is off by default; without it, training
and evaluation are unchanged.

### Prompt module

Set `STATE_END_TOKEN` to the special token your decision model uses to mark
the end of the state. Your prompt module returns its ID from
`state_end_token_id(tokenizer)`. The token must occur exactly once in each
prepared prompt. Every question about the same state must
share the same token IDs up to and including this token. The
[template](../templates/decision_prompt.py) and the
[synthetic example](../examples/decision_model/prompt_module.py) include both.

### Data

One-question rows from step 1 work unchanged; each is read as a one-question
pack. A row may also hold several questions about one state, with one target per
question key:

```jsonl
{"id":"news-007","input":{"state":"Example Corp acquired Sample Labs.","questions":{"event":{"type":"choice","instructions":"Which event best describes this headline?","criteria":{"acquisition":"One company buys another.","other":"None of the listed events."}},"sector":{"type":"choice","instructions":"Which sector is the buyer in?","criteria":{"technology":"Software or hardware.","finance":"Banking or payments."}}}},"reference":{"targets":{"event":"acquisition","sector":"technology"}}}
```

Use exactly one target format per row. `soft_labels` apply to one-question
rows with `reference.target`. A row whose packed sequence exceeds
`--max-tokens` is excluded, as in section 2.

Run the split command from section 1 on the complete labeled JSONL. When any
row uses `reference.targets`, it groups rows with the same `input.state` and
uses SciPy to assign whole states to train, validation, and test. It balances
row and question-label counts toward the requested fractions. Each observed
question-label pair must occur in at least three distinct states. If their
overlap makes full coverage impossible, splitting fails before writing the
outputs. The split metadata records row counts and per-question label counts.

### Commands

Add `--packed-read` to the commands of sections 2 to 5:

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  --method lora \
  --checkpoint "$CHECKPOINT" \
  --prompt-module "$PROMPT_MODULE" \
  --train "$TRAIN_JSONL" \
  --output "$RUN_ROOT/lora-packed" \
  --lr 3e-5 \
  --batch 4 \
  --per-device-batch-size 1 \
  --rank 32 \
  --alpha 64 \
  --epochs 1 \
  --packed-read \
  --no-save-best

CUDA_VISIBLE_DEVICES=0 python -m scripts.decision_model.validate_checkpoint \
  --checkpoint "$RUN_ROOT/lora-packed/last" \
  --data "$VALIDATION_JSONL" \
  --method lora \
  --packed-read

python -m scripts.decision_model.evaluate \
  --checkpoint "$RUN_ROOT/lora-packed/last" \
  --input "$TEST_JSONL" \
  --output-dir "$RESULTS_ROOT/lora-packed-test" \
  --gpus 0,1,2,3 \
  --packed-read
```

With `--packed-read`, the validator also reads the first validation row packed
and reports its prediction for every question.

The trainer logs both `rows` and `questions` in `validation.jsonl`.
Validation accuracy and cross-entropy are averaged over questions.

### Optional FlexAttention backend

Packed reads use SDPA with a dense token-pair mask by default. FlashAttention 2
cannot represent this mask. On PyTorch builds with
FlexAttention support, the equivalent block mask avoids carrying a dense
additive mask through every attention layer:

```bash
torchrun --standalone --nproc-per-node=4 --module scripts.decision_model.finetune_fsdp2 \
  ... \
  --packed-read \
  --packed-attention flex \
  --attn-implementation flex_attention
```

`--packed-attention flex` requires both other options shown above. The dense
backend remains the portable default. FlexAttention performance and memory use
are hardware-, shape-, and PyTorch-version-dependent, so benchmark both methods
at the intended production batch size before selecting it. PyTorch may
materialize a temporary boolean mask while constructing the BlockMask.

### Evaluate several questions per state

With `--packed-read`, test rows may also hold several questions about one state.
The evaluator reads each row once, in one packed sequence, and writes one
prediction per question. One-question rows keep the one-answer-per-row record:

```jsonl
{"id":"news-007","questions":{"event":{"probs":{"acquisition":0.97,"other":0.03},"pred":"acquisition"},"sector":{"probs":{"technology":0.88,"finance":0.12},"pred":"technology"}}}
{"id":"news-008","probs":{"acquisition":0.04,"partnership":0.91,"earnings":0.02,"other":0.03},"pred":"partnership"}
```

Each question gets the same token IDs, positions, and attention as in a
one-question pack of the same state. When the references hold
`reference.targets`, the scorer reports accuracy, log loss, and Brier score per
question, accuracy by question key, and the share of rows with every question
correct. `--taxonomy` is optional for these references:

```bash
python -m scripts.decision_model.score \
  --references "$TEST_JSONL" \
  --predictions "$RESULTS_ROOT/lora-packed-test/predictions.jsonl" \
  --run-metadata "$RESULTS_ROOT/lora-packed-test/run.json" \
  --output-json "$RESULTS_ROOT/lora-packed-test/score.json" \
  --output-markdown "$RESULTS_ROOT/lora-packed-test/score.md"
```

## CPU checks

Tests use synthetic fixtures and require no model downloads:

```bash
python -m pytest -m "not gpu and not distributed and not slow and not requires_model"
```

To exercise native full and LoRA fine-tuning, export, reload, and evaluation
with your own checkpoint, set its local snapshot path. The test uses up to two
visible NVIDIA GPUs and writes a full export for each method.

```bash
DECISION_CHECKPOINT=/models/decision-model python -m pytest -q tests/decision_model/test_native_checkpoint_gpu.py
```
