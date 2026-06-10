# Refactor Copy Usage

This directory is a copy of `Dino+LLaMA` made to avoid changing the active source tree.

## Environment

From the project root:

```bash
source scripts/env_school.sh
```

This sets the school proxy, Java 11, and `TOKENIZERS_PARALLELISM=false`.

## Preflight

Check required paths without loading the LLM:

```bash
bash scripts/preflight_mimic_qwen3_4b.sh
```

Equivalent:

```bash
python -u train.py --config configs/experiments/mimic_qwen3_4b.yaml --preflight_only
```

For a slower annotation/image sample check:

```bash
python -u train.py \
  --config configs/experiments/mimic_qwen3_4b.yaml \
  --preflight_only \
  --data_sample_check 1
```

## Smoke Run

Run a tiny one-device job before launching full training:

```bash
bash scripts/smoke_mimic_qwen3_4b.sh
```

The smoke config writes to:

```text
./save/smoke/mimic_qwen3_4b
```

## KG Entity Label Cache

The KG annotation is large:

```text
dataset/mimic_kg_annotation.json
```

Training uses a compact cache instead:

```text
dataset/mimic_kg_entity_labels.pt
```

Rebuild it after changing the KG file or label policy:

```bash
python scripts/build_kg_label_cache.py \
  --kg_annotation dataset/mimic_kg_annotation.json \
  --output dataset/mimic_kg_entity_labels.pt \
  --top_k 512 \
  --min_freq 5 \
  --mode tokens_label
```

## Full Qwen3 MIMIC Run

Use the full config after preflight/smoke passes:

```bash
python -u train.py --config configs/experiments/mimic_qwen3_4b.yaml
```

CLI values override config-file values, so this is valid:

```bash
python -u train.py \
  --config configs/experiments/mimic_qwen3_4b.yaml \
  --devices 1 \
  --strategy auto \
  --savedmodel_path ./save/mimic_cxr/manual_debug
```

## What Changed

- `train.py` supports JSON/YAML experiment configs.
- `train.py` runs preflight checks before loading large models.
- `models/R2GenGPT.py` now uses one prompt/BOS input builder for train/validation/test.
- `models/R2GenGPT.py` supports a Q-Former visual projector through `projector_type: qformer`.
- `models/R2GenGPT.py` supports KG entity multi-label auxiliary loss through `use_kg_label_loss: true`.
- `models/BranchGPT.py` still saves checkpoints in the same path/name format, but includes provenance metadata inside the `.pth`.
- Dataset image errors include the sample id and full missing path.
- Qwen LoRA target modules are set to attention only: `q_proj,k_proj,v_proj,o_proj`.
- Full training configs can use `metric_val_batches` for lightweight validation during fit while preserving full validation/test settings.
