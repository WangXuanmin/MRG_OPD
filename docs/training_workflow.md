# Dino+LLaMA-refactor Training Workflow

This document records the recommended run order for the refactor copy at:

```bash
/public/home/ai_user_5/wxm/Dino+LLaMA-refactor
```

All commands below should be run inside the Jupyter/GPU terminal, not from the SSH login node `admin1`.

## 0. Environment

Each new terminal should load the school environment first:

```bash
cd /public/home/ai_user_5/wxm/Dino+LLaMA-refactor
source scripts/env_school.sh
```

`scripts/env_school.sh` sets the proxy variables and Java 11:

```bash
export https_proxy=http://172.19.98.250:32221
export http_proxy=http://172.19.98.250:32221
export all_proxy=socks5://172.19.98.250:32221
export JAVA_HOME=/public/home/ai_user_5/jdk-11.0.1
export PATH=$JAVA_HOME/bin:$PATH
```

## 1. Base SFT

Run the image-only SFT student first. This is the baseline student model used later by OPSD.

```bash
cd /public/home/ai_user_5/wxm/Dino+LLaMA-refactor
bash scripts/train_qformer_kg_sft_cuda0123.sh
```

This script uses:

```bash
configs/experiments/mimic_qwen3_4b.yaml
```

Expected output directory:

```bash
save/mimic_cxr/v1_qwen3_4b_instruct_refactor
```

After training, select the best checkpoint under:

```bash
save/mimic_cxr/v1_qwen3_4b_instruct_refactor/checkpoints/
```

The saved checkpoint is trainable-only and can be used as `STUDENT_DELTA` for OPSD.

## 2. Teacher Training With Label Prompt Ablation

Teacher models see the image and a fraction of KG observation labels in the prompt.

Available label fractions:

```bash
25
50
80
100
```

Run one teacher experiment:

```bash
cd /public/home/ai_user_5/wxm/Dino+LLaMA-refactor
bash scripts/train_teacher_label_ablation_cuda0123.sh 100
```

The script maps the fraction to one of:

```bash
configs/experiments/teacher_mimic_qwen3_4b_label25.yaml
configs/experiments/teacher_mimic_qwen3_4b_label50.yaml
configs/experiments/teacher_mimic_qwen3_4b_label80.yaml
configs/experiments/teacher_mimic_qwen3_4b_label100.yaml
```

Expected output directories:

```bash
save/teacher/qwen3_4b_label25
save/teacher/qwen3_4b_label50
save/teacher/qwen3_4b_label80
save/teacher/qwen3_4b_label100
```

After training, select the best teacher checkpoint under the corresponding `checkpoints/` directory. This checkpoint will be passed to OPSD as `TEACHER_CKPT`.

## 3. OPSD Training

OPSD trains a student that only sees the image, while a frozen teacher sees image + label prompt.

The implemented loss is:

```text
student SFT CE
+ opsd_kl_weight * token-level KL(student logits, teacher logits)
+ opsd_entity_weight * BCE(student entity logits, teacher entity probabilities)
```

Run OPSD after both checkpoints are available:

```bash
cd /public/home/ai_user_5/wxm/Dino+LLaMA-refactor

TEACHER_CKPT=save/teacher/qwen3_4b_label100/checkpoints/<teacher_checkpoint>.pth \
STUDENT_DELTA=save/mimic_cxr/v1_qwen3_4b_instruct_refactor/checkpoints/<sft_student_checkpoint>.pth \
bash scripts/train_opsd_cuda0123.sh
```

Default OPSD config:

```bash
configs/experiments/opsd_mimic_qwen3_4b.yaml
```

Default OPSD output directory:

```bash
save/opsd/qwen3_4b_label100_teacher
```

To use a different teacher label fraction, override `TEACHER_CONFIG` and `TEACHER_CKPT` together:

```bash
TEACHER_CONFIG=configs/experiments/teacher_mimic_qwen3_4b_label50.yaml \
TEACHER_CKPT=save/teacher/qwen3_4b_label50/checkpoints/<teacher_checkpoint>.pth \
STUDENT_DELTA=save/mimic_cxr/v1_qwen3_4b_instruct_refactor/checkpoints/<sft_student_checkpoint>.pth \
bash scripts/train_opsd_cuda0123.sh
```

## 4. Quick Checks Before Long Runs

Run a config/path preflight without loading large models:

```bash
python train.py \
  --config configs/experiments/mimic_qwen3_4b.yaml \
  --preflight_only \
  --data_sample_check 1
```

Teacher preflight:

```bash
python train.py \
  --config configs/experiments/teacher_mimic_qwen3_4b_label100.yaml \
  --preflight_only \
  --data_sample_check 1
```

OPSD preflight requires a real teacher checkpoint path:

```bash
python train_opsd.py \
  --config configs/experiments/opsd_mimic_qwen3_4b.yaml \
  --teacher_ckpt_file <teacher_checkpoint>.pth \
  --preflight_only \
  --data_sample_check 1
```

## 5. Notes

- Do not run GPU training from SSH `admin1`; use the Jupyter/GPU terminal.
- The refactor copy is independent from the original `Dino+LLaMA`, so experiments here should not pollute the original source tree.
- Current KG label cache is observation-focused:

```bash
dataset/mimic_kg_entity_labels.pt
```

- If anatomy labels are needed later, rebuild a separate KG cache with anatomy included and create a separate experiment config.
