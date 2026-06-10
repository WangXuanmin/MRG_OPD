# MRG-OPD: Label-Privileged Teacher Distillation for Medical Report Generation

This repository contains a Llama2-based medical report generation project built on MIMIC-CXR. The core idea is to train a stronger teacher model with privileged label knowledge, then distill its behavior into an image-only student model that can be used at inference time without labels.

The project focuses on a practical teacher-student pipeline:

1. Train a label-privileged Llama2 teacher.
2. Train an image-only Llama2 student baseline.
3. Distill the teacher into the image-only student with token-level OPSD.
4. Evaluate teacher and student models on MIMIC-CXR report generation metrics.

## Main Contributions

- Built a Llama2-7B report generation pipeline using Rad-DINO visual features and a Q-Former projector.
- Added label-privileged teacher training, where disease/entity labels are used as privileged knowledge during teacher learning.
- Implemented an image-only student setting, where labels are removed from the input so the student matches the real inference scenario.
- Implemented an OPSD-style student-teacher distillation framework in `train_opsd.py`.
- Added token-level KL distillation over valid next-token positions between teacher and student.
- Added optional entity probability distillation through the KG/entity classifier head.
- Organized reproducible configs, scripts, checkpoint manifests, and experiment summaries for the Llama2 teacher-student workflow.

## Architecture

The model follows a vision-language report generation design:

```text
Chest X-ray image
  -> Rad-DINO vision encoder
  -> Q-Former projector
  -> Llama2-7B chat model
  -> radiology report
```

The teacher and student share the same general architecture, but differ in input knowledge:

```text
Teacher:
  image + label privileged knowledge -> report

Student:
  image only -> report
```

During OPSD training, the teacher is frozen. For each batch, both models run forward passes, and the student is optimized with:

```text
student loss =
  report CE loss
  + token-level KL(student logits, teacher logits)
  + optional entity probability distillation
```

This is online teacher-forward distillation: teacher logits are computed during training rather than precomputed as an offline cache.

## Project Layout

```text
configs/experiments/
  teacher_mimic_llama2_7b_label_priv_5ep.yaml
  student_mimic_llama2_7b_image_only.yaml
  opsd_mimic_llama2_7b_label_teacher_image_student.yaml

scripts/
  train_teacher_llama2_label_priv_5ep_8gpu.sh
  test_teacher_llama2_label_priv_8gpu.sh
  train_student_llama2_image_only_sft_8gpu.sh
  test_student_llama2_image_only_sft_8gpu.sh
  train_opsd_llama2_label_teacher_image_student_8gpu.sh
  test_opsd_llama2_label_teacher_image_student_8gpu.sh

models/
  BranchGPT.py
  R2GenGPT.py

train.py
train_opsd.py
experiments/
  EXPERIMENT_RESULTS.md
  checkpoints_manifest/
  log_summaries/
  results/
```

Large checkpoints and dataset/cache tensors are intentionally not included in this repository. Checkpoint filenames and sizes are recorded under `experiments/checkpoints_manifest/`.

## Training

Train the label-privileged teacher:

```bash
bash scripts/train_teacher_llama2_label_priv_5ep_8gpu.sh
```

Train the image-only SFT student:

```bash
bash scripts/train_student_llama2_image_only_sft_8gpu.sh
```

Train the OPSD student from the label-privileged teacher:

```bash
bash scripts/train_opsd_llama2_label_teacher_image_student_8gpu.sh
```

The OPSD script supports overriding checkpoint paths:

```bash
TEACHER_DELTA=/path/to/teacher.pth \
STUDENT_DELTA=/path/to/student_init.pth \
bash scripts/train_opsd_llama2_label_teacher_image_student_8gpu.sh
```

## Testing

Test the teacher:

```bash
bash scripts/test_teacher_llama2_label_priv_8gpu.sh
```

Test the image-only SFT student:

```bash
bash scripts/test_student_llama2_image_only_sft_8gpu.sh
```

Test the OPSD student:

```bash
bash scripts/test_opsd_llama2_label_teacher_image_student_8gpu.sh
```

## Experiments

### Label-Privileged Teacher

Best tested teacher checkpoint:

```text
trainable_epoch1_step4232_val0.187637_bleu0.221698_cider0.131850.pth
```

Independent test results:

| Metric | Score |
|---|---:|
| Bleu_1 | 0.4812 |
| Bleu_2 | 0.3327 |
| Bleu_3 | 0.2434 |
| Bleu_4 | 0.1857 |
| ROUGE_L | 0.3048 |
| METEOR | 0.2003 |
| CIDEr | 0.1979 |

### Image-Only SFT Student

Available SFT student checkpoint:

```text
trainable_epoch0_step2116_val0.138560_bleu0.132381_cider0.047953.pth
```

This model is the image-only baseline before OPSD.

### OPSD Student

Best validation OPSD student checkpoint:

```text
opsd_student_epoch1_step33849_val0.149230_bleu0.143375_cider0.082841.pth
```

Independent test results:

| Metric | Score |
|---|---:|
| Bleu_1 | 0.3894 |
| Bleu_2 | 0.2438 |
| Bleu_3 | 0.1585 |
| Bleu_4 | 0.1083 |
| ROUGE_L | 0.2582 |
| METEOR | 0.1476 |
| CIDEr | 0.0717 |

More result files and log summaries are available under `experiments/`.

## Notes

- This repository is a lightweight export for code sharing and reproduction.
- MIMIC-CXR images, annotation files, local model weights, and checkpoints are not included.
- Large METEOR/CoreNLP runtime files from `evalcap` are omitted; see `evalcap/README_OMITTED_RUNTIME_ARTIFACTS.md`.
- Local paths in YAML configs reflect the original training server and should be changed before running in a new environment.

## Acknowledgement

This project builds on the code structure and medical report generation pipeline of [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT). We thank the R2GenGPT authors for releasing their implementation.
