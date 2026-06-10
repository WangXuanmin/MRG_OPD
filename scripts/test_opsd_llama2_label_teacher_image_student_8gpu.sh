#!/usr/bin/env bash
set -euo pipefail
cd /public/home/ai_user_5/wxm/Dino+LLaMA-llama2-label-teacher
source scripts/env_school.sh
export PATH=/public/home/ai_user_5/wxm/envs/wxm_qwen_py310/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
config="configs/experiments/opsd_mimic_llama2_7b_label_teacher_image_student.yaml"
checkpoint_dir="save/opsd/llama2_7b_label_teacher_image_student/checkpoints"
if [[ -n "${CHECKPOINT:-}" ]]; then
  checkpoint="${CHECKPOINT}"
else
  checkpoint="$(
    for file in "${checkpoint_dir}"/opsd_student_*.pth; do
      [[ -f "${file}" ]] || continue
      name="$(basename "${file}")"
      score="${name#*_val}"
      score="${score%%_bleu*}"
      printf '%s %s\n' "${score}" "${file}"
    done | sort -nr | head -n 1 | cut -d ' ' -f 2-
  )"
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "No Llama2 OPSD student checkpoint found under ${checkpoint_dir}." >&2
  exit 2
fi
mkdir -p logs
python -u train.py \
  --config "${config}" \
  --delta_file "${checkpoint}" \
  --test \
  --devices 8 \
  --strategy ddp \
  --test_batch_size "${TEST_BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-8}" \
  2>&1 | tee logs/test_opsd_llama2_label_teacher_image_student_8gpu.log
