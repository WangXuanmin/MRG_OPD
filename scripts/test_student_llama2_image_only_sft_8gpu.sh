#!/usr/bin/env bash
set -euo pipefail
cd /public/home/ai_user_5/wxm/Dino+LLaMA-llama2-label-teacher
source scripts/env_school.sh
export PATH=/public/home/ai_user_5/wxm/envs/wxm_qwen_py310/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
config="configs/experiments/student_mimic_llama2_7b_image_only.yaml"
checkpoint_dir="save/student/llama2_7b_image_only_sft/checkpoints"
if [[ -n "${CHECKPOINT:-}" ]]; then
  checkpoint="${CHECKPOINT}"
else
  checkpoint="$(ls -1 "${checkpoint_dir}"/*.pth 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "No Llama2 image-only SFT student checkpoint found under ${checkpoint_dir}." >&2
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
  2>&1 | tee logs/test_student_llama2_image_only_sft_8gpu.log
