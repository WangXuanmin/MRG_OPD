#!/usr/bin/env bash
set -euo pipefail
cd /public/home/ai_user_5/wxm/Dino+LLaMA-llama2-label-teacher
source scripts/env_school.sh
export PATH=/public/home/ai_user_5/wxm/envs/wxm_qwen_py310/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python -u train.py \
  --config configs/experiments/teacher_mimic_llama2_7b_label_priv_5ep.yaml \
  2>&1 | tee -a logs/teacher_llama2_label_priv_5ep_8gpu.log
