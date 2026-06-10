#!/usr/bin/env bash
set -euo pipefail
cd /public/home/ai_user_5/wxm/Dino+LLaMA-llama2-label-teacher
source scripts/env_school.sh
export PATH=/public/home/ai_user_5/wxm/envs/wxm_qwen_py310/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
config="configs/experiments/opsd_mimic_llama2_7b_label_teacher_image_student.yaml"
teacher_config="configs/experiments/teacher_mimic_llama2_7b_label_priv_5ep.yaml"
teacher_delta="${TEACHER_DELTA:-save/teacher/llama2_7b_label_priv_5ep/checkpoints/trainable_epoch1_step4232_val0.187637_bleu0.221698_cider0.131850.pth}"
student_delta="${STUDENT_DELTA:-save/opsd/llama2_7b_label_teacher_image_student/checkpoints/opsd_student_epoch1_step33849_val0.149230_bleu0.143375_cider0.082841.pth}"
args=(
  --config "${config}"
  --teacher_config "${teacher_config}"
  --teacher_delta_file "${teacher_delta}"
  --batch_size "${BATCH_SIZE:-1}"
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-2}"
)
if [[ -n "${student_delta}" ]]; then
  args+=(--student_delta_file "${student_delta}")
fi
mkdir -p logs
python -u train_opsd.py "${args[@]}" \
  2>&1 | tee -a logs/opsd_llama2_label_teacher_image_student_8gpu.log
