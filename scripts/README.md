# Active Scripts

Current Llama2 label-teacher / image-only student workflow:

1. Train label-privileged teacher:
   `bash scripts/train_teacher_llama2_label_priv_5ep_8gpu.sh`
2. Test teacher:
   `bash scripts/test_teacher_llama2_label_priv_8gpu.sh`
3. Train image-only SFT student:
   `bash scripts/train_student_llama2_image_only_sft_8gpu.sh`
4. Test image-only SFT student:
   `bash scripts/test_student_llama2_image_only_sft_8gpu.sh`
5. Train OPSD student from teacher:
   `bash scripts/train_opsd_llama2_label_teacher_image_student_8gpu.sh`
6. Test OPSD student:
   `bash scripts/test_opsd_llama2_label_teacher_image_student_8gpu.sh`

Shared environment setup lives in `scripts/env_school.sh`.
`build_kg_label_cache.py` is kept because the active configs use `dataset/mimic_kg_entity_labels.pt`.

Legacy Qwen/KG-token/ablation scripts were moved to `archive/20260610_cleanup/scripts_legacy/`.
