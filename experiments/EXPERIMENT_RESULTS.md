# Key Experiment Results

This export intentionally excludes large model checkpoints (`*.pth`) and data/cache tensors (`*.pt`).
Checkpoint filenames and sizes are preserved under `experiments/checkpoints_manifest/`.

## Teacher: Llama2-7B label-privileged

Best tested checkpoint on remote:
`save/teacher/llama2_7b_label_priv_5ep/checkpoints/trainable_epoch1_step4232_val0.187637_bleu0.221698_cider0.131850.pth`

Independent test metrics:
- Bleu_1: 0.481153169278088
- Bleu_2: 0.33273335403601706
- Bleu_3: 0.2434033430954826
- Bleu_4: 0.18565104714216368
- ROUGE_L: 0.30480572087457186
- METEOR: 0.2002568658855177
- CIDEr: 0.19786219642716874

## Student: Llama2-7B image-only SFT

Available checkpoint on remote:
`save/student/llama2_7b_image_only_sft/checkpoints/trainable_epoch0_step2116_val0.138560_bleu0.132381_cider0.047953.pth`

## OPSD Student: label-teacher -> image-only student

Best validation checkpoint on remote:
`save/opsd/llama2_7b_label_teacher_image_student/checkpoints/opsd_student_epoch1_step33849_val0.149230_bleu0.143375_cider0.082841.pth`

OPSD uses frozen teacher forward on each batch, token-level KL distillation over valid next-token positions, plus optional entity probability distillation.
