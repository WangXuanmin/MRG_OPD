# Dino+LLaMA Improvement Notes

These notes are safe to keep while training is running. They do not change training code, model code, checkpoint naming, or saved outputs.

## Do Not Change During An Active Run

Avoid editing these while a training job is running from this tree:

```text
train.py
configs/config.py
models/
dataset/
lightning_tools/
scripts/*.sh used by the active run
save/
```

Editing Python source usually will not affect an already-loaded process, but a DDP restart, resumed run, or notebook reload could pick up partially changed files. Keep behavioral changes for a clean window between runs.

## Safe Backlog

1. Add a startup preflight module.
   Check `annotation`, `base_dir`, model directories, `delta_file`, and branch/Chexbert paths before loading large models or requesting GPUs.

2. Move experiment definitions out of shell scripts.
   Keep shell scripts thin and move paths/hyperparameters into versioned YAML files such as `configs/experiments/qwen3_mimic.yaml`.

3. Unify prompt/BOS handling.
   `BranchGPT` uses chat templates for validation/testing, while `R2GenGPT.forward()` still manually injects BOS. A shared prompt-input module should build train/validation/test inputs consistently.

4. Make validation cheaper during development.
   Add a smoke mode with tiny `limit_train_batches`, `limit_val_batches`, low beam size, and one-device execution.

5. Record complete checkpoint provenance.
   For each trainable-only checkpoint, save the base LLM path, vision model path, LoRA target modules, prompt mode, dataset annotation path, and git state when a repo is initialized.

6. Separate code from artifacts.
   Keep generated outputs, checkpoints, TensorBoard logs, core dumps, and caches out of version control.

## Observed Cleanup Items

The project root currently contains a large core dump:

```text
core.34057
```

Do not delete it during active debugging. If it is no longer needed, archive or remove it after confirming no one needs the crash dump.

Some defaults in `configs/config.py` still point to:

```text
/public/home/ai_user_5/wxm/MoEPP-Branch
```

The observed directory under `wxm` is:

```text
/public/home/ai_user_5/wxm/old_MoEPP-Branch
```

Treat this as a preflight failure to fix between runs, not during an active run.
