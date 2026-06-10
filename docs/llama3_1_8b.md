# Run with Llama 3.1 8B

The training code accepts the LLM through `--llama_model`, so the Llama 3.1 switch should be done at runtime instead of hardcoding a new local path into `configs/config.py`.

Use the optimized BranchGPT script:

```bash
LLAMA_MODEL=/path/to/Llama-3.1-8B bash scripts/8-1.llama3_1_8b_run.sh
```

The script defaults to the gated Hugging Face instruct model id after logging in with a token that has access:

```bash
LLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/8-1.llama3_1_8b_run.sh
```

You can also override dataset and output paths without editing the script:

```bash
ANNOTATION=data/mimic_cxr/my_mimic_anno.json \
BASE_DIR=./data/mimic_cxr/images \
SAVEPATH=./save/mimic_cxr/v1_llama3_1_8b_shallow \
LLAMA_MODEL=/path/to/Llama-3.1-8B \
bash scripts/8-1.llama3_1_8b_run.sh
```

Notes:

- `transformers>=4.43.1` is required for Llama 3.1 support.
- The code now uses `AutoTokenizer` and `AutoModelForCausalLM`, so Llama 2 remains supported.
- Padding is derived from the tokenizer instead of assuming Llama 2 token id `0`.
- `scripts/8-1.llama3_1_8b_run.sh` uses `--model_name branchgpt`, which isolates training optimizations in `models/BranchGPT.py`.
- BranchGPT logs `val_score`, enabling `EarlyStopping` and top-k checkpointing without changing the original `R2GenGPT` training path.
- BranchGPT checkpoints are trainable-only `.pth` files. Load them with `--delta_file path/to/checkpoint.pth`; if passed through `--ckpt_file`, `train.py` redirects trainable-only files to `--delta_file`.
