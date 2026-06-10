import json
import os
import sys
from pprint import pprint
from configs.config import parser


def _explicit_cli_dests(argv):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def _parse_scalar(value):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if "," in value and not (value.startswith("[") and value.endswith("]")):
        return [_parse_scalar(part) for part in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_config_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()

    if config_path.endswith(".json"):
        return json.loads(text)

    try:
        import yaml
    except ImportError:
        yaml = None

    if yaml is not None:
        data = yaml.safe_load(text)
        return data or {}

    data = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def _apply_config_file(args, argv):
    if args.config is None:
        return args

    config = _load_config_file(args.config)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {args.config}")

    explicit = _explicit_cli_dests(argv)
    actions_by_dest = {action.dest: action for action in parser._actions}
    valid_dests = set(actions_by_dest)
    unknown = sorted(set(config) - valid_dests)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {args.config}: {', '.join(unknown)}")

    for key, value in config.items():
        if key not in explicit:
            action = actions_by_dest[key]
            if isinstance(value, str) and action.type in {int, float}:
                value = action.type(value)
            elif isinstance(value, str) and action.type is not None and action.type is not str:
                value = action.type(value)
            setattr(args, key, value)
    return args


def _path_exists(path):
    return path is not None and os.path.exists(path)


def _load_annotation(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for split in ("train", "val", "test"):
        if split not in data:
            raise ValueError(f"Annotation missing split '{split}': {path}")
        if not isinstance(data[split], list):
            raise ValueError(f"Annotation split '{split}' must be a list: {path}")
    return data


def _validate_image_samples(args, annotation):
    max_samples = max(0, args.data_sample_check)
    if max_samples == 0:
        return
    for split in ("train", "val", "test"):
        for item in annotation[split][:max_samples]:
            if "id" not in item:
                raise ValueError(f"Annotation item in split '{split}' missing id")
            if "image_path" not in item or not isinstance(item["image_path"], list):
                raise ValueError(f"Annotation item {item.get('id')} missing image_path list")
            for rel_path in item["image_path"]:
                image_path = os.path.join(args.base_dir, rel_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Missing image for {item.get('id')}: {image_path}")


def preflight(args):
    required_paths = {
        "annotation": args.annotation,
        "base_dir": args.base_dir,
        "vision_model": args.vision_model,
        "llama_model": args.llama_model,
    }
    if args.delta_file is not None:
        required_paths["delta_file"] = args.delta_file
    if args.ckpt_file is not None:
        required_paths["ckpt_file"] = args.ckpt_file
    if args.use_kg_label_loss:
        required_paths["kg_label_cache"] = args.kg_label_cache
    if getattr(args, 'use_kg_tokens', False):
        required_paths["kg_token_cache"] = args.kg_token_cache

    if args.require_branch_assets:
        required_paths.update({
            "branch_vocab_path": args.branch_vocab_path,
            "branch_model_path": args.branch_model_path,
        })

    missing = [f"{name}={path}" for name, path in required_paths.items() if not _path_exists(path)]
    if missing:
        raise FileNotFoundError("Preflight missing required path(s):\n  " + "\n  ".join(missing))

    annotation = None
    if args.data_sample_check > 0:
        annotation = _load_annotation(args.annotation)
        _validate_image_samples(args, annotation)
    return {
        "splits": {split: len(annotation[split]) for split in ("train", "val", "test")} if annotation else "not scanned",
        "checked_samples_per_split": args.data_sample_check,
    }


def write_provenance(args, preflight_result):
    if not args.write_provenance:
        return
    os.makedirs(args.savedmodel_path, exist_ok=True)
    payload = {
        "args": vars(args),
        "preflight": preflight_result,
        "cwd": os.getcwd(),
        "command": sys.argv,
    }
    with open(os.path.join(args.savedmodel_path, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def train(args):
    import torch
    import pytorch_lightning as pl
    from dataset.data_module import DataModule
    from lightning_tools.callbacks import add_callbacks
    from models.BranchGPT import BranchGPT
    from models.R2GenGPT import R2GenGPT

    dm = DataModule(args)
    callbacks = add_callbacks(args)
    fit_limit_val_batches = (
        args.metric_val_batches
        if args.metric_val_batches is not None and not args.test and not args.validate
        else args.limit_val_batches
    )

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = fit_limit_val_batches,
        limit_train_batches = args.limit_train_batches,
        limit_test_batches = args.limit_test_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    model_cls = BranchGPT if args.model_name == 'branchgpt' else R2GenGPT

    if args.ckpt_file is not None and args.model_name == 'branchgpt':
        ckpt = torch.load(args.ckpt_file, map_location='cpu')
        if ckpt.get('trainable_only', False):
            args.delta_file = args.ckpt_file
            args.ckpt_file = None

    if args.ckpt_file is not None:
        model = model_cls.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = model_cls(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    args = _apply_config_file(args, sys.argv[1:])
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    preflight_result = preflight(args)
    pprint({"preflight": preflight_result})
    write_provenance(args, preflight_result)
    if args.preflight_only:
        return
    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()
