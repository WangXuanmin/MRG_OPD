import copy
import json
import math
import os
import sys
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.BranchGPT import BranchGPT
from models.R2GenGPT import R2GenGPT, gather_generation_outputs
from train import _apply_config_file, preflight, write_provenance


def _load_args(argv):
    args = parser.parse_args(argv)
    return _apply_config_file(args, argv)


def _model_class(args):
    return BranchGPT if args.model_name == 'branchgpt' else R2GenGPT


def _load_model(args, ckpt_file=None, delta_file=None):
    model_cls = _model_class(args)
    args = copy.deepcopy(args)
    if delta_file is not None:
        args.delta_file = delta_file
        args.ckpt_file = None
        return model_cls(args)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        if isinstance(ckpt, dict) and ckpt.get('trainable_only', False):
            args.delta_file = ckpt_file
            args.ckpt_file = None
            return model_cls(args)
        return model_cls.load_from_checkpoint(ckpt_file, strict=False)
    return model_cls(args)


def _masked_report_logits(outputs, temperature, max_tokens=0):
    logits = outputs["logits"][:, :-1, :] / temperature
    targets = outputs["targets"][:, 1:]
    mask = targets.ne(-100)
    flat_logits = logits[mask]
    if max_tokens and flat_logits.shape[0] > max_tokens:
        flat_logits = flat_logits[:max_tokens]
    return flat_logits


class OPSDModule(pl.LightningModule):
    def __init__(self, args, teacher_args):
        super().__init__()
        self.args = args
        self.teacher_args = teacher_args
        self.save_hyperparameters({
            "student": vars(args),
            "teacher": vars(teacher_args),
        })

        student_delta = args.student_delta_file or args.delta_file
        student_ckpt = args.student_ckpt_file or args.ckpt_file
        self.student = _load_model(args, ckpt_file=student_ckpt, delta_file=student_delta)
        teacher_delta = args.teacher_delta_file
        if teacher_delta is None and args.teacher_ckpt_file is None:
            teacher_delta = teacher_args.delta_file
        self.teacher = _load_model(
            teacher_args,
            ckpt_file=args.teacher_ckpt_file,
            delta_file=teacher_delta,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.val_step_outputs = []
        self.val_score = 0.0
        self.best_trainable_checkpoints = []
        self.entity_distill = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        self.student.llama_tokenizer.padding_side = "right"
        self.teacher.llama_tokenizer.padding_side = "right"

        student_outputs = self.student(batch, return_logits=True)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch, return_logits=True)

        temperature = self.args.opsd_temperature
        student_logits = _masked_report_logits(
            student_outputs,
            temperature,
            self.args.opsd_max_kl_tokens,
        )
        teacher_logits = _masked_report_logits(
            teacher_outputs,
            temperature,
            self.args.opsd_max_kl_tokens,
        )
        if student_logits.shape[0] != teacher_logits.shape[0]:
            keep = min(student_logits.shape[0], teacher_logits.shape[0])
            student_logits = student_logits[:keep]
            teacher_logits = teacher_logits[:keep]

        kl_loss = F.kl_div(
            F.log_softmax(student_logits.float(), dim=-1),
            F.softmax(teacher_logits.float(), dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        loss = student_outputs["loss"] + self.args.opsd_kl_weight * kl_loss
        logs = {
            "loss": loss,
            "student_loss": student_outputs["loss"].detach(),
            "opsd_kl_loss": kl_loss.detach(),
        }

        if "kg_logits" in student_outputs and "kg_logits" in teacher_outputs:
            teacher_entity_probs = torch.sigmoid(teacher_outputs["kg_logits"].detach().float())
            entity_loss = self.entity_distill(
                student_outputs["kg_logits"].float(),
                teacher_entity_probs,
            )
            loss = loss + self.args.opsd_entity_weight * entity_loss
            logs.update({
                "loss": loss,
                "opsd_entity_loss": entity_loss.detach(),
            })

        self.log_dict(logs, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, samples, batch_idx):
        self.student.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.student.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.student.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        inputs_embeds, attention_mask, _ = self.student.build_prompt_inputs(image)

        generation_max_length = inputs_embeds.shape[1] + self.student.hparams.max_new_tokens
        outputs = self.student.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.student.hparams.beam_size,
            do_sample=self.student.hparams.do_sample,
            min_new_tokens=self.student.hparams.min_new_tokens,
            max_length=generation_max_length,
            repetition_penalty=self.student.hparams.repetition_penalty,
            no_repeat_ngram_size=self.student.hparams.no_repeat_ngram_size,
            length_penalty=self.student.hparams.length_penalty,
            temperature=self.student.hparams.temperature,
            pad_token_id=self.student.llama_pad_token_id,
            eos_token_id=self.student.llama_tokenizer.eos_token_id,
        )
        hypo = [self.student.decode(i) for i in outputs]
        ref = [self.student.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_step_outputs.clear()
            return
        ref, hypo = gather_generation_outputs(self.val_step_outputs)
        eval_res = self.student.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        val_score = 0
        for score_type, weight in zip(self.student.hparams.scorer_types, self.student.hparams.weights):
            val_score += eval_res[score_type] * weight
        self.log('val_score', val_score, sync_dist=True, logger=True, prog_bar=True)

        if self.trainer.is_global_zero:
            result_folder = os.path.join(self.args.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print({**eval_res, 'val_score': val_score})

        if self.trainer.local_rank == 0 and val_score > self.val_score:
            self.save_student_checkpoint(eval_res, val_score)
            self.val_score = val_score
        self.val_step_outputs.clear()

    def save_student_checkpoint(self, eval_res, val_score):
        trainable_params = {
            k: v.requires_grad for k, v in self.student.named_parameters() if v.requires_grad
        }
        state_dict = self.student.state_dict()
        for key in list(state_dict.keys()):
            if key not in trainable_params:
                del state_dict[key]

        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        save_obj = {
            "model": state_dict,
            "config": self.student.hparams,
            "epoch": current_epoch,
            "step": global_step,
            "val_score": val_score,
            "metrics": eval_res,
            "trainable_only": True,
            "opsd": {
                "teacher_config": self.args.teacher_config,
                "teacher_ckpt_file": self.args.teacher_ckpt_file,
                "teacher_delta_file": self.args.teacher_delta_file,
                "opsd_kl_weight": self.args.opsd_kl_weight,
                "opsd_entity_weight": self.args.opsd_entity_weight,
                "opsd_temperature": self.args.opsd_temperature,
            },
        }
        checkpoint_dir = os.path.join(self.args.savedmodel_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_to = os.path.join(
            checkpoint_dir,
            "opsd_student_epoch{}_step{}_val{:.6f}_bleu{:.6f}_cider{:.6f}.pth".format(
                current_epoch, global_step, val_score, eval_res['Bleu_4'], eval_res['CIDEr']
            ),
        )
        self.print(f"Saving OPSD student checkpoint at step {global_step} to {save_to}.")
        torch.save(save_obj, save_to)

        self.best_trainable_checkpoints.append((val_score, save_to))
        self.best_trainable_checkpoints.sort(key=lambda x: x[0], reverse=True)
        save_top_k = max(1, getattr(self.args, 'save_top_k', 1))
        for _, stale_path in self.best_trainable_checkpoints[save_top_k:]:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        self.best_trainable_checkpoints = self.best_trainable_checkpoints[:save_top_k]

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith('bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': self.args.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.args.learning_rate,
        )

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = self.args.warmup_steps
        if warmup_steps <= 0:
            warmup_steps = int(total_steps * self.args.warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0
        min_lr_ratio = self.args.min_lr / self.args.learning_rate

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'warmup_cosine_lr',
            },
        }


def main():
    args = _load_args(sys.argv[1:])
    if args.teacher_config is None:
        raise ValueError("--teacher_config is required for OPSD training")
    if args.teacher_ckpt_file is None and args.teacher_delta_file is None:
        raise ValueError("Provide --teacher_ckpt_file or --teacher_delta_file for OPSD training")

    teacher_args = _load_args(["--config", args.teacher_config])
    teacher_args.use_label_prompt = True
    args.use_label_prompt = False
    data_args = copy.deepcopy(args)
    data_args.use_label_prompt = True
    data_args.label_prompt_fraction = teacher_args.label_prompt_fraction
    data_args.label_prompt_max_items = teacher_args.label_prompt_max_items
    data_args.label_prompt_seed = teacher_args.label_prompt_seed
    data_args.label_prompt_prefix = teacher_args.label_prompt_prefix

    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint({"student": vars(args), "teacher": vars(teacher_args)})
    preflight_result = preflight(data_args)
    teacher_preflight = preflight(teacher_args)
    pprint({"student_preflight": preflight_result, "teacher_preflight": teacher_preflight})
    write_provenance(args, {"student": preflight_result, "teacher": teacher_preflight})
    if args.preflight_only:
        return

    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)

    dm = DataModule(data_args)
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
        val_check_interval=args.val_check_interval,
        limit_val_batches=fit_limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
    )
    trainer.fit(OPSDModule(args, teacher_args), datamodule=dm)


if __name__ == '__main__':
    main()
