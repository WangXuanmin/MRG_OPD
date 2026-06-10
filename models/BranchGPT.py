import math
import os
import json

import torch

from models.R2GenGPT import R2GenGPT, gather_generation_outputs


class BranchGPT(R2GenGPT):
    """
    Training-oriented R2GenGPT variant.

    Keeps the original multimodal path intact and isolates optimization changes:
    chat-template prompt wrapping, val_score logging, and step-level warmup+cosine LR.
    """
    def __init__(self, args):
        super().__init__(args)
        self.prompt = getattr(args, 'report_prompt', self.prompt)
        self.best_trainable_checkpoints = []
        if getattr(args, 'end_sym', None) == 'auto':
            self.end_sym = self.llama_tokenizer.eos_token or ''

    def _format_prompt(self, label_prompt=None):
        extra = f' {label_prompt}' if label_prompt else ''
        user_content = f'<Img><ImageHere></Img> {self.prompt}{extra}'
        chat_template = getattr(self.llama_tokenizer, 'chat_template', None)
        if getattr(self.hparams, 'use_chat_template', False) and chat_template:
            messages = [{'role': 'user', 'content': user_content}]
            return self.llama_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f'Human: {user_content} \nAssistant:'

    def prompt_wrap(self, img_embeds, atts_img, label_prompts=None):
        batch_size = img_embeds.shape[0]
        if label_prompts is None:
            label_prompts = [None] * batch_size
        elif isinstance(label_prompts, str):
            label_prompts = [label_prompts] * batch_size

        wrapped = []
        masks = []
        for sample_idx, label_prompt in enumerate(label_prompts):
            prompt = self._format_prompt(label_prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)
            sample_embeds = torch.cat([
                p_before_embeds,
                img_embeds[sample_idx:sample_idx + 1],
                p_after_embeds,
            ], dim=1)
            wrapped.append(sample_embeds.squeeze(0))
            masks.append(torch.ones(sample_embeds.shape[1], dtype=atts_img.dtype, device=img_embeds.device))

        max_len = max(item.shape[0] for item in wrapped)
        hidden = img_embeds.shape[-1]
        wrapped_img_embeds = img_embeds.new_zeros(batch_size, max_len, hidden)
        wrapped_atts_img = atts_img.new_zeros(batch_size, max_len)
        for sample_idx, sample_embeds in enumerate(wrapped):
            seq_len = sample_embeds.shape[0]
            wrapped_img_embeds[sample_idx, :seq_len] = sample_embeds
            wrapped_atts_img[sample_idx, :seq_len] = masks[sample_idx]
        return wrapped_img_embeds, wrapped_atts_img

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_step_outputs.clear()
            return
    
        ref, hypo = gather_generation_outputs(self.val_step_outputs)
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight
        self.log('val_score', val_score, sync_dist=True, logger=True, prog_bar=True)

        if self.trainer.is_global_zero:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print({**eval_res, 'val_score': val_score})

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res, val_score)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def save_checkpoint(self, eval_res, val_score=None):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        trainable_params = {
            k: v.requires_grad for k, v in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in trainable_params:
                del state_dict[k]

        if val_score is None:
            val_score = 0
            for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
                val_score += eval_res[score_type] * weight

        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
            "val_score": val_score,
            "metrics": eval_res,
            "trainable_only": True,
            "provenance": {
                "llama_model": getattr(self.hparams, "llama_model", None),
                "vision_model": getattr(self.hparams, "vision_model", None),
                "vision_model_type": getattr(self.hparams, "vision_model_type", None),
                "annotation": getattr(self.hparams, "annotation", None),
                "base_dir": getattr(self.hparams, "base_dir", None),
                "use_chat_template": getattr(self.hparams, "use_chat_template", None),
                "end_sym": getattr(self.hparams, "end_sym", None),
                "llm_use_lora": getattr(self.hparams, "llm_use_lora", None),
                "llm_lora_target_modules": getattr(self.hparams, "llm_lora_target_modules", None),
                "vis_use_lora": getattr(self.hparams, "vis_use_lora", None),
                "vis_lora_target_modules": getattr(self.hparams, "vis_lora_target_modules", None),
                "projector_type": getattr(self.hparams, "projector_type", None),
                "qformer_num_query_tokens": getattr(self.hparams, "qformer_num_query_tokens", None),
                "qformer_num_layers": getattr(self.hparams, "qformer_num_layers", None),
                "qformer_num_heads": getattr(self.hparams, "qformer_num_heads", None),
                "use_kg_label_loss": getattr(self.hparams, "use_kg_label_loss", None),
                "kg_label_cache": getattr(self.hparams, "kg_label_cache", None),
                "kg_num_labels": getattr(self.hparams, "kg_num_labels", None),
                "kg_label_loss_weight": getattr(self.hparams, "kg_label_loss_weight", None),
                "use_label_prompt": getattr(self.hparams, "use_label_prompt", None),
                "label_prompt_fraction": getattr(self.hparams, "label_prompt_fraction", None),
                "label_prompt_max_items": getattr(self.hparams, "label_prompt_max_items", None),
                "use_kg_tokens": getattr(self.hparams, "use_kg_tokens", None),
                "kg_token_cache": getattr(self.hparams, "kg_token_cache", None),
                "kg_max_nodes": getattr(self.hparams, "kg_max_nodes", None),
                "kg_max_text_tokens": getattr(self.hparams, "kg_max_text_tokens", None),
            },
        }
        checkpoint_dir = os.path.join(self.hparams.savedmodel_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_to = os.path.join(
            checkpoint_dir,
            "trainable_epoch{}_step{}_val{:.6f}_bleu{:.6f}_cider{:.6f}.pth".format(
                current_epoch, global_step, val_score, eval_res['Bleu_4'], eval_res['CIDEr']
            ),
        )
        self.print("Saving trainable-only checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

        self.best_trainable_checkpoints.append((val_score, save_to))
        self.best_trainable_checkpoints.sort(key=lambda x: x[0], reverse=True)
        save_top_k = max(1, getattr(self.hparams, 'save_top_k', 1))
        for _, stale_path in self.best_trainable_checkpoints[save_top_k:]:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        self.best_trainable_checkpoints = self.best_trainable_checkpoints[:save_top_k]

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith('bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.hparams.learning_rate,
        )

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = self.hparams.warmup_steps
        if warmup_steps <= 0:
            warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0
        min_lr_ratio = self.hparams.min_lr / self.hparams.learning_rate

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
