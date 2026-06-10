import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
# import lightning.pytorch as pl
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import AutoModel, SwinModel
from peft import get_peft_model, LoraConfig, TaskType


def gather_generation_outputs(step_outputs):
    gathered = [step_outputs]
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, step_outputs)

    ref, hypo = {}, {}
    for rank_outputs in gathered:
        for item in rank_outputs:
            for sample_id, sample_ref, sample_hypo in zip(item['id'], item['ref'], item['hypo']):
                ref[sample_id] = [sample_ref]
                hypo[sample_id] = [sample_hypo]
    return ref, hypo


class VisualQFormerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, visual_tokens):
        attended, _ = self.cross_attn(
            query=query_tokens,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )
        query_tokens = self.attn_norm(query_tokens + self.dropout(attended))
        query_tokens = self.ffn_norm(query_tokens + self.dropout(self.ffn(query_tokens)))
        return query_tokens


class KGTokenEncoder(nn.Module):
    def __init__(
            self,
            token_embed,
            llama_hidden_size,
            visual_hidden_size,
            type_vocab_size,
            status_vocab_size,
            relation_vocab_size,
            dropout,
    ):
        super().__init__()
        self.token_embed = token_embed
        self.text_proj = nn.Linear(llama_hidden_size, visual_hidden_size)
        self.type_embed = nn.Embedding(type_vocab_size, visual_hidden_size)
        self.status_embed = nn.Embedding(status_vocab_size, visual_hidden_size)
        self.relation_embed = nn.Embedding(relation_vocab_size, visual_hidden_size)
        self.norm = nn.LayerNorm(visual_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            kg_token_ids,
            kg_token_mask,
            kg_type_ids,
            kg_status_ids,
            kg_relation_ids,
            kg_attention_mask,
            dtype=None,
    ):
        kg_token_ids = kg_token_ids.long()
        kg_token_mask = kg_token_mask.to(kg_token_ids.device).unsqueeze(-1).float()
        token_embeds = self.token_embed(kg_token_ids)
        denom = kg_token_mask.sum(dim=2).clamp_min(1.0)
        text_embeds = (token_embeds.float() * kg_token_mask).sum(dim=2) / denom
        node_embeds = self.text_proj(text_embeds)
        node_embeds = node_embeds + self.type_embed(kg_type_ids.long())
        node_embeds = node_embeds + self.status_embed(kg_status_ids.long())
        node_embeds = node_embeds + self.relation_embed(kg_relation_ids.long())
        node_embeds = self.norm(node_embeds)
        node_embeds = self.dropout(node_embeds)
        node_mask = kg_attention_mask.unsqueeze(-1).to(node_embeds.device).float()
        node_embeds = node_embeds * node_mask
        if dtype is not None:
            node_embeds = node_embeds.to(dtype=dtype)
        return node_embeds


class VisualQFormer(nn.Module):
    def __init__(self, visual_hidden_size, llama_hidden_size, num_query_tokens, num_layers, num_heads, dropout):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, visual_hidden_size) * 0.02)
        self.layers = nn.ModuleList([
            VisualQFormerBlock(visual_hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(visual_hidden_size)
        self.output_proj = nn.Linear(visual_hidden_size, llama_hidden_size)

    def forward(self, visual_tokens):
        batch_size = visual_tokens.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens, visual_tokens)
        query_tokens = self.output_norm(query_tokens)
        return self.output_proj(query_tokens)


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        vision_model_type = getattr(args, 'vision_model_type', 'swin')
        if vision_model_type == 'swin':
            self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        else:
            self.visual_encoder = AutoModel.from_pretrained(args.vision_model)

        if args.vis_use_lora:
            vis_lora_target_modules = getattr(args, 'vis_lora_target_modules', 'query,value')
            target_modules = [module.strip() for module in vis_lora_target_modules.split(',') if module.strip()]
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=target_modules,
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=True)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_pad_token_id = self.llama_tokenizer.pad_token_id
        self.llama_bos_token_id = self.llama_tokenizer.bos_token_id
        if self.llama_bos_token_id is None:
            self.llama_bos_token_id = self.llama_tokenizer.eos_token_id
        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            llm_lora_target_modules = getattr(args, 'llm_lora_target_modules', '')
            llm_target_modules = [module.strip() for module in llm_lora_target_modules.split(',') if module.strip()]
            lora_kwargs = {'target_modules': llm_target_modules} if llm_target_modules else {}
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                **lora_kwargs,
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        visual_hidden_size = self._get_visual_hidden_size()
        llama_hidden_size = self.llama_model.config.hidden_size
        self.kg_token_encoder = None
        if getattr(args, 'use_kg_tokens', False):
            self.kg_token_encoder = KGTokenEncoder(
                token_embed=self.embed_tokens,
                llama_hidden_size=llama_hidden_size,
                visual_hidden_size=visual_hidden_size,
                type_vocab_size=args.kg_type_vocab_size,
                status_vocab_size=args.kg_status_vocab_size,
                relation_vocab_size=args.kg_relation_vocab_size,
                dropout=args.kg_token_dropout,
            )
            print(
                'Loading KG token encoder: '
                f'max_nodes={args.kg_max_nodes}, '
                f'max_text_tokens={args.kg_max_text_tokens}'
            )

        if getattr(args, 'projector_type', 'linear') == 'qformer':
            self.visual_projector = VisualQFormer(
                visual_hidden_size=visual_hidden_size,
                llama_hidden_size=llama_hidden_size,
                num_query_tokens=args.qformer_num_query_tokens,
                num_layers=args.qformer_num_layers,
                num_heads=args.qformer_num_heads,
                dropout=args.qformer_dropout,
            )
            print(
                'Loading Q-Former projector: '
                f'{args.qformer_num_query_tokens} queries, '
                f'{args.qformer_num_layers} layers, '
                f'{args.qformer_num_heads} heads'
            )
        else:
            self.visual_projector = nn.Linear(visual_hidden_size, llama_hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        if getattr(args, 'use_kg_label_loss', False):
            self.kg_label_classifier = nn.Linear(llama_hidden_size, args.kg_num_labels)
            self.kg_label_loss = nn.BCEWithLogitsLoss()
        else:
            self.kg_label_classifier = None
            self.kg_label_loss = None
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def _get_visual_hidden_size(self):
        if hasattr(self.visual_encoder, 'num_features'):
            return self.visual_encoder.num_features
        if hasattr(self.visual_encoder, 'config') and hasattr(self.visual_encoder.config, 'hidden_size'):
            return self.visual_encoder.config.hidden_size
        raise AttributeError('Cannot infer visual encoder hidden size. Add a projection input size for this vision model.')

    def _use_manual_bos(self):
        return not getattr(self.hparams, 'use_chat_template', False)

    def _format_prompt(self, label_prompt=None):
        extra = f' {label_prompt}' if label_prompt else ''
        return f'Human: <Img><ImageHere></Img> {self.prompt}{extra} \nAssistant:'
    
    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images, kg_inputs=None):
        image_embeds = []
        for image in images:
            device = image.device
            visual_outputs = self.visual_encoder(image)
            if self.hparams.global_only:
                if hasattr(visual_outputs, 'pooler_output') and visual_outputs.pooler_output is not None:
                    image_embed = visual_outputs.pooler_output.unsqueeze(1).to(device)
                else:
                    image_embed = visual_outputs.last_hidden_state[:, 0:1, :].to(device)
            else:
                image_embed = visual_outputs.last_hidden_state.to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        if kg_inputs is not None:
            if self.kg_token_encoder is None:
                raise RuntimeError('Received KG token inputs but use_kg_tokens is disabled for this model.')
            kg_embeds = self.kg_token_encoder(
                kg_inputs['kg_token_ids'].to(image_embeds.device),
                kg_inputs['kg_token_mask'].to(image_embeds.device),
                kg_inputs['kg_type_ids'].to(image_embeds.device),
                kg_inputs['kg_status_ids'].to(image_embeds.device),
                kg_inputs['kg_relation_ids'].to(image_embeds.device),
                kg_inputs['kg_attention_mask'].to(image_embeds.device),
                dtype=image_embeds.dtype,
            )
            image_embeds = torch.cat([image_embeds, kg_embeds], dim=1)
        inputs_llama = self.visual_projector(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        visual_summary = inputs_llama.mean(dim=1)
        return inputs_llama, atts_llama, visual_summary


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

    def build_prompt_inputs(self, image, include_text=None, label_prompts=None, kg_inputs=None, return_visual_summary=False):
        img_embeds, atts_img, visual_summary = self.encode_img(image, kg_inputs=kg_inputs)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, label_prompts=label_prompts)

        inputs_embeds = img_embeds
        attention_mask = atts_img
        target_prefix_len = atts_img.shape[1]

        if self._use_manual_bos():
            batch_size = img_embeds.shape[0]
            bos = torch.ones(
                [batch_size, 1],
                dtype=atts_img.dtype,
                device=atts_img.device,
            ) * self.llama_bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, attention_mask], dim=1)
            target_prefix_len += 1

        if include_text is None:
            if return_visual_summary:
                return inputs_embeds, attention_mask, target_prefix_len, visual_summary
            return inputs_embeds, attention_mask, target_prefix_len

        text = [t + self.end_sym for t in include_text]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, to_regress_tokens.attention_mask], dim=1)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_pad_token_id, -100
        )
        empty_targets = torch.ones(
            [targets.shape[0], target_prefix_len],
            dtype=torch.long,
            device=image[0].device,
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        if return_visual_summary:
            return inputs_embeds, attention_mask, targets, visual_summary
        return inputs_embeds, attention_mask, targets


    def _sample_label_prompts(self, samples):
        if not getattr(self.hparams, 'use_label_prompt', False):
            return None
        return samples.get("label_prompt")

    def _sample_kg_inputs(self, samples):
        if not getattr(self.hparams, 'use_kg_tokens', False):
            return None
        required = [
            'kg_token_ids',
            'kg_token_mask',
            'kg_type_ids',
            'kg_status_ids',
            'kg_relation_ids',
            'kg_attention_mask',
        ]
        missing = [key for key in required if key not in samples]
        if missing:
            raise KeyError(f'use_kg_tokens=true but batch is missing: {missing}')
        return {key: samples[key] for key in required}

    def forward(self, samples, return_logits=False):
        image = samples["image"]
        self.llama_tokenizer.padding_side = "right"
        inputs_embeds, attention_mask, targets, visual_summary = self.build_prompt_inputs(
            image,
            include_text=samples["input_text"],
            label_prompts=self._sample_label_prompts(samples),
            kg_inputs=self._sample_kg_inputs(samples),
            return_visual_summary=True,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        lm_loss = outputs.loss
        loss = lm_loss
        result = {"loss": loss, "lm_loss": lm_loss.detach()}
        if return_logits:
            result.update({
                "logits": outputs.logits,
                "targets": targets,
                "attention_mask": attention_mask,
                "visual_summary": visual_summary,
            })
        if self.kg_label_classifier is not None and "kg_labels" in samples:
            kg_logits = self.kg_label_classifier(visual_summary.float())
            kg_labels = samples["kg_labels"].to(kg_logits.device)
            kg_loss = self.kg_label_loss(kg_logits, kg_labels)
            loss = loss + self.hparams.kg_label_loss_weight * kg_loss
            with torch.no_grad():
                kg_pred = (torch.sigmoid(kg_logits) >= self.hparams.kg_label_threshold).float()
                kg_match = (kg_pred == kg_labels).float().mean()
            result.update({
                "loss": loss,
                "kg_label_loss": kg_loss.detach(),
                "kg_label_match": kg_match.detach(),
            })
            if return_logits:
                result["kg_logits"] = kg_logits
        return result

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        inputs_embeds, attention_mask, _ = self.build_prompt_inputs(
            image,
            label_prompts=self._sample_label_prompts(samples),
            kg_inputs=self._sample_kg_inputs(samples),
        )

        generation_max_length = inputs_embeds.shape[1] + self.hparams.max_new_tokens
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_length=generation_max_length,
            repetition_penalty=self.hparams.repetition_penalty,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.llama_pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        output_text = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        if self.end_sym:
            output_text = output_text.split(self.end_sym)[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_step_outputs.clear()
            return
        ref, hypo = gather_generation_outputs(self.val_step_outputs)
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        if self.trainer.is_global_zero:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        inputs_embeds, attention_mask, _ = self.build_prompt_inputs(
            image,
            label_prompts=self._sample_label_prompts(samples),
            kg_inputs=self._sample_kg_inputs(samples),
        )

        generation_max_length = inputs_embeds.shape[1] + self.hparams.max_new_tokens
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_length=generation_max_length,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.llama_pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo = gather_generation_outputs(self.test_step_outputs)
        eval_res = self.score(ref=ref,hypo=hypo)

        if self.trainer.is_global_zero:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
            json.dump(eval_res, open(os.path.join(result_folder, 'test_metrics.json'), 'w'), indent=2)
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
