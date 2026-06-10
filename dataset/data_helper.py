
import os
import json
import re
import hashlib
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from transformers import AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args,
            label_lookup=None,
            label_vocab=None,
            num_labels=0,
            kg_token_lookup=None,
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model, use_fast=False)
        self.label_lookup = label_lookup or {}
        self.label_vocab = label_vocab or []
        self.num_labels = num_labels
        self.kg_token_lookup = kg_token_lookup or {}


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report

    def _label_prompt(self, sample_id):
        if not getattr(self.args, 'use_label_prompt', False) or self.num_labels <= 0:
            return ''

        label_ids = [
            idx for idx in self.label_lookup.get(sample_id, [])
            if 0 <= idx < len(self.label_vocab)
        ]
        if not label_ids:
            return ''

        fraction = max(0.0, min(1.0, float(getattr(self.args, 'label_prompt_fraction', 1.0))))
        if fraction <= 0:
            return ''
        max_items = max(1, int(getattr(self.args, 'label_prompt_max_items', 32)))
        keep = min(len(label_ids), max(1, int(round(len(label_ids) * fraction))), max_items)

        seed = getattr(self.args, 'label_prompt_seed', 42)
        def stable_key(idx):
            raw = f'{sample_id}:{idx}:{seed}'.encode('utf-8')
            return hashlib.md5(raw).hexdigest()

        selected = sorted(label_ids, key=stable_key)[:keep]
        selected = sorted(selected)
        labels = [str(self.label_vocab[idx]).replace('_', ' ') for idx in selected]
        prefix = getattr(self.args, 'label_prompt_prefix', 'Known observations:')
        return f"{prefix} {', '.join(labels)}."


    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            full_path = os.path.join(self.args.base_dir, image_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing image for sample {features['id']}: {full_path}")
            with Image.open(full_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        if self.num_labels > 0:
            label = torch.zeros(self.num_labels, dtype=torch.float32)
            for idx in self.label_lookup.get(features['id'], []):
                if 0 <= idx < self.num_labels:
                    label[idx] = 1.0
            to_return["kg_labels"] = label
            to_return["label_prompt"] = self._label_prompt(features['id'])
        if getattr(self.args, 'use_kg_tokens', False):
            kg_item = self.kg_token_lookup.get(features['id'])
            if kg_item is None:
                max_nodes = int(getattr(self.args, 'kg_max_nodes', 64))
                max_text_tokens = int(getattr(self.args, 'kg_max_text_tokens', 8))
                kg_item = {
                    'kg_token_ids': torch.zeros(max_nodes, max_text_tokens, dtype=torch.long),
                    'kg_token_mask': torch.zeros(max_nodes, max_text_tokens, dtype=torch.long),
                    'kg_type_ids': torch.zeros(max_nodes, dtype=torch.long),
                    'kg_status_ids': torch.zeros(max_nodes, dtype=torch.long),
                    'kg_relation_ids': torch.zeros(max_nodes, dtype=torch.long),
                    'kg_attention_mask': torch.zeros(max_nodes, dtype=torch.long),
                }
            for key, value in kg_item.items():
                to_return[key] = value.clone() if torch.is_tensor(value) else torch.as_tensor(value)
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        with open(args.annotation, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        if split not in self.meta:
            raise KeyError(f"Annotation file {args.annotation} missing split '{split}'")
        self.meta = self.meta[split]
        label_lookup, label_vocab, num_labels = {}, [], 0
        kg_token_lookup = {}
        if getattr(args, 'use_kg_label_loss', False) or getattr(args, 'use_label_prompt', False):
            try:
                cache = torch.load(args.kg_label_cache, map_location='cpu', weights_only=True)
            except TypeError:
                cache = torch.load(args.kg_label_cache, map_location='cpu')
            label_lookup = cache['labels'].get(split, {})
            label_vocab = cache['vocab']
            num_labels = len(label_vocab)
            if getattr(args, 'kg_num_labels', num_labels) != num_labels:
                raise ValueError(
                    f"kg_num_labels={args.kg_num_labels} does not match cache vocab size {num_labels}"
                )
        if getattr(args, 'use_kg_tokens', False):
            try:
                kg_cache = torch.load(args.kg_token_cache, map_location='cpu', weights_only=True)
            except TypeError:
                kg_cache = torch.load(args.kg_token_cache, map_location='cpu')
            split_cache = kg_cache['samples'].get(split, {})
            ids = split_cache.get('ids', [])
            id_to_index = split_cache.get('id_to_index') or {sample_id: idx for idx, sample_id in enumerate(ids)}
            tensors = {
                key: split_cache[key]
                for key in [
                    'kg_token_ids',
                    'kg_token_mask',
                    'kg_type_ids',
                    'kg_status_ids',
                    'kg_relation_ids',
                    'kg_attention_mask',
                ]
            }
            kg_token_lookup = {
                sample_id: {key: value[idx] for key, value in tensors.items()}
                for sample_id, idx in id_to_index.items()
            }
        self.parser = FieldParser(
            args,
            label_lookup=label_lookup,
            label_vocab=label_vocab,
            num_labels=num_labels,
            kg_token_lookup=kg_token_lookup,
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


