import argparse
import collections
import json
import re
from pathlib import Path

import torch


def normalize_entity(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def iter_split_objects(path, split):
    marker = f'"{split}"'
    with open(path, "r", encoding="utf-8") as f:
        tail = ""
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return
            haystack = tail + chunk
            pos = haystack.find(marker)
            if pos >= 0:
                rest = haystack[pos + len(marker):]
                break
            tail = haystack[-len(marker):]

        while "[" not in rest:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return
            rest += chunk
        rest = rest[rest.find("[") + 1:]

        depth = 0
        in_string = False
        escape = False
        obj = []

        while True:
            if not rest:
                rest = f.read(1024 * 1024)
                if not rest:
                    return
            ch, rest = rest[0], rest[1:]

            if depth == 0:
                if ch == "]":
                    return
                if ch != "{":
                    continue

            obj.append(ch)

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield json.loads("".join(obj))
                    obj = []


def iter_entities(item, observation_only):
    serialized = item.get("serialized_result") or {}
    for result in serialized.values():
        entities = result.get("entities") or {}
        for entity in entities.values():
            label = entity.get("label") or ""
            if observation_only and not label.startswith("Observation::"):
                continue
            tokens = normalize_entity(entity.get("tokens") or "")
            if not tokens:
                continue
            yield tokens, label


def entity_key(tokens, label, mode):
    if mode == "tokens":
        return tokens
    if mode == "label":
        return label
    return f"{tokens}||{label}"


def main():
    parser = argparse.ArgumentParser(description="Build compact KG entity label cache")
    parser.add_argument("--kg_annotation", default="dataset/mimic_kg_annotation.json")
    parser.add_argument("--output", default="dataset/mimic_kg_entity_labels.pt")
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--mode", choices=["tokens", "label", "tokens_label"], default="tokens_label")
    parser.add_argument("--include_anatomy", action="store_true")
    parser.add_argument("--stream", action="store_true", help="use low-memory streaming parser; slower")
    args = parser.parse_args()

    kg_path = Path(args.kg_annotation)
    counts = collections.Counter()
    splits = ["train", "val", "test"]

    if args.stream:
        split_iter = lambda split: iter_split_objects(kg_path, split)
    else:
        print(f"loading {kg_path}")
        with open(kg_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        split_iter = lambda split: loaded[split]

    for split in splits:
        seen = 0
        for item in split_iter(split):
            for tokens, label in iter_entities(item, observation_only=not args.include_anatomy):
                counts[entity_key(tokens, label, args.mode)] += 1
            seen += 1
        print(f"counted {split}: {seen} samples", flush=True)

    vocab = [
        key
        for key, count in counts.most_common()
        if count >= args.min_freq
    ][:args.top_k]
    index = {key: idx for idx, key in enumerate(vocab)}

    labels = {}
    for split in splits:
        split_labels = {}
        seen = 0
        for item in split_iter(split):
            entity_indices = set()
            for tokens, label in iter_entities(item, observation_only=not args.include_anatomy):
                idx = index.get(entity_key(tokens, label, args.mode))
                if idx is not None:
                    entity_indices.add(idx)
            split_labels[item["id"]] = sorted(entity_indices)
            seen += 1
        labels[split] = split_labels
        print(f"labeled {split}: {seen} samples", flush=True)

    out = {
        "vocab": vocab,
        "counts": {key: counts[key] for key in vocab},
        "labels": labels,
        "source": str(kg_path),
        "mode": args.mode,
        "include_anatomy": args.include_anatomy,
        "min_freq": args.min_freq,
        "top_k": args.top_k,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"saved {len(vocab)} labels to {output}")


if __name__ == "__main__":
    main()
