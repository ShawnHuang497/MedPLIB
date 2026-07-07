import argparse
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

RAG_ENCODER_DEFAULT_PATHS = {
    "clip_encoder": "/data/3/MedPLIB/checkpoint/clip-vit-large-patch14-336",
    "med_encoder": "/data/3/MedPLIB/checkpoint/med_encoder",
    "det_encoder": "/data/3/MedPLIB/checkpoint/det_encoder",
    "mask_encoder": "/data/3/MedPLIB/checkpoint/mask_encoder",
}


def resolve_path(path: str, image_folder: str) -> str:
    if path is None:
        return None
    if os.path.exists(path):
        return path
    return os.path.join(image_folder, path)


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_features(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12)


def extract_target_mask(item: Dict) -> Optional[str]:
    for key in ["target_mask", "mask", "mask3"]:
        if item.get(key) is not None:
            return item[key]
    for turn in item.get("conversations", []):
        value = str(turn.get("value", ""))
        start = value.find("<mask>")
        end = value.find("</mask>")
        if start >= 0 and end > start:
            return value[start + len("<mask>"):end]
    return None


def extract_query_image(item: Dict) -> Optional[str]:
    if item.get("image") is not None:
        return item["image"]
    indexed = sorted(
        int(key.replace("image", ""))
        for key in item
        if key.startswith("image") and key.replace("image", "").isdigit()
    )
    if indexed:
        return item[f"image{indexed[-1]}"]
    return None


def collect_candidates(items: List[Dict]) -> List[Dict[str, str]]:
    candidates = []
    for item in items:
        image = extract_query_image(item)
        mask = extract_target_mask(item)
        if image is not None and mask is not None:
            candidates.append({"image": image, "mask": mask})

        for example in item.get("icl_examples", item.get("examples", [])):
            if example.get("image") is not None and example.get("mask") is not None:
                candidates.append({"image": example["image"], "mask": example["mask"]})

        indexed = sorted(
            int(key.replace("image", ""))
            for key in item
            if key.startswith("image") and key.replace("image", "").isdigit()
        )
        for idx in indexed:
            image_key = f"image{idx}"
            mask_key = f"mask{idx}"
            if item.get(image_key) is not None and item.get(mask_key) is not None:
                candidates.append({"image": item[image_key], "mask": item[mask_key]})
    return candidates


class ImageRAGEncoder:
    def __init__(
        self,
        encoder_type: str = "clip_encoder",
        encoder_path: Optional[str] = None,
        device: str = "cuda",
        precision: str = "bf16",
    ):
        if encoder_type not in RAG_ENCODER_DEFAULT_PATHS:
            raise ValueError(
                f"Unsupported RAG encoder type: {encoder_type}. "
                f"Choose from {list(RAG_ENCODER_DEFAULT_PATHS.keys())}."
            )
        if encoder_path is None or encoder_path == "":
            encoder_path = RAG_ENCODER_DEFAULT_PATHS[encoder_type]
        self.encoder_type = encoder_type
        self.encoder_path = encoder_path
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.processor = CLIPImageProcessor.from_pretrained(encoder_path)
        self.model = CLIPVisionModel.from_pretrained(encoder_path)
        if precision == "fp16" and self.device != "cpu":
            self.model = self.model.half()
        elif precision == "bf16" and self.device != "cpu":
            self.model = self.model.bfloat16()
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_paths(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        all_features = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            images = [load_rgb(path) for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if next(self.model.parameters()).dtype == torch.float16:
                pixel_values = pixel_values.half()
            elif next(self.model.parameters()).dtype == torch.bfloat16:
                pixel_values = pixel_values.bfloat16()
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            features = outputs.last_hidden_state[:, 1:].mean(dim=1)
            all_features.append(features.float().cpu().numpy())
        return normalize_features(np.concatenate(all_features, axis=0))


def build_index(args):
    with open(args.candidate_json, "r") as f:
        items = json.load(f)
    candidates = collect_candidates(items)
    if len(candidates) == 0:
        raise ValueError("No image/mask candidates found.")

    paths = [resolve_path(candidate["image"], args.image_folder) for candidate in candidates]
    encoder = ImageRAGEncoder(
        encoder_type=args.rag_encoder_type,
        encoder_path=args.rag_encoder_path,
        device=args.device,
        precision=args.precision,
    )
    embeddings = encoder.encode_paths(paths, batch_size=args.batch_size)

    os.makedirs(args.index_dir, exist_ok=True)
    np.save(os.path.join(args.index_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(args.index_dir, "metadata.json"), "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Saved {len(candidates)} candidates to {args.index_dir}")


def load_index(index_dir: str):
    embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
    with open(os.path.join(index_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return normalize_features(embeddings), metadata


def retrieve(query_feature: np.ndarray, embeddings: np.ndarray, metadata: List[Dict], top_k: int):
    scores = embeddings @ query_feature
    indices = np.argsort(-scores)[:top_k]
    return [metadata[int(idx)] for idx in indices]


def augment(args):
    with open(args.query_json, "r") as f:
        query_items = json.load(f)
    embeddings, metadata = load_index(args.index_dir)

    query_paths = [resolve_path(extract_query_image(item), args.image_folder) for item in query_items]
    encoder = ImageRAGEncoder(
        encoder_type=args.rag_encoder_type,
        encoder_path=args.rag_encoder_path,
        device=args.device,
        precision=args.precision,
    )
    query_features = encoder.encode_paths(query_paths, batch_size=args.batch_size)

    augmented = []
    for item, feature in zip(query_items, query_features):
        new_item = dict(item)
        new_item["image"] = extract_query_image(new_item)
        target_mask = extract_target_mask(new_item)
        if target_mask is not None:
            new_item["target_mask"] = target_mask
        new_item["icl_examples"] = retrieve(feature, embeddings, metadata, args.top_k)
        augmented.append(new_item)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(augmented, f, indent=2)
    print(f"Saved RAG-augmented ICL JSON to {args.output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Decoupled image-RAG for MedPLIB-ICL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument(
            "--rag_encoder_type",
            default="clip_encoder",
            choices=list(RAG_ENCODER_DEFAULT_PATHS.keys()),
        )
        subparser.add_argument("--rag_encoder_path", default=None)
        subparser.add_argument("--image_folder", default="/data/3/MedPLIB/dataset/images-and-masks-root")
        subparser.add_argument("--index_dir", default="/data/3/MedPLIB/dataset/rag_index")
        subparser.add_argument("--batch_size", type=int, default=16)
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")

    build_parser = subparsers.add_parser("build")
    add_common(build_parser)
    build_parser.add_argument("--candidate_json", default="/data/3/MedPLIB/dataset/MedPLIB_ICL_train.json")

    augment_parser = subparsers.add_parser("augment")
    add_common(augment_parser)
    augment_parser.add_argument("--query_json", default="/data/3/MedPLIB/dataset/MedPLIB_ICL_test.json")
    augment_parser.add_argument("--output_json", default="/data/3/MedPLIB/dataset/MedPLIB_ICL_RAG_test.json")
    augment_parser.add_argument("--top_k", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.command == "build":
        build_index(parsed_args)
    elif parsed_args.command == "augment":
        augment(parsed_args)
