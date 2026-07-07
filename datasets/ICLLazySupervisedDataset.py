import copy
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .LazySupervisedDataset import (
    LazySupervisedDataset,
    extract_masks_fun,
    preprocess,
    preprocess_multimodal,
    process_mask,
)


class ICLLazySupervisedDataset(LazySupervisedDataset):
    """Dataset for MedPLIB-ICL segmentation with 1-3 in-context examples."""

    def _resolve_path(self, file_name: str) -> str:
        image_folder = self.data_args.image_folder
        if os.path.exists(file_name):
            return file_name
        if "llavamed" in file_name:
            return os.path.join("/".join(image_folder.split("/")[:-1]), file_name)
        return os.path.join(image_folder, file_name)

    def _load_rgb(self, image_file: str) -> Tuple[str, np.ndarray]:
        image_path = self._resolve_path(image_file)
        assert os.path.exists(image_path), f"{image_path} dose not exist"
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_path, image_rgb

    def _load_mask(self, mask_file: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        mask_path = self._resolve_path(mask_file)
        assert os.path.exists(mask_path), f"{mask_path} dose not exist"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if target_shape is not None and mask.shape[:2] != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 1).astype(np.uint8)
        return mask

    def _overlay_mask(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        color = np.array([118, 158, 224], dtype=np.float32)
        image = image_rgb.astype(np.float32)
        image[mask > 0] = image[mask > 0] * 0.45 + color * 0.55
        return np.clip(image, 0, 255).astype(np.uint8)

    def _mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        mask_rgb = (mask * 255).astype(np.uint8)
        return np.stack([mask_rgb, mask_rgb, mask_rgb], axis=-1)

    def _preprocess_sam_image(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image_resize = self.transform.apply_image(image_rgb)
        resize = image_resize.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_resize).permute(2, 0, 1).contiguous(),
            self.sam_img_size,
        )
        return image_sam, resize

    def _preprocess_clip_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        processor = self.data_args.image_processor
        image_clip = self.transform_clip.apply_image(image_rgb)
        image_clip = self.preprocess(
            torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(),
            self.clip_img_size,
            normalize=False,
        )
        if self.data_args.image_aspect_ratio == "pad":
            return processor.preprocess(image_clip, return_tensors="pt")["pixel_values"][0]
        return processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]

    def _preprocess_encoder_mask(self, mask: np.ndarray) -> torch.Tensor:
        mask_resize = self.transform_clip.apply_image(mask.astype(np.uint8) * 255)
        mask_tensor = self.preprocess(
            torch.from_numpy(mask_resize).contiguous(),
            self.clip_img_size,
            normalize=False,
            is_mask=True,
        )
        return (mask_tensor > 0).float().unsqueeze(0)

    def _image_token_len(self) -> int:
        if getattr(self.data_args, "mm_token_compress", False):
            return getattr(self.data_args, "mm_compressed_token_count", 256)
        return 576

    def _mask_token_len(self) -> int:
        return getattr(self.data_args, "mask_encoder_token_count", 64)

    def _use_mask_encoder(self) -> bool:
        return self._mask_mode() == "separate" and getattr(self.data_args, "icl_mask_encoder", False)

    def _get_flat_icl_examples(self, source: Dict) -> List[Dict[str, str]]:
        examples = source.get("icl_examples", source.get("examples", []))
        if examples:
            return examples[:3]

        indexed_images = sorted(
            int(key.replace("image", ""))
            for key in source
            if key.startswith("image") and key.replace("image", "").isdigit()
        )
        if not indexed_images:
            return []

        target_idx = None
        if "image" not in source:
            target_idx = indexed_images[-1]
            source.setdefault("image", source[f"image{target_idx}"])
            if f"mask{target_idx}" in source:
                source.setdefault("target_mask", source[f"mask{target_idx}"])

        flat_examples = []
        for idx in indexed_images:
            if idx == target_idx:
                continue
            image_key = f"image{idx}"
            mask_key = f"mask{idx}"
            if image_key in source and mask_key in source:
                flat_examples.append({"image": source[image_key], "mask": source[mask_key]})
        return flat_examples[:3]

    def _has_target_mask_tag(self, source: Dict) -> bool:
        return any(re.search(r"<mask>(.*?)</mask>", str(item.get("value", ""))) for item in source["conversations"])

    def _count_image_tokens(self, source: Dict) -> int:
        return sum(str(item.get("value", "")).count("<image>") for item in source.get("conversations", []))

    def _mask_mode(self) -> str:
        mask_mode = getattr(self.data_args, "icl_mask_mode", "overlay")
        assert mask_mode in ["overlay", "separate"], f"Unsupported ICL mask mode: {mask_mode}"
        return mask_mode

    def _expected_image_tokens(self, num_examples: int) -> int:
        if self._mask_mode() == "separate":
            return num_examples * 2 + 1
        return num_examples + 1

    def _build_default_conversation(self, source: Dict, num_examples: int) -> List[Dict[str, str]]:
        image_blocks = []
        if self._mask_mode() == "separate":
            for idx in range(num_examples):
                image_blocks.append(
                    f"Example {idx + 1} image: <image>\nExample {idx + 1} mask: <image>"
                )
        else:
            for idx in range(num_examples):
                image_blocks.append(
                    f"Example {idx + 1}: <image>\nThe blue overlay is the reference segmentation mask."
                )
        image_blocks.append(
            "Query: <image>\nRefer to the previous examples and segment the corresponding target in this image."
        )
        answer = "<SEG>"
        target_mask = source.get("target_mask", source.get("mask", source.get("mask3", None)))
        if target_mask is not None:
            answer += f"<mask>{target_mask}</mask>"
        return [
            {"from": "human", "value": "\n".join(image_blocks)},
            {"from": "gpt", "value": answer},
        ]

    def _prepare_source(self, source: Dict, num_examples: int) -> Dict:
        source = copy.deepcopy(source)
        if "conversations" not in source or self._count_image_tokens(source) < self._expected_image_tokens(num_examples):
            source["conversations"] = self._build_default_conversation(source, num_examples)
        elif not self._has_target_mask_tag(source):
            target_mask = source.get("target_mask", source.get("mask", source.get("mask3", None)))
            if target_mask is not None:
                source["conversations"][-1]["value"] = (
                    str(source["conversations"][-1]["value"]) + f"<mask>{target_mask}</mask>"
                )
        return source

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw_source = self.list_data_dict[i]
        examples = self._get_flat_icl_examples(raw_source)
        assert 1 <= len(examples) <= 3, "MedPLIB-ICL requires 1 to 3 in-context examples."

        source = self._prepare_source(raw_source, len(examples))
        answer_type = source.get("answer_type", None)

        masks, sources = extract_masks_fun(source, self.data_args.image_folder, pattern=r"<mask>(.*?)</mask>")
        region_masks = []
        is_valid_region = True

        target_image_file = source.get("image", source.get("image3", None))
        assert target_image_file is not None, "MedPLIB-ICL requires a target image in `image` or `image3`."
        image_path, target_rgb = self._load_rgb(target_image_file)
        image_sam, resize = self._preprocess_sam_image(target_rgb)

        images_clip = []
        mask_images = []
        image_token_types = []
        image_token_lengths = []
        icl_image_paths = []
        for example in examples:
            example_path, example_rgb = self._load_rgb(example["image"])
            example_mask = self._load_mask(example["mask"], target_shape=example_rgb.shape[:2])
            if self._mask_mode() == "separate":
                images_clip.append(self._preprocess_clip_image(example_rgb))
                image_token_types.append("image")
                image_token_lengths.append(self._image_token_len())
                if self._use_mask_encoder():
                    mask_images.append(self._preprocess_encoder_mask(example_mask))
                    image_token_types.append("mask")
                    image_token_lengths.append(self._mask_token_len())
                else:
                    images_clip.append(self._preprocess_clip_image(self._mask_to_rgb(example_mask)))
                    image_token_types.append("image")
                    image_token_lengths.append(self._image_token_len())
                icl_image_paths.extend([example_path, self._resolve_path(example["mask"])])
            else:
                images_clip.append(self._preprocess_clip_image(self._overlay_mask(example_rgb, example_mask)))
                image_token_types.append("image")
                image_token_lengths.append(self._image_token_len())
                icl_image_paths.append(example_path)
        images_clip.append(self._preprocess_clip_image(target_rgb))
        image_token_types.append("image")
        image_token_lengths.append(self._image_token_len())

        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.data_args,
        )
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        masks = [process_mask(mask) for mask in masks]
        region_masks = [process_mask(mask).unsqueeze(0) for mask in region_masks]

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            conversations=data_dict["conversations"],
            question=data_dict["question"],
            gt=data_dict["gt"],
        )
        data_dict["image_clip"] = torch.stack(images_clip, dim=0)
        data_dict["image_sam"] = image_sam
        data_dict["image_path"] = image_path
        data_dict["icl_image_paths"] = icl_image_paths
        data_dict["icl_image_count"] = len(images_clip)
        data_dict["mask_images"] = torch.stack(mask_images, dim=0) if len(mask_images) > 0 else torch.empty(0)
        data_dict["image_token_types"] = image_token_types
        data_dict["image_token_lengths"] = image_token_lengths
        data_dict["masks"] = masks
        data_dict["region_masks"] = region_masks
        data_dict["inference"] = False
        data_dict["tokenizer"] = self.tokenizer
        data_dict["answer_type"] = answer_type

        if len(masks) > 0:
            label = [torch.ones(masks[0].shape[0], masks[0].shape[1]) * self.ignore_label] * len(masks)
            data_dict["label"] = label
            data_dict["resize"] = [resize] * len(masks)

        if len(region_masks) > 0 and not is_valid_region:
            data_dict["labels"] = torch.zeros_like(data_dict["labels"])
            data_dict["labels"][...] = -100

        return data_dict
