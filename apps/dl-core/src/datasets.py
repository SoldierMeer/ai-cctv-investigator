"""Datasets for Stage 2 & 3 cross-modal person search.

Supports CUHK-PEDES, ICFG-PEDES, RSTPReid, and `campus_custom` JSON manifests,
plus an offline `mock=True` mode for lightweight unit testing and CI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from _contracts import DATA_DIR

CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

SUPPORTED_DATASETS = ("cuhk_pedes", "icfg_pedes", "rstpreid", "campus_custom")

_MOCK_TEMPLATES = [
    ("red jacket", "black jeans", "black backpack", (210, 50, 50), (30, 30, 40)),
    ("blue hoodie", "gray sweatpants", "white sneakers", (50, 100, 210), (120, 120, 130)),
    ("green coat", "khaki trousers", "brown shoulder bag", (45, 150, 75), (180, 155, 110)),
    ("white t-shirt", "blue denim shorts", "red cap", (235, 235, 235), (60, 100, 175)),
    ("yellow sweater", "black slacks", "silver laptop bag", (225, 195, 45), (25, 25, 30)),
    ("purple cardigan", "dark skirt", "tan tote bag", (140, 65, 175), (45, 35, 55)),
    ("orange windbreaker", "navy pants", "sports duffel", (230, 120, 35), (30, 45, 95)),
    ("black blazer", "charcoal pants", "leather briefcase", (35, 35, 40), (70, 75, 80)),
]


def build_default_transform(
    image_size: Tuple[int, int] = (224, 224),
    is_train: bool = False,
) -> transforms.Compose:
    """Build standard CLIP-compatible image transform pipeline."""
    ops: List[Any] = [transforms.Resize(image_size)]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return transforms.Compose(ops)


def _normalize_dataset_name(dataset_name: str) -> str:
    norm = dataset_name.strip().lower().replace("-", "_")
    if norm not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Expected one of {SUPPORTED_DATASETS}."
        )
    return norm


def _resolve_manifest_path(
    dataset_name: str,
    root_dir: Optional[Union[str, Path]],
    annotation_file: Optional[Union[str, Path]],
) -> Tuple[Path, Path]:
    """Resolve dataset root directory and JSON manifest path relative to repo `data/`."""
    if root_dir is None:
        base_dir = DATA_DIR / ("campus_custom" if dataset_name == "campus_custom" else f"benchmarks/{dataset_name}")
    else:
        base_dir = Path(root_dir)

    if annotation_file is not None:
        ann_path = Path(annotation_file)
        if not ann_path.is_absolute():
            ann_path = base_dir / ann_path
        return base_dir, ann_path

    default_names = {
        "cuhk_pedes": "reid_raw.json",
        "icfg_pedes": "ICFG-PEDES.json",
        "rstpreid": "data_captions.json",
        "campus_custom": "manifest.json",
    }
    return base_dir, base_dir / default_names[dataset_name]


def _extract_record_fields(record: Dict[str, Any]) -> Tuple[int, List[str], List[str], str]:
    """Uniformly parse CUHK-PEDES, ICFG-PEDES, RSTPReid, and campus_custom records."""
    pid_raw = (
        record.get("id")
        if "id" in record
        else record.get("pid", record.get("person_id", record.get("person_track_id")))
    )
    if pid_raw is None:
        raise KeyError(f"Record missing person identity key: {list(record.keys())}")
    pid = int(pid_raw)

    if "crop_paths" in record and isinstance(record["crop_paths"], list) and record["crop_paths"]:
        img_paths = [str(p) for p in record["crop_paths"]]
    else:
        raw_img = record.get("file_path") or record.get("img_path") or record.get("image_path")
        if not raw_img:
            raise KeyError(f"Record missing image path key: {list(record.keys())}")
        img_paths = [str(raw_img)]

    raw_captions = record.get("captions")
    if isinstance(raw_captions, list) and raw_captions:
        captions = [str(c).strip() for c in raw_captions if str(c).strip()]
    else:
        single_cap = record.get("caption") or record.get("description") or record.get("query")
        if not single_cap:
            raise KeyError(f"Record missing caption text key: {list(record.keys())}")
        captions = [str(single_cap).strip()]

    split = str(record.get("split", "train")).strip().lower()
    return pid, img_paths, captions, split


class PersonSearchDataset(Dataset):
    """Cross-modal person search dataset with real manifest + offline mock modes."""

    def __init__(
        self,
        dataset_name: str = "cuhk_pedes",
        split: str = "train",
        root_dir: Optional[Union[str, Path]] = None,
        annotation_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        image_size: Tuple[int, int] = (224, 224),
        mock: bool = False,
        mock_size: int = 32,
        mock_num_pids: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_name = _normalize_dataset_name(dataset_name)
        self.split = split.strip().lower()
        self.mock = bool(mock)
        self.image_size = image_size
        self.transform = transform or build_default_transform(
            image_size=image_size, is_train=(self.split == "train")
        )
        self.samples: List[Dict[str, Any]] = []

        if self.mock:
            self.root_dir = DATA_DIR
            self._build_mock_samples(mock_size=mock_size, mock_num_pids=mock_num_pids, seed=seed)
        else:
            self.root_dir, self.annotation_path = _resolve_manifest_path(
                self.dataset_name, root_dir, annotation_file
            )
            self._load_manifest(self.annotation_path)

    def _build_mock_samples(self, mock_size: int, mock_num_pids: int, seed: int) -> None:
        rng = random.Random(seed)
        num_pids = max(1, min(mock_num_pids, mock_size))
        for idx in range(mock_size):
            pid = idx % num_pids
            top_desc, bot_desc, acc_desc, top_rgb, bot_rgb = _MOCK_TEMPLATES[
                pid % len(_MOCK_TEMPLATES)
            ]
            caption = (
                f"A pedestrian wearing a {top_desc} and {bot_desc} "
                f"with a {acc_desc} seen on camera."
            )
            self.samples.append(
                {
                    "pid": pid,
                    "caption": caption,
                    "image_path": f"mock://{self.dataset_name}/{self.split}/p{pid:03d}_{idx:04d}.jpg",
                    "top_rgb": top_rgb,
                    "bot_rgb": bot_rgb,
                    "jitter": rng.randint(-12, 12),
                }
            )

    def _load_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at '{manifest_path}'. "
                "Place the dataset under data/ or pass mock=True for offline testing."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            records = (
                payload.get("samples")
                or payload.get("annotations")
                or payload.get("tracklets")
                or payload.get("data")
                or []
            )
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported manifest root type: {type(payload)}")

        # Remap arbitrary dataset person IDs to contiguous integers 0..K-1
        pid_map: Dict[int, int] = {}
        for rec in records:
            raw_pid, img_paths, captions, rec_split = _extract_record_fields(rec)
            if self.split != "all" and rec_split != self.split:
                continue

            if raw_pid not in pid_map:
                pid_map[raw_pid] = len(pid_map)
            mapped_pid = pid_map[raw_pid]

            for img_rel in img_paths:
                img_path = Path(img_rel)
                if not img_path.is_absolute():
                    candidate = self.root_dir / img_path
                    if not candidate.exists() and (DATA_DIR.parent / img_path).exists():
                        candidate = DATA_DIR.parent / img_path
                    img_path = candidate
                for caption in captions:
                    self.samples.append(
                        {
                            "pid": mapped_pid,
                            "raw_pid": raw_pid,
                            "caption": caption,
                            "image_path": str(img_path),
                        }
                    )

    @staticmethod
    def _render_mock_image(
        top_rgb: Tuple[int, int, int],
        bot_rgb: Tuple[int, int, int],
        jitter: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        h, w = size
        img = Image.new("RGB", (w, h), color=(24, 24, 28))
        draw = ImageDraw.Draw(img)

        def _clamp(c: int) -> int:
            return max(0, min(255, c + jitter))

        top_c = tuple(_clamp(c) for c in top_rgb)
        bot_c = tuple(_clamp(c) for c in bot_rgb)

        # Draw synthetic 4-stripe pedestrian silhouette (head, torso, legs, bag)
        draw.ellipse([w * 0.35, h * 0.05, w * 0.65, h * 0.22], fill=(220, 185, 160))
        draw.rectangle([w * 0.25, h * 0.24, w * 0.75, h * 0.56], fill=top_c)
        draw.rectangle([w * 0.28, h * 0.57, w * 0.72, h * 0.92], fill=bot_c)
        draw.rectangle([w * 0.68, h * 0.30, w * 0.85, h * 0.62], fill=(60, 50, 45))
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if self.mock:
            pil_img = self._render_mock_image(
                top_rgb=sample["top_rgb"],
                bot_rgb=sample["bot_rgb"],
                jitter=sample["jitter"],
                size=self.image_size,
            )
        else:
            with Image.open(sample["image_path"]) as im:
                pil_img = im.convert("RGB")

        image_tensor = self.transform(pil_img)
        return {
            "image": image_tensor,
            "caption": sample["caption"],
            "pid": int(sample["pid"]),
            "image_path": sample["image_path"],
        }
