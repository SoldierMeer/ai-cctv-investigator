"""Inference helper for encoding pedestrian tracklet crops and text search queries.

Produces strictly L2-normalized `(1, 512)` float32 NumPy embeddings ready for FAISS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from _contracts import CHECKPOINTS_DIR, EMBED_DIM, REPO_ROOT, TOP_CROPS_PER_TRACKLET
from backbone_clip import CLIPBackbone
from datasets import build_default_transform
from stripe_attention_head import IRRAStripeHead


class EmbedExtractor:
    """Encodes up to 3 tracklet crops (averaged + re-normalized) or text queries into `(1, 512)` float32 arrays."""

    def __init__(
        self,
        backbone: Optional[CLIPBackbone] = None,
        stripe_head: Optional[IRRAStripeHead] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        pretrained: Optional[str] = "openai",
        device: Optional[str] = None,
        use_stripe_head: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone or CLIPBackbone(
            model_name="ViT-B-16",
            pretrained=pretrained,
            device=self.device,
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        self.use_stripe_head = use_stripe_head
        self.stripe_head = stripe_head

        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))
        elif self.stripe_head is not None:
            self.stripe_head.to(self.device)
            self.stripe_head.eval()

        self.transform = getattr(self.backbone, "preprocess", None) or build_default_transform(
            image_size=self.backbone.image_size, is_train=False
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load fine-tuned CLIP backbone and IRRA 4-stripe head weights."""
        resolved = checkpoint_path if checkpoint_path.is_absolute() else REPO_ROOT / checkpoint_path
        if not resolved.exists():
            default_ckpt = CHECKPOINTS_DIR / checkpoint_path.name
            if default_ckpt.exists():
                resolved = default_ckpt
            else:
                raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

        payload = torch.load(resolved, map_location=self.device, weights_only=False)
        if "backbone_state_dict" in payload and payload["backbone_state_dict"]:
            incompatible = self.backbone.load_state_dict(
                payload["backbone_state_dict"], strict=False
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys in backbone checkpoint: {incompatible.unexpected_keys}"
                )
        if "stripe_head_state_dict" in payload:
            if self.stripe_head is None:
                self.stripe_head = IRRAStripeHead(embed_dim=EMBED_DIM).to(self.device)
            self.stripe_head.load_state_dict(payload["stripe_head_state_dict"])

        self.backbone.eval()
        if self.stripe_head is not None:
            self.stripe_head.to(self.device)
            self.stripe_head.eval()

    def _resolve_crop_image(self, item: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        path = Path(item)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Crop image not found at '{item}' (resolved: '{path}').")
        with Image.open(path) as img:
            return img.convert("RGB")

    @torch.no_grad()
    def encode_crops(
        self, crop_paths: Sequence[Union[str, Path, Image.Image]]
    ) -> np.ndarray:
        """Encode 1..3 pedestrian crops, average them, and L2 re-normalize to `(1, 512)` float32."""
        if not crop_paths:
            raise ValueError("crop_paths must contain at least 1 crop.")
        if len(crop_paths) > TOP_CROPS_PER_TRACKLET:
            raise ValueError(
                f"Expected at most {TOP_CROPS_PER_TRACKLET} crops per tracklet, got {len(crop_paths)}."
            )

        tensors = [self.transform(self._resolve_crop_image(item)) for item in crop_paths]
        batch = torch.stack(tensors, dim=0).to(self.device)

        if self.use_stripe_head and self.stripe_head is not None:
            cls_embed, spatial_tokens = self.backbone.extract_visual_features(batch)
            crop_embeds = self.stripe_head(
                spatial_tokens=spatial_tokens,
                text_tokens=None,
                cls_embed=cls_embed,
            )
        else:
            crop_embeds = self.backbone.encode_image(batch)

        averaged = crop_embeds.mean(dim=0, keepdim=True)
        normalized = F.normalize(averaged.float(), p=2, dim=-1)
        return normalized.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode a natural language query string into `(1, 512)` L2-normalized float32 array."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string must be non-empty.")
        embed = self.backbone.encode_text([query.strip()])
        return embed.cpu().numpy().astype(np.float32)

    def encode(
        self, inputs: Union[str, Sequence[Union[str, Path, Image.Image]]]
    ) -> np.ndarray:
        """Dispatch either a text query or a sequence of up to 3 crop paths to `(1, 512)` float32."""
        if isinstance(inputs, str):
            return self.encode_text_query(inputs)
        return self.encode_crops(inputs)
