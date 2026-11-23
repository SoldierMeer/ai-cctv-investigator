"""Dual CLIP ViT-B/16 backbone for Stage 2 & Stage 3 cross-modal person search.

Wraps `open_clip` `'ViT-B-16'` (512-D shared space) and exposes:
- `encode_image`: L2-normalized `(B, 512)` float32 image embeddings.
- `encode_text`: L2-normalized `(B, 512)` float32 text embeddings (context length 77).
- `extract_spatial_tokens`: Projected spatial patch tokens `(B, N_patches, 512)`.
- `extract_visual_features` / `extract_text_tokens`: Global + token-level representations for Stage 3.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import open_clip
from open_clip.model import text_global_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

from _contracts import EMBED_DIM


class CLIPBackbone(nn.Module):
    """Dual CLIP ViT-B/16 image/text encoder anchored to `EMBED_DIM = 512`."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: Optional[str] = "openai",
        device: Optional[Union[str, torch.device]] = None,
        context_length: int = 77,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.context_length = int(context_length)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        output_dim = getattr(self.model.visual, "output_dim", EMBED_DIM)
        if int(output_dim) != EMBED_DIM:
            raise ValueError(
                f"Expected CLIP output_dim == {EMBED_DIM}, got {output_dim} for '{model_name}'."
            )
        self.embed_dim = EMBED_DIM

        img_size = getattr(self.model.visual, "image_size", (224, 224))
        if isinstance(img_size, int):
            self.image_size = (img_size, img_size)
        else:
            self.image_size = (int(img_size[0]), int(img_size[1]))

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        """Return device of the backbone parameters."""
        return next(self.model.parameters()).device

    def _prepare_images(self, image_tensors: torch.Tensor) -> torch.Tensor:
        if image_tensors.ndim != 4 or image_tensors.size(1) != 3:
            raise ValueError(
                f"Expected image_tensors of shape (B, 3, H, W), got {image_tensors.shape}"
            )
        x = image_tensors.to(device=self.device, dtype=torch.float32)
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(
                x, size=self.image_size, mode="bicubic", align_corners=False
            )
        return x

    def _project_visual(self, tensor: torch.Tensor) -> torch.Tensor:
        proj = getattr(self.model.visual, "proj", None)
        if proj is None:
            return tensor
        if isinstance(proj, nn.Linear):
            return proj(tensor)
        return tensor @ proj

    def _project_text(self, tensor: torch.Tensor) -> torch.Tensor:
        proj = getattr(self.model, "text_projection", None)
        if proj is None:
            return tensor
        if isinstance(proj, nn.Linear):
            return proj(tensor)
        return tensor @ proj

    def extract_visual_features(
        self, image_tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract unnormalized `[CLS]` global features `(B, 512)` and spatial tokens `(B, N_patches, 512)`."""
        x = self._prepare_images(image_tensors)
        visual = self.model.visual
        prev_flag = getattr(visual, "output_tokens", False)
        visual.output_tokens = True
        try:
            out = visual(x)
        finally:
            visual.output_tokens = prev_flag

        if isinstance(out, tuple):
            pooled, patch_tokens = out
        else:
            pooled = out
            patch_tokens = out.unsqueeze(1)

        if getattr(visual, "final_ln_after_pool", False) and hasattr(visual, "ln_post"):
            patch_tokens = visual.ln_post(patch_tokens)

        spatial_projected = self._project_visual(patch_tokens)
        return pooled.float(), spatial_projected.float()

    def extract_spatial_tokens(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """Expose patch tokens `(B, N_patches, 512)` before global pooling for stripe slicing."""
        _, spatial_tokens = self.extract_visual_features(image_tensors)
        return spatial_tokens

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """Encode image batch into L2-normalized `(B, 512)` float32 embeddings."""
        x = self._prepare_images(image_tensors)
        visual = self.model.visual
        prev_flag = getattr(visual, "output_tokens", False)
        visual.output_tokens = False
        try:
            feats = visual(x)
        finally:
            visual.output_tokens = prev_flag

        if isinstance(feats, tuple):
            feats = feats[0]
        return F.normalize(feats.float(), p=2, dim=-1)

    def tokenize(self, text_list: Sequence[str]) -> torch.Tensor:
        """Tokenize strings to fixed `context_length` (77) integer tensor."""
        if not text_list:
            raise ValueError("text_list must be non-empty.")
        tokens = self.tokenizer(list(text_list), context_length=self.context_length)
        return tokens.to(device=self.device)

    def extract_text_tokens(
        self, text_list: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract unnormalized global text vector `(B, 512)`, sequence tokens `(B, L, 512)`, and pad mask."""
        tokens = self.tokenize(text_list)
        cast_dtype = self.model.transformer.get_cast_dtype()

        x = self.model.token_embedding(tokens).to(cast_dtype)
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)

        pooled = text_global_pool(
            x,
            tokens,
            getattr(self.model, "text_pool_type", "argmax"),
            eos_token_id=getattr(self.model, "text_eos_id", None),
        )
        global_proj = self._project_text(pooled)
        seq_proj = self._project_text(x)
        key_padding_mask = tokens == 0
        return global_proj.float(), seq_proj.float(), key_padding_mask

    def encode_text(self, text_list: Sequence[str]) -> torch.Tensor:
        """Tokenize and encode strings into L2-normalized `(B, 512)` float32 embeddings."""
        tokens = self.tokenize(text_list)
        feats = self.model.encode_text(tokens, normalize=False)
        return F.normalize(feats.float(), p=2, dim=-1)

    def freeze_early_layers(
        self,
        num_trainable_visual_blocks: int = 2,
        num_trainable_text_blocks: int = 2,
    ) -> int:
        """Freeze early CLIP transformer layers, leaving top blocks and projections trainable.

        Returns:
            Number of trainable scalar parameters in the backbone.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        visual = self.model.visual
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = visual.transformer.resblocks
            if num_trainable_visual_blocks > 0:
                for block in blocks[-num_trainable_visual_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
        if hasattr(visual, "ln_post"):
            for param in visual.ln_post.parameters():
                param.requires_grad = True
        if getattr(visual, "proj", None) is not None:
            if isinstance(visual.proj, nn.Parameter):
                visual.proj.requires_grad = True
            elif isinstance(visual.proj, nn.Module):
                for param in visual.proj.parameters():
                    param.requires_grad = True

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "resblocks"):
            t_blocks = self.model.transformer.resblocks
            if num_trainable_text_blocks > 0:
                for block in t_blocks[-num_trainable_text_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
        if hasattr(self.model, "ln_final"):
            for param in self.model.ln_final.parameters():
                param.requires_grad = True
        if getattr(self.model, "text_projection", None) is not None:
            if isinstance(self.model.text_projection, nn.Parameter):
                self.model.text_projection.requires_grad = True
            elif isinstance(self.model.text_projection, nn.Module):
                for param in self.model.text_projection.parameters():
                    param.requires_grad = True

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
