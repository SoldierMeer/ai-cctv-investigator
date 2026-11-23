"""IRRA 4-Stripe Horizontal Attention Head for fine-grained attribute alignment.

Partitions spatial visual patch tokens into 4 horizontal anatomical stripes
(Head, Torso, Legs, Accessories/Bag), aligns them against text token embeddings
via Cross-Modal Multi-Head Attention, and fuses them with the global [CLS] token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from _contracts import EMBED_DIM, NUM_HORIZONTAL_STRIPES


class IRRAStripeHead(nn.Module):
    """4-Stripe horizontal regional attention head with residual MLP fusion."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_stripes: int = NUM_HORIZONTAL_STRIPES,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = int(embed_dim)
        self.num_stripes = int(num_stripes)

        self.stripe_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_stripes, self.embed_dim)
        )
        nn.init.trunc_normal_(self.stripe_pos_embed, std=0.02)

        self.norm_q = nn.LayerNorm(self.embed_dim)
        self.norm_kv = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        hidden_dim = int(self.embed_dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.Dropout(dropout),
        )

    def partition_stripes(
        self,
        spatial_tokens: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Reshape spatial patch tokens into 2D grid and pool into horizontal stripes.

        Args:
            spatial_tokens: Tensor of shape (B, N_patches, D).
            grid_size: Optional (H, W) patch grid dimensions. Defaults to square grid.

        Returns:
            Regional visual tokens of shape (B, NUM_HORIZONTAL_STRIPES, D).
        """
        if spatial_tokens.ndim != 3:
            raise ValueError(
                f"Expected 3D spatial_tokens (B, N_patches, D), got {spatial_tokens.shape}"
            )
        _, n_patches, dim = spatial_tokens.shape
        if dim != self.embed_dim:
            raise ValueError(
                f"Expected token dim {self.embed_dim}, got {dim}."
            )

        if grid_size is None:
            side = math.isqrt(n_patches)
            if side * side != n_patches:
                raise ValueError(
                    f"N_patches ({n_patches}) is not a perfect square; provide explicit grid_size=(H, W)."
                )
            h, w = side, side
        else:
            h, w = grid_size
            if h * w != n_patches:
                raise ValueError(
                    f"grid_size {grid_size} product ({h * w}) != N_patches ({n_patches})."
                )

        # (B, H*W, D) -> (B, D, H, W)
        grid = rearrange(spatial_tokens, "b (h w) d -> b d h w", h=h, w=w)
        # Partition vertically into `num_stripes` equal horizontal anatomical regions
        pooled = F.adaptive_avg_pool2d(grid, output_size=(self.num_stripes, 1))
        return rearrange(pooled, "b d s 1 -> b s d")

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        cls_embed: Optional[torch.Tensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Align 4 anatomical stripe tokens with text tokens and fuse with [CLS].

        Args:
            spatial_tokens: Patch tokens of shape (B, N_patches, 512).
            text_tokens: Text token embeddings (B, L, 512) or pooled (B, 512).
                If None, runs self-attention across the 4 stripes for gallery-only inference.
            cls_embed: Global [CLS] embedding of shape (B, 512). If None, uses mean patch token.
            text_key_padding_mask: Optional boolean mask (B, L) where True indicates padding.
            grid_size: Optional (H, W) spatial patch grid dimensions.

        Returns:
            Strictly L2-normalized embeddings of shape (B, 512).
        """
        stripes = self.partition_stripes(spatial_tokens, grid_size=grid_size)
        q = self.norm_q(stripes + self.stripe_pos_embed.to(dtype=stripes.dtype))

        if text_tokens is not None:
            kv_raw = text_tokens.unsqueeze(1) if text_tokens.ndim == 2 else text_tokens
            if kv_raw.ndim != 3 or kv_raw.size(-1) != self.embed_dim:
                raise ValueError(
                    f"Expected text_tokens of shape (B, L, {self.embed_dim}) or (B, {self.embed_dim}), "
                    f"got {text_tokens.shape}"
                )
            kv = self.norm_kv(kv_raw)
            pad_mask = text_key_padding_mask
            if pad_mask is not None:
                # Guard against all-masked rows producing NaNs in softmax
                pad_mask = pad_mask.bool()
                all_masked = pad_mask.all(dim=-1, keepdim=True)
                if all_masked.any():
                    pad_mask = pad_mask.clone()
                    pad_mask[all_masked.squeeze(-1), 0] = False

            attn_out, _ = self.cross_attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=pad_mask,
                need_weights=False,
            )
        else:
            attn_out, _ = self.cross_attn(
                query=q,
                key=q,
                value=q,
                need_weights=False,
            )

        stripe_refined = stripes + self.dropout(attn_out)
        stripe_pooled = stripe_refined.mean(dim=1)

        if cls_embed is None:
            cls_vec = spatial_tokens.mean(dim=1)
        else:
            cls_vec = cls_embed

        combined = cls_vec + stripe_pooled
        fused = combined + self.mlp(self.norm_mlp(combined))
        return F.normalize(fused.float(), p=2, dim=-1)
