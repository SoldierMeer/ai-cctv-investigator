"""Composite loss functions for Stage 3 IRRA cross-modal person search.

Implements:
- InfoNCELoss: Symmetric image-to-text / text-to-image contrastive loss.
- SDMLoss: Similarity Distribution Matching KL-divergence loss over identity labels.
- CompositeIRRALoss: Combined objective `L_InfoNCE + lambda_sdm * L_SDM`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric image-to-text and text-to-image contrastive cross-entropy loss."""

    def __init__(self, tau: float = 0.02, learnable: bool = False) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError(f"Temperature tau must be positive, got {tau}")
        self.learnable = learnable
        if learnable:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau), dtype=torch.float32))
        else:
            self.register_buffer("tau_buf", torch.tensor(float(tau), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """Return positive temperature tensor."""
        if self.learnable:
            return self.log_tau.exp().clamp(min=1e-4, max=1.0)
        return self.tau_buf

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            image_embeds: Tensor of shape (B, D).
            text_embeds: Tensor of shape (B, D).

        Returns:
            Scalar loss tensor.
        """
        if image_embeds.shape != text_embeds.shape:
            raise ValueError(
                f"Shape mismatch: image_embeds {image_embeds.shape} vs text_embeds {text_embeds.shape}"
            )
        batch_size = image_embeds.size(0)
        if batch_size == 0:
            raise ValueError("Batch size must be > 0 for InfoNCELoss.")

        img_norm = F.normalize(image_embeds.float(), p=2, dim=-1)
        txt_norm = F.normalize(text_embeds.float(), p=2, dim=-1)

        tau = self.temperature.to(device=img_norm.device, dtype=torch.float32)
        logits_i2t = torch.matmul(img_norm, txt_norm.t()) / tau
        logits_t2i = logits_i2t.t()

        targets = torch.arange(batch_size, device=img_norm.device, dtype=torch.long)
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)


class SDMLoss(nn.Module):
    """Similarity Distribution Matching (SDM) loss using KL divergence.

    Minimizes KL divergence between predicted cosine similarity softmax distributions
    and row-normalized ground-truth identity matching distributions.
    """

    def __init__(self, tau: float = 0.02, learnable: bool = False, eps: float = 1e-8) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError(f"Temperature tau must be positive, got {tau}")
        self.learnable = learnable
        self.eps = eps
        if learnable:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau), dtype=torch.float32))
        else:
            self.register_buffer("tau_buf", torch.tensor(float(tau), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """Return positive temperature tensor."""
        if self.learnable:
            return self.log_tau.exp().clamp(min=1e-4, max=1.0)
        return self.tau_buf

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute symmetric SDM KL-divergence loss.

        Args:
            image_embeds: Tensor of shape (B, D).
            text_embeds: Tensor of shape (B, D).
            pids: Optional identity labels of shape (B,). Defaults to unique IDs [0..B-1].

        Returns:
            Scalar KL-divergence loss tensor.
        """
        if image_embeds.shape != text_embeds.shape:
            raise ValueError(
                f"Shape mismatch: image_embeds {image_embeds.shape} vs text_embeds {text_embeds.shape}"
            )
        batch_size = image_embeds.size(0)
        if batch_size == 0:
            raise ValueError("Batch size must be > 0 for SDMLoss.")

        img_norm = F.normalize(image_embeds.float(), p=2, dim=-1)
        txt_norm = F.normalize(text_embeds.float(), p=2, dim=-1)

        if pids is None:
            pids = torch.arange(batch_size, device=img_norm.device, dtype=torch.long)
        else:
            pids = pids.to(device=img_norm.device).view(-1)
            if pids.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} pids, got {pids.numel()}")

        # Ground-truth identity match matrix M_ij = 1 if pid_i == pid_j else 0
        matches = (pids.unsqueeze(1) == pids.unsqueeze(0)).float()
        # Normalize rows to form target probability distribution Q
        target_dist = matches / (matches.sum(dim=1, keepdim=True) + self.eps)

        tau = self.temperature.to(device=img_norm.device, dtype=torch.float32)
        logits_i2t = torch.matmul(img_norm, txt_norm.t()) / tau
        logits_t2i = logits_i2t.t()

        log_pred_i2t = F.log_softmax(logits_i2t, dim=1)
        log_pred_t2i = F.log_softmax(logits_t2i, dim=1)

        # KL(Q || P) via PyTorch F.kl_div(log P, Q)
        kl_i2t = F.kl_div(log_pred_i2t, target_dist, reduction="batchmean", log_target=False)
        kl_t2i = F.kl_div(log_pred_t2i, target_dist, reduction="batchmean", log_target=False)
        return 0.5 * (kl_i2t + kl_t2i)


class CompositeIRRALoss(nn.Module):
    """Combined IRRA loss: `L_InfoNCE + lambda_sdm * L_SDM`."""

    def __init__(
        self,
        tau: float = 0.02,
        lambda_sdm: float = 1.0,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_sdm = float(lambda_sdm)
        self.infonce = InfoNCELoss(tau=tau, learnable=learnable_tau)
        self.sdm = SDMLoss(tau=tau, learnable=learnable_tau)
        self.last_components: Dict[str, float] = {}

    def compute_components(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss along with individual InfoNCE and SDM terms."""
        l_infonce = self.infonce(image_embeds, text_embeds)
        l_sdm = self.sdm(image_embeds, text_embeds, pids=pids)
        total = l_infonce + self.lambda_sdm * l_sdm
        self.last_components = {
            "total": float(total.detach().item()),
            "infonce": float(l_infonce.detach().item()),
            "sdm": float(l_sdm.detach().item()),
        }
        return total, l_infonce, l_sdm

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar composite loss `L_InfoNCE + lambda_sdm * L_SDM`."""
        total, _, _ = self.compute_components(image_embeds, text_embeds, pids=pids)
        return total
