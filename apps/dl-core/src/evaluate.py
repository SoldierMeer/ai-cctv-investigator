"""Retrieval evaluation metrics for cross-modal person search.

Computes Text-to-Image Recall@1, Recall@5, Recall@10, and Mean Average Precision (mAP).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int]]


def _to_float_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def _to_long_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long().view(-1)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().long().view(-1)
    return torch.tensor(list(x), dtype=torch.long).view(-1)


class RetrievalEvaluator:
    """Evaluator for cross-modal text-to-image person retrieval."""

    def __init__(self, as_percentage: bool = False) -> None:
        self.as_percentage = as_percentage

    def compute_metrics(
        self,
        image_embeds: Union[torch.Tensor, np.ndarray],
        text_embeds: Union[torch.Tensor, np.ndarray],
        gt_ids: Union[ArrayLike, Tuple[ArrayLike, ArrayLike]],
        text_gt_ids: Optional[ArrayLike] = None,
    ) -> Dict[str, float]:
        """Compute Recall@1, Recall@5, Recall@10, and mAP for text-to-image retrieval.

        Args:
            image_embeds: Gallery image embeddings of shape (N_img, D).
            text_embeds: Query text embeddings of shape (N_txt, D).
            gt_ids: Either 1D identity labels of shape (N_img,) when shared with text_embeds
                (or alongside `text_gt_ids`), or a tuple `(image_gt_ids, text_gt_ids)`.
            text_gt_ids: Optional 1D query identity labels of shape (N_txt,).

        Returns:
            Dictionary with keys `"Recall@1"`, `"Recall@5"`, `"Recall@10"`, and `"mAP"`.
            Values lie in `[0.0, 1.0]` by default (or `[0.0, 100.0]` if `as_percentage=True`).
        """
        img_t = _to_float_tensor(image_embeds)
        txt_t = _to_float_tensor(text_embeds)

        if img_t.ndim != 2 or txt_t.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got image {img_t.shape} and text {txt_t.shape}"
            )
        if img_t.size(0) == 0 or txt_t.size(0) == 0:
            raise ValueError("image_embeds and text_embeds must both be non-empty.")

        if isinstance(gt_ids, tuple) and len(gt_ids) == 2 and text_gt_ids is None:
            img_pids = _to_long_tensor(gt_ids[0])
            txt_pids = _to_long_tensor(gt_ids[1])
        else:
            img_pids = _to_long_tensor(gt_ids)  # type: ignore[arg-type]
            txt_pids = _to_long_tensor(text_gt_ids) if text_gt_ids is not None else img_pids

        if img_pids.numel() != img_t.size(0):
            raise ValueError(
                f"Gallery gt_ids length ({img_pids.numel()}) != num images ({img_t.size(0)})"
            )
        if txt_pids.numel() != txt_t.size(0):
            raise ValueError(
                f"Query gt_ids length ({txt_pids.numel()}) != num texts ({txt_t.size(0)})"
            )

        img_norm = F.normalize(img_t, p=2, dim=-1)
        txt_norm = F.normalize(txt_t, p=2, dim=-1)

        # Cosine similarity matrix: (N_txt, N_img)
        sim_matrix = torch.matmul(txt_norm, img_norm.t())
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

        # Binary relevance matrix sorted per query: (N_txt, N_img)
        ranked_gallery_pids = img_pids[sorted_indices]
        matches = (ranked_gallery_pids == txt_pids.unsqueeze(1)).float()

        num_positives = matches.sum(dim=1)
        valid_mask = num_positives > 0
        if not valid_mask.any():
            return {"Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "mAP": 0.0}

        valid_matches = matches[valid_mask]
        valid_pos = num_positives[valid_mask]

        r1 = float((valid_matches[:, :1].sum(dim=1) > 0).float().mean().item())
        r5 = float((valid_matches[:, :5].sum(dim=1) > 0).float().mean().item())
        r10 = float((valid_matches[:, :10].sum(dim=1) > 0).float().mean().item())

        ranks = torch.arange(1, valid_matches.size(1) + 1, dtype=torch.float32).unsqueeze(0)
        cum_hits = torch.cumsum(valid_matches, dim=1)
        precision_at_k = cum_hits / ranks
        ap_per_query = (precision_at_k * valid_matches).sum(dim=1) / valid_pos
        mean_ap = float(ap_per_query.mean().item())

        scale = 100.0 if self.as_percentage else 1.0
        return {
            "Recall@1": r1 * scale,
            "Recall@5": r5 * scale,
            "Recall@10": r10 * scale,
            "mAP": mean_ap * scale,
        }
