"""Deep Learning Core (Stages 2 & 3) for AI CCTV Investigator.

Stage 2: Dual CLIP ViT-B/16 Backbone (512-D L2-normalized embedding space).
Stage 3: IRRA 4-Stripe Horizontal Attention Head trained with InfoNCE + SDM Loss.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ._contracts import EMBED_DIM, NUM_HORIZONTAL_STRIPES
from .backbone_clip import CLIPBackbone
from .datasets import PersonSearchDataset
from .embed_extractor import EmbedExtractor
from .evaluate import RetrievalEvaluator
from .losses import CompositeIRRALoss, InfoNCELoss, SDMLoss
from .stripe_attention_head import IRRAStripeHead

__all__ = [
    "EMBED_DIM",
    "NUM_HORIZONTAL_STRIPES",
    "CLIPBackbone",
    "IRRAStripeHead",
    "InfoNCELoss",
    "SDMLoss",
    "CompositeIRRALoss",
    "PersonSearchDataset",
    "RetrievalEvaluator",
    "EmbedExtractor",
]
