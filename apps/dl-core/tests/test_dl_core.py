"""Unit test suite for Stage 2 & 3 Deep Learning Core (apps/dl-core).

Tests:
1. CLIPBackbone (B, 512) output shapes, spatial tokens, layer freezing, and unit L2 norm.
2. IRRAStripeHead 4-stripe partitioning and (B, 512) unit-norm cross-modal/visual outputs.
3. InfoNCELoss, SDMLoss, and CompositeIRRALoss backward gradient flow with zero NaNs/Infs.
4. PersonSearchDataset (mock & JSON manifests), RetrievalEvaluator exact Recall@1/5/10 + mAP,
   EmbedExtractor (1, 512) float32 outputs, and lightweight mock training/checkpointing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

DL_CORE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = DL_CORE_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _contracts import EMBED_DIM, NUM_HORIZONTAL_STRIPES, ensure_shared_contracts_on_path  # noqa: E402
from backbone_clip import CLIPBackbone  # noqa: E402
from datasets import PersonSearchDataset, build_default_transform  # noqa: E402
from embed_extractor import EmbedExtractor  # noqa: E402
from evaluate import RetrievalEvaluator  # noqa: E402
from losses import CompositeIRRALoss, InfoNCELoss, SDMLoss  # noqa: E402
from stripe_attention_head import IRRAStripeHead  # noqa: E402
import train as train_module  # noqa: E402


@pytest.fixture(scope="module")
def clip_backbone() -> CLIPBackbone:
    """Shared randomly initialized ViT-B-16 backbone for fast offline tests."""
    backbone = CLIPBackbone(model_name="ViT-B-16", pretrained=None, device="cpu")
    backbone.eval()
    return backbone


# ============================================================================
# Test 1: CLIPBackbone shapes, spatial tokens, freezing, and strict unit norm
# ============================================================================
def test_clip_backbone_shapes_and_unit_norm(clip_backbone: CLIPBackbone) -> None:
    contracts_path = ensure_shared_contracts_on_path()
    assert contracts_path.exists()

    batch_size = 3
    images = torch.randn(batch_size, 3, 224, 224)
    texts = [
        "Person wearing a red jacket and dark jeans",
        "Pedestrian with a blue hoodie carrying a backpack",
        "Woman in a green coat walking near the main gate",
    ]

    with torch.no_grad():
        img_embeds = clip_backbone.encode_image(images)
        txt_embeds = clip_backbone.encode_text(texts)
        spatial_tokens = clip_backbone.extract_spatial_tokens(images)
        cls_feat, feat_tokens = clip_backbone.extract_visual_features(images)
        txt_global, txt_seq, txt_pad_mask = clip_backbone.extract_text_tokens(texts)
        tokens_ids = clip_backbone.tokenize(texts)

        # Also verify non-224x224 Stage-1 crop inputs (384x128) resize smoothly
        rect_images = torch.randn(2, 3, 384, 128)
        rect_embeds = clip_backbone.encode_image(rect_images)

    assert img_embeds.shape == (batch_size, EMBED_DIM)
    assert img_embeds.dtype == torch.float32
    assert torch.allclose(img_embeds.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    assert txt_embeds.shape == (batch_size, EMBED_DIM)
    assert txt_embeds.dtype == torch.float32
    assert torch.allclose(txt_embeds.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    assert tokens_ids.shape == (batch_size, 77)
    assert spatial_tokens.shape == (batch_size, 196, EMBED_DIM)
    assert cls_feat.shape == (batch_size, EMBED_DIM)
    assert feat_tokens.shape == (batch_size, 196, EMBED_DIM)
    assert txt_global.shape == (batch_size, EMBED_DIM)
    assert txt_seq.shape == (batch_size, 77, EMBED_DIM)
    assert txt_pad_mask.shape == (batch_size, 77)
    assert rect_embeds.shape == (2, EMBED_DIM)
    assert torch.allclose(rect_embeds.norm(dim=-1), torch.ones(2), atol=1e-5)

    # Verify early layer freezing keeps early blocks frozen and top blocks trainable
    trainable_count = clip_backbone.freeze_early_layers(
        num_trainable_visual_blocks=2, num_trainable_text_blocks=2
    )
    assert trainable_count > 0
    assert not any(
        p.requires_grad for p in clip_backbone.model.visual.transformer.resblocks[0].parameters()
    )
    assert all(
        p.requires_grad for p in clip_backbone.model.visual.transformer.resblocks[-1].parameters()
    )
    assert not any(
        p.requires_grad for p in clip_backbone.model.transformer.resblocks[0].parameters()
    )
    assert all(
        p.requires_grad for p in clip_backbone.model.transformer.resblocks[-1].parameters()
    )


# ============================================================================
# Test 2: IRRAStripeHead 4-stripe partitioning and unit-norm output
# ============================================================================
def test_irra_stripe_head_partition_and_unit_norm() -> None:
    torch.manual_seed(0)
    head = IRRAStripeHead(embed_dim=EMBED_DIM, num_stripes=NUM_HORIZONTAL_STRIPES, num_heads=8)
    head.eval()

    batch_size = 4
    n_patches = 196  # 14 x 14 patch grid from ViT-B/16
    spatial_tokens = torch.randn(batch_size, n_patches, EMBED_DIM)
    text_seq = torch.randn(batch_size, 77, EMBED_DIM)
    cls_embed = torch.randn(batch_size, EMBED_DIM)
    pad_mask = torch.zeros(batch_size, 77, dtype=torch.bool)
    pad_mask[:, 12:] = True

    with torch.no_grad():
        stripes = head.partition_stripes(spatial_tokens)
        assert stripes.shape == (batch_size, NUM_HORIZONTAL_STRIPES, EMBED_DIM)

        # Cross-modal multi-head attention against 3D text tokens
        cross_out = head(
            spatial_tokens=spatial_tokens,
            text_tokens=text_seq,
            cls_embed=cls_embed,
            text_key_padding_mask=pad_mask,
        )
        # Cross-modal attention with 2D text embedding
        cross_2d_out = head(
            spatial_tokens=spatial_tokens,
            text_tokens=torch.randn(batch_size, EMBED_DIM),
            cls_embed=cls_embed,
        )
        # Gallery-only visual self-attention mode (text_tokens=None)
        vis_out = head(spatial_tokens=spatial_tokens, text_tokens=None, cls_embed=cls_embed)

    for out in (cross_out, cross_2d_out, vis_out):
        assert out.shape == (batch_size, EMBED_DIM)
        assert out.dtype == torch.float32
        assert torch.allclose(out.norm(dim=-1), torch.ones(batch_size), atol=1e-5)
        assert torch.isfinite(out).all()


# ============================================================================
# Test 3: CompositeIRRALoss backward gradient flow with zero NaNs/Infs
# ============================================================================
def test_composite_irra_loss_gradient_flow() -> None:
    torch.manual_seed(7)
    batch_size = 6
    spatial_tokens = torch.randn(batch_size, 196, EMBED_DIM, requires_grad=True)
    cls_embed = torch.randn(batch_size, EMBED_DIM, requires_grad=True)
    text_seq = torch.randn(batch_size, 16, EMBED_DIM, requires_grad=True)
    text_global = torch.randn(batch_size, EMBED_DIM, requires_grad=True)
    # Include repeated PIDs to exercise multi-positive SDM distribution matching
    pids = torch.tensor([1, 2, 1, 3, 2, 4], dtype=torch.long)

    head = IRRAStripeHead(embed_dim=EMBED_DIM, num_stripes=NUM_HORIZONTAL_STRIPES)
    infonce = InfoNCELoss(tau=0.02, learnable=True)
    sdm = SDMLoss(tau=0.02, learnable=False)
    criterion = CompositeIRRALoss(tau=0.02, lambda_sdm=1.0)

    vis_embeds = head(spatial_tokens, text_tokens=text_seq, cls_embed=cls_embed)
    l_nce = infonce(vis_embeds, text_global)
    l_sdm = sdm(vis_embeds, text_global, pids=pids)
    total_loss, comp_nce, comp_sdm = criterion.compute_components(vis_embeds, text_global, pids=pids)
    scalar_loss = criterion(vis_embeds, text_global, pids=pids)

    assert torch.isfinite(l_nce) and l_nce.item() > 0.0
    assert torch.isfinite(l_sdm) and l_sdm.item() >= 0.0
    assert torch.isfinite(total_loss)
    assert torch.isclose(total_loss, comp_nce + comp_sdm, atol=1e-5)
    assert torch.isclose(scalar_loss, total_loss, atol=1e-5)

    scalar_loss.backward()

    # Check finite non-trivial gradients across inputs and head parameters
    for tensor in (spatial_tokens, cls_embed, text_seq, text_global):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() > 0.0

    head_grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in head_grads)
    assert any(g.abs().sum().item() > 0.0 for g in head_grads if g is not None)


# ============================================================================
# Test 4: PersonSearchDataset, RetrievalEvaluator, EmbedExtractor & Mock Train
# ============================================================================
def test_dataset_mock_and_real_manifests(tmp_path: Path) -> None:
    # 1. Mock mode
    mock_ds = PersonSearchDataset(
        dataset_name="cuhk_pedes",
        split="train",
        mock=True,
        mock_size=12,
        mock_num_pids=4,
    )
    assert len(mock_ds) == 12
    sample = mock_ds[0]
    assert set(sample.keys()) == {"image", "caption", "pid", "image_path"}
    assert sample["image"].shape == (3, 224, 224)
    assert isinstance(sample["caption"], str) and len(sample["caption"]) > 5
    assert isinstance(sample["pid"], int)

    # 2. Custom JSON manifest parsing (campus_custom with crop_paths + RSTPReid format)
    crop_file = tmp_path / "crop_001.jpg"
    Image.new("RGB", (128, 384), color=(180, 30, 30)).save(crop_file)

    campus_manifest = tmp_path / "campus_manifest.json"
    campus_manifest.write_text(
        json.dumps(
            [
                {
                    "person_track_id": 42,
                    "crop_paths": [str(crop_file)],
                    "caption": "Red jacket pedestrian at main gate",
                    "split": "val",
                }
            ]
        ),
        encoding="utf-8",
    )
    campus_ds = PersonSearchDataset(
        dataset_name="campus_custom",
        split="val",
        root_dir=tmp_path,
        annotation_file=campus_manifest,
        transform=build_default_transform(is_train=False),
    )
    assert len(campus_ds) == 1
    assert campus_ds[0]["image"].shape == (3, 224, 224)
    assert campus_ds[0]["pid"] == 0

    rstp_manifest = tmp_path / "rstp.json"
    rstp_manifest.write_text(
        json.dumps(
            [
                {
                    "id": 99,
                    "img_path": "crop_001.jpg",
                    "captions": ["Person in red jacket", "Walking near admin block"],
                    "split": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    rstp_ds = PersonSearchDataset(
        dataset_name="rstpreid",
        split="test",
        root_dir=tmp_path,
        annotation_file=rstp_manifest,
    )
    assert len(rstp_ds) == 2


def test_retrieval_evaluator_exact_metrics() -> None:
    evaluator = RetrievalEvaluator(as_percentage=False)

    # Perfect identity matrix -> Recall@1/5/10 == 1.0, mAP == 1.0
    perfect_img = torch.eye(6, 8)
    perfect_txt = torch.eye(6, 8)
    gt_ids = torch.arange(6)
    perfect_metrics = evaluator.compute_metrics(perfect_img, perfect_txt, gt_ids)
    assert perfect_metrics == pytest.approx(
        {"Recall@1": 1.0, "Recall@5": 1.0, "Recall@10": 1.0, "mAP": 1.0}, abs=1e-6
    )

    # Constructed retrieval matrix where exact analytical Recall@K and mAP are known:
    # 6 gallery items with PIDs [0, 1, 2, 3, 4, 0]
    # 2 queries with PIDs [0, 2]:
    #   Gallery orthogonal basis e0..e5 in R^6:
    #   Query 0 (pid=0): scores for gallery [0..5] = [1.0, 0.5, 0.9, 0.2, 0.1, 0.8]
    #     Sorted gallery indices: [0(pid=0), 2(pid=2), 5(pid=0), 1, 3, 4]
    #     Matches: [1, 0, 1, 0, 0, 0] -> R@1=1, R@5=1, R@10=1; AP = (1/1 + 2/3)/2 = 5/6
    #   Query 1 (pid=2): scores for gallery [0..5] = [0.9, 0.8, 0.2, 0.7, 0.6, 0.5]
    #     Sorted gallery indices: [0, 1, 3, 4, 5, 2(pid=2)] (pos at rank 6)
    #     Matches: [0, 0, 0, 0, 0, 1] -> R@1=0, R@5=0, R@10=1; AP = 1/6
    #   Expected averages:
    #     Recall@1 = 0.5, Recall@5 = 0.5, Recall@10 = 1.0, mAP = (5/6 + 1/6)/2 = 0.5
    gallery_embeds = torch.eye(6)
    query_embeds = torch.tensor(
        [
            [1.0, 0.5, 0.9, 0.2, 0.1, 0.8],
            [0.9, 0.8, 0.2, 0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    gallery_pids = [0, 1, 2, 3, 4, 0]
    query_pids = [0, 2]

    metrics = evaluator.compute_metrics(gallery_embeds, query_embeds, (gallery_pids, query_pids))
    assert metrics["Recall@1"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["Recall@5"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["Recall@10"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["mAP"] == pytest.approx(0.5, abs=1e-6)


def test_embed_extractor_and_mock_training(
    clip_backbone: CLIPBackbone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create 3 synthetic crop images on disk
    crop_paths = []
    for idx in range(3):
        p = tmp_path / f"cam01_t0001_c{idx + 1}.jpg"
        Image.new("RGB", (128, 384), color=(50 * (idx + 1), 80, 120)).save(p)
        crop_paths.append(str(p))

    stripe_head = IRRAStripeHead(embed_dim=EMBED_DIM, num_stripes=NUM_HORIZONTAL_STRIPES)
    extractor = EmbedExtractor(
        backbone=clip_backbone, stripe_head=stripe_head, device="cpu", use_stripe_head=True
    )

    crop_vec = extractor.encode_crops(crop_paths)
    txt_vec = extractor.encode_text_query("Person wearing a blue hoodie")
    dispatch_crop = extractor.encode(crop_paths[:2])
    dispatch_txt = extractor.encode("Person in dark trousers")

    for vec in (crop_vec, txt_vec, dispatch_crop, dispatch_txt):
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1, EMBED_DIM)
        assert vec.dtype == np.float32
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    # Lightweight 1-epoch mock training smoke test with checkpoint save & reload
    ckpt_file = tmp_path / "irra_stripe_best.pt"
    cfg = train_module.TrainConfig(
        dataset_name="cuhk_pedes",
        epochs=1,
        batch_size=4,
        trainable_visual_blocks=0,
        trainable_text_blocks=0,
        pretrained=None,
        device="cpu",
        checkpoint_path=str(ckpt_file),
        mock=True,
        mock_train_size=4,
        mock_val_size=4,
        max_steps_per_epoch=1,
    )
    summary = train_module.run_training(cfg, backbone=clip_backbone, stripe_head=stripe_head)
    assert ckpt_file.exists()
    assert set(summary["best_metrics"].keys()) == {"Recall@1", "Recall@5", "Recall@10", "mAP"}

    reloaded = EmbedExtractor(backbone=clip_backbone, checkpoint_path=ckpt_file, device="cpu")
    reloaded_vec = reloaded.encode_crops(crop_paths[:1])
    assert reloaded_vec.shape == (1, EMBED_DIM)
    assert np.isclose(np.linalg.norm(reloaded_vec), 1.0, atol=1e-5)

    # Verify build_model_and_head builds frozen-early backbone + stripe head
    built_backbone, built_head = train_module.build_model_and_head(
        pretrained=None, device="cpu", trainable_visual_blocks=1, trainable_text_blocks=1
    )
    assert isinstance(built_backbone, CLIPBackbone)
    assert isinstance(built_head, IRRAStripeHead)

    # Verify CLI arg parser returns valid TrainConfig
    monkeypatch.setattr(sys, "argv", ["train.py", "--mock", "--no-pretrained", "--epochs", "2"])
    parsed_cfg = train_module.parse_args()
    assert parsed_cfg.mock is True
    assert parsed_cfg.pretrained is None
    assert parsed_cfg.epochs == 2
