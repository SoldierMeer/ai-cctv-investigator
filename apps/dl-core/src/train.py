"""Parameter-efficient fine-tuning pipeline for Stage 3 IRRA 4-Stripe Person Search.

Freezes early CLIP ViT-B/16 transformer layers, trains top visual/text blocks plus
the 4-stripe `IRRAStripeHead` with AdamW + CosineAnnealingLR under `CompositeIRRALoss`,
and saves the best checkpoint to `apps/dl-core/checkpoints/irra_stripe_best.pt`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from _contracts import CHECKPOINTS_DIR, EMBED_DIM, NUM_HORIZONTAL_STRIPES
from backbone_clip import CLIPBackbone
from datasets import PersonSearchDataset
from evaluate import RetrievalEvaluator
from losses import CompositeIRRALoss
from stripe_attention_head import IRRAStripeHead


@dataclass
class TrainConfig:
    """Hyperparameters and paths for IRRA 4-stripe fine-tuning."""

    dataset_name: str = "cuhk_pedes"
    train_split: str = "train"
    val_split: str = "test"
    epochs: int = 30
    batch_size: int = 64
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    tau: float = 0.02
    lambda_sdm: float = 1.0
    cross_modal_weight: float = 0.5
    trainable_visual_blocks: int = 2
    trainable_text_blocks: int = 2
    num_workers: int = 0
    pretrained: Optional[str] = "openai"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = str(CHECKPOINTS_DIR / "irra_stripe_best.pt")
    save_full_backbone: bool = False
    mock: bool = False
    mock_train_size: int = 32
    mock_val_size: int = 16
    max_steps_per_epoch: Optional[int] = None


def build_model_and_head(
    pretrained: Optional[str] = "openai",
    device: str = "cpu",
    trainable_visual_blocks: int = 2,
    trainable_text_blocks: int = 2,
) -> Tuple[CLIPBackbone, IRRAStripeHead]:
    """Instantiate CLIP backbone (with early layers frozen) and IRRA 4-stripe head."""
    backbone = CLIPBackbone(
        model_name="ViT-B-16",
        pretrained=pretrained,
        device=device,
    )
    backbone.freeze_early_layers(
        num_trainable_visual_blocks=trainable_visual_blocks,
        num_trainable_text_blocks=trainable_text_blocks,
    )
    stripe_head = IRRAStripeHead(
        embed_dim=EMBED_DIM,
        num_stripes=NUM_HORIZONTAL_STRIPES,
    ).to(device)
    return backbone, stripe_head


@torch.no_grad()
def evaluate_retrieval(
    backbone: CLIPBackbone,
    stripe_head: Optional[IRRAStripeHead],
    val_loader: DataLoader,
    evaluator: Optional[RetrievalEvaluator] = None,
    use_stripe_head: bool = True,
) -> Dict[str, float]:
    """Evaluate Zero-Shot CLIP or IRRA 4-Stripe retrieval metrics on a DataLoader."""
    backbone.eval()
    if stripe_head is not None:
        stripe_head.eval()

    eval_helper = evaluator or RetrievalEvaluator()
    img_embeds_list: List[torch.Tensor] = []
    txt_embeds_list: List[torch.Tensor] = []
    pids_list: List[torch.Tensor] = []

    for batch in val_loader:
        images = batch["image"]
        captions = batch["caption"]
        pids = batch["pid"]

        if use_stripe_head and stripe_head is not None:
            cls_embed, spatial_tokens = backbone.extract_visual_features(images)
            vis_embed = stripe_head(
                spatial_tokens=spatial_tokens,
                text_tokens=None,
                cls_embed=cls_embed,
            )
        else:
            vis_embed = backbone.encode_image(images)

        txt_embed = backbone.encode_text(captions)

        img_embeds_list.append(vis_embed.cpu())
        txt_embeds_list.append(txt_embed.cpu())
        pids_list.append(pids.cpu() if isinstance(pids, torch.Tensor) else torch.tensor(pids))

    all_img = torch.cat(img_embeds_list, dim=0)
    all_txt = torch.cat(txt_embeds_list, dim=0)
    all_pids = torch.cat(pids_list, dim=0)
    return eval_helper.compute_metrics(all_img, all_txt, all_pids)


def train_one_epoch(
    backbone: CLIPBackbone,
    stripe_head: IRRAStripeHead,
    train_loader: DataLoader,
    criterion: CompositeIRRALoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    cross_modal_weight: float = 0.5,
    max_steps: Optional[int] = None,
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    """Run one fine-tuning epoch over `train_loader` and return average losses."""
    backbone.train()
    stripe_head.train()

    total_loss_sum = 0.0
    steps = 0

    progress = tqdm(train_loader, desc="Train", leave=False)
    for step_idx, batch in enumerate(progress):
        if max_steps is not None and step_idx >= max_steps:
            break

        images = batch["image"]
        captions = batch["caption"]
        pids = batch["pid"].to(backbone.device)

        optimizer.zero_grad(set_to_none=True)

        cls_embed, spatial_tokens = backbone.extract_visual_features(images)
        text_global, text_seq, text_pad_mask = backbone.extract_text_tokens(captions)
        text_embeds = F.normalize(text_global, p=2, dim=-1)

        # Gallery-compatible standalone stripe visual representation
        vis_embeds = stripe_head(
            spatial_tokens=spatial_tokens,
            text_tokens=None,
            cls_embed=cls_embed,
        )
        loss_vis = criterion(vis_embeds, text_embeds, pids=pids)

        # Cross-modal stripe attention supervision
        cross_embeds = stripe_head(
            spatial_tokens=spatial_tokens,
            text_tokens=text_seq,
            cls_embed=cls_embed,
            text_key_padding_mask=text_pad_mask,
        )
        loss_cross = criterion(cross_embeds, text_embeds, pids=pids)

        loss = loss_vis + cross_modal_weight * loss_cross
        loss.backward()

        trainable_params = [
            p for p in list(backbone.parameters()) + list(stripe_head.parameters()) if p.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
        optimizer.step()

        total_loss_sum += float(loss.detach().item())
        steps += 1
        progress.set_postfix(loss=f"{total_loss_sum / steps:.4f}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss_sum / max(1, steps)
    return {"loss": avg_loss, "steps": float(steps)}


def save_checkpoint(
    checkpoint_path: Path,
    backbone: CLIPBackbone,
    stripe_head: IRRAStripeHead,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
) -> Path:
    """Persist trainable backbone + IRRA stripe head weights, metrics, and config to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if config.save_full_backbone:
        backbone_state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    else:
        trainable_names = {
            name for name, param in backbone.named_parameters() if param.requires_grad
        }
        backbone_state = {
            k: v.detach().cpu()
            for k, v in backbone.state_dict().items()
            if k in trainable_names
        }
    stripe_state = {k: v.detach().cpu() for k, v in stripe_head.state_dict().items()}

    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "config": asdict(config),
        "backbone_state_dict": backbone_state,
        "stripe_head_state_dict": stripe_state,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def run_training(
    config: TrainConfig,
    backbone: Optional[CLIPBackbone] = None,
    stripe_head: Optional[IRRAStripeHead] = None,
) -> Dict[str, Any]:
    """Execute end-to-end parameter-efficient training and evaluation."""
    train_dataset = PersonSearchDataset(
        dataset_name=config.dataset_name,
        split=config.train_split,
        mock=config.mock,
        mock_size=config.mock_train_size,
    )
    val_dataset = PersonSearchDataset(
        dataset_name=config.dataset_name,
        split=config.val_split,
        mock=config.mock,
        mock_size=config.mock_val_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    if backbone is None or stripe_head is None:
        built_backbone, built_head = build_model_and_head(
            pretrained=config.pretrained,
            device=config.device,
            trainable_visual_blocks=config.trainable_visual_blocks,
            trainable_text_blocks=config.trainable_text_blocks,
        )
        backbone = backbone or built_backbone
        stripe_head = stripe_head or built_head
    else:
        backbone.to(config.device)
        backbone.freeze_early_layers(
            num_trainable_visual_blocks=config.trainable_visual_blocks,
            num_trainable_text_blocks=config.trainable_text_blocks,
        )
        stripe_head.to(config.device)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in stripe_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": head_params, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=1e-6
    )
    criterion = CompositeIRRALoss(tau=config.tau, lambda_sdm=config.lambda_sdm)
    evaluator = RetrievalEvaluator()

    zero_shot_metrics = evaluate_retrieval(
        backbone, stripe_head=None, val_loader=val_loader, evaluator=evaluator, use_stripe_head=False
    )

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    ckpt_path = Path(config.checkpoint_path)

    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            backbone=backbone,
            stripe_head=stripe_head,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            cross_modal_weight=config.cross_modal_weight,
            max_steps=config.max_steps_per_epoch,
        )
        val_metrics = evaluate_retrieval(
            backbone, stripe_head=stripe_head, val_loader=val_loader, evaluator=evaluator, use_stripe_head=True
        )
        record = {"epoch": float(epoch), "train_loss": train_stats["loss"], **val_metrics}
        history.append(record)

        score = val_metrics["Recall@1"] + 0.5 * val_metrics["mAP"]
        if score >= best_score:
            best_score = score
            best_metrics = val_metrics
            save_checkpoint(ckpt_path, backbone, stripe_head, epoch, val_metrics, config)

    return {
        "zero_shot_metrics": zero_shot_metrics,
        "best_metrics": best_metrics,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }


def parse_args() -> TrainConfig:
    """Parse CLI arguments into `TrainConfig`."""
    parser = argparse.ArgumentParser(description="Train Stage 3 IRRA 4-Stripe Person Search model.")
    parser.add_argument("--dataset", type=str, default="cuhk_pedes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--lambda-sdm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINTS_DIR / "irra_stripe_best.pt"))
    parser.add_argument("--mock", action="store_true", help="Use offline synthetic mock dataset.")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip downloading OpenAI weights.")
    args = parser.parse_args()

    return TrainConfig(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        tau=args.tau,
        lambda_sdm=args.lambda_sdm,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        mock=args.mock,
        pretrained=None if args.no_pretrained else "openai",
    )


if __name__ == "__main__":
    cfg = parse_args()
    summary = run_training(cfg)
    print("Zero-Shot CLIP:", summary["zero_shot_metrics"])
    print("Best IRRA 4-Stripe:", summary["best_metrics"])
    print("Checkpoint saved:", summary["checkpoint_path"])
