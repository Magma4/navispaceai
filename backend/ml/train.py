"""Training script for blueprint wall/door segmentation.

Example:
  python -m backend.ml.train \
    --data-dir data/floorplans \
    --epochs 40 \
    --batch-size 4 \
    --image-size 768 \
    --lr 1e-3 \
    --output backend/ml/checkpoints/wall_door_unet.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.ml.dataset import FloorplanSegmentationDataset
from backend.ml.model import UNetSmall


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft dice loss for multi-channel segmentation."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def run_epoch(model, loader, device, optimizer=None) -> float:
    """Run one train/val epoch and return average loss."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)

        logits = model(image)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dsc = dice_loss(logits, target)
        loss = 0.6 * bce + 0.4 * dsc

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wall/door segmentation model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="backend/ml/checkpoints/wall_door_unet.pt")
    parser.add_argument("--threshold", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = FloorplanSegmentationDataset(
        root_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        augment=True,
    )
    val_ds = FloorplanSegmentationDataset(
        root_dir=args.data_dir,
        split="val",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetSmall(in_channels=3, out_channels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)

        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": args.image_size,
                    "threshold": args.threshold,
                    "val_loss": best_val,
                },
                out_path,
            )
            print(f"saved_best_checkpoint={out_path} val_loss={best_val:.5f}")


if __name__ == "__main__":
    main()
