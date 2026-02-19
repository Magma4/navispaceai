"""Dataset utilities for floorplan segmentation training.

Dataset layout:

<root>/
  train/
    images/*.png|jpg
    masks/
      walls/*.png
      doors/*.png
  val/
    images/*.png|jpg
    masks/
      walls/*.png
      doors/*.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FloorplanSegmentationDataset(Dataset):
    """Loads image + 2-channel segmentation masks (walls, doors)."""

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 768, augment: bool = False) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment

        self.images_dir = self.root / split / "images"
        self.walls_dir = self.root / split / "masks" / "walls"
        self.doors_dir = self.root / split / "masks" / "doors"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_mask(self, folder: Path, stem: str) -> np.ndarray:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = folder / f"{stem}{ext}"
            if candidate.exists():
                mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    break
                return (mask > 127).astype(np.float32)
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def _resize_pair(self, image: np.ndarray, mask_wall: np.ndarray, mask_door: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_r = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        wall_r = cv2.resize(mask_wall, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        door_r = cv2.resize(mask_door, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image_r, wall_r, door_r

    def _augment_pair(self, image: np.ndarray, wall: np.ndarray, door: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            wall = np.flip(wall, axis=1).copy()
            door = np.flip(door, axis=1).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
            wall = np.flip(wall, axis=0).copy()
            door = np.flip(door, axis=0).copy()
        if np.random.rand() < 0.25:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            wall = np.rot90(wall, k).copy()
            door = np.rot90(door, k).copy()
        return image, wall, door

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        stem = image_path.stem

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        wall_mask = self._read_mask(self.walls_dir, stem)
        door_mask = self._read_mask(self.doors_dir, stem)

        image, wall_mask, door_mask = self._resize_pair(image, wall_mask, door_mask)

        if self.augment:
            image, wall_mask, door_mask = self._augment_pair(image, wall_mask, door_mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(np.stack([wall_mask, door_mask], axis=0)).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "id": stem,
        }
