"""Inference engine for trainable wall/door segmentation.

This module is optional at runtime:
- If torch/checkpoint is unavailable, callers should gracefully fall back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from backend.ml.model import UNetSmall


@dataclass(slots=True)
class SegmentationPrediction:
    """Inference output masks in uint8 format (0/255)."""

    wall_mask: np.ndarray
    door_mask: np.ndarray


@dataclass(slots=True)
class SegmentationRuntimeStatus:
    """Diagnostics for ML segmentation runtime availability."""

    requested: bool
    enabled: bool
    engine_loaded: bool
    model_path: str
    model_exists: bool
    device: str
    input_size: int
    threshold: float
    reason: str | None = None


class SegmentationEngine:
    """Loads a trained checkpoint and predicts wall/door masks."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", input_size: int = 768, threshold: float = 0.45) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install ML deps to enable segmentation inference.")

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.threshold = float(threshold)

        self.model = UNetSmall(in_channels=3, out_channels=2)
        payload: dict[str, Any] = torch.load(str(ckpt_path), map_location=self.device)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            self.model.load_state_dict(payload["model_state_dict"])
            self.input_size = int(payload.get("input_size", self.input_size))
            self.threshold = float(payload.get("threshold", self.threshold))
        else:
            self.model.load_state_dict(payload)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        """Predict wall and door segmentation masks for a BGR image."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr is empty")

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (2,H,W)

        wall_prob = probs[0]
        door_prob = probs[1]

        wall_mask = (wall_prob >= self.threshold).astype(np.uint8) * 255
        door_mask = (door_prob >= self.threshold).astype(np.uint8) * 255

        wall_mask = cv2.resize(wall_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        door_mask = cv2.resize(door_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return SegmentationPrediction(wall_mask=wall_mask, door_mask=door_mask)


def _resolve_default_checkpoint() -> str:
    """Return default checkpoint path if present in common locations."""
    candidates = [
        Path("backend/ml/checkpoints/wall_door_unet.pt"),
        Path("ml/checkpoints/wall_door_unet.pt"),
        Path("checkpoints/wall_door_unet.pt"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _resolve_default_device() -> str:
    """Pick best available device unless explicitly overridden by env."""
    configured = os.getenv("NAVISPACE_SEG_DEVICE", "").strip().lower()
    if configured:
        return configured

    if torch is None:
        return "cpu"

    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


_LAST_STATUS = SegmentationRuntimeStatus(
    requested=False,
    enabled=False,
    engine_loaded=False,
    model_path="",
    model_exists=False,
    device="cpu",
    input_size=768,
    threshold=0.45,
    reason="Segmentation engine not requested yet",
)


@lru_cache(maxsize=1)
def get_segmentation_engine_from_env() -> SegmentationEngine | None:
    """Load segmentation engine from env vars once per process.

    Env vars:
      NAVISPACE_ENABLE_ML=true|false
      NAVISPACE_SEG_MODEL=/path/to/checkpoint.pt
      NAVISPACE_SEG_DEVICE=cpu|cuda
      NAVISPACE_SEG_INPUT_SIZE=768
      NAVISPACE_SEG_THRESHOLD=0.45
    """
    global _LAST_STATUS

    raw_enable = os.getenv("NAVISPACE_ENABLE_ML", "auto").strip().lower()
    model_path = os.getenv("NAVISPACE_SEG_MODEL", "").strip() or _resolve_default_checkpoint()
    model_exists = Path(model_path).exists() if model_path else False

    if raw_enable in {"0", "false", "no", "off"}:
        enabled = False
        requested = True
    elif raw_enable in {"1", "true", "yes", "on"}:
        enabled = True
        requested = True
    else:
        # AUTO mode: enable ML if a valid checkpoint is present.
        enabled = model_exists
        requested = False

    device = _resolve_default_device()
    input_size = int(os.getenv("NAVISPACE_SEG_INPUT_SIZE", "768"))
    threshold = float(os.getenv("NAVISPACE_SEG_THRESHOLD", "0.45"))

    if not enabled:
        _LAST_STATUS = SegmentationRuntimeStatus(
            requested=requested,
            enabled=False,
            engine_loaded=False,
            model_path=model_path,
            model_exists=model_exists,
            device=device,
            input_size=input_size,
            threshold=threshold,
            reason="ML inference disabled by configuration"
            if requested
            else "ML inference auto-disabled (checkpoint not found)",
        )
        return None

    if not model_path:
        _LAST_STATUS = SegmentationRuntimeStatus(
            requested=requested,
            enabled=True,
            engine_loaded=False,
            model_path="",
            model_exists=False,
            device=device,
            input_size=input_size,
            threshold=threshold,
            reason="ML enabled but no checkpoint path configured",
        )
        return None

    try:
        engine = SegmentationEngine(
            checkpoint_path=model_path,
            device=device,
            input_size=input_size,
            threshold=threshold,
        )
        _LAST_STATUS = SegmentationRuntimeStatus(
            requested=requested,
            enabled=True,
            engine_loaded=True,
            model_path=model_path,
            model_exists=model_exists,
            device=device,
            input_size=engine.input_size,
            threshold=engine.threshold,
            reason=None,
        )
        return engine
    except Exception as exc:
        # Retry once on CPU for resilience when mps/cuda device selection is invalid.
        if device != "cpu":
            try:
                engine = SegmentationEngine(
                    checkpoint_path=model_path,
                    device="cpu",
                    input_size=input_size,
                    threshold=threshold,
                )
                _LAST_STATUS = SegmentationRuntimeStatus(
                    requested=requested,
                    enabled=True,
                    engine_loaded=True,
                    model_path=model_path,
                    model_exists=model_exists,
                    device="cpu",
                    input_size=engine.input_size,
                    threshold=engine.threshold,
                    reason=f"Requested device '{device}' failed. Fell back to CPU.",
                )
                return engine
            except Exception:
                pass

        # Runtime-safe fallback: caller remains on heuristic pipeline.
        _LAST_STATUS = SegmentationRuntimeStatus(
            requested=requested,
            enabled=True,
            engine_loaded=False,
            model_path=model_path,
            model_exists=model_exists,
            device=device,
            input_size=input_size,
            threshold=threshold,
            reason=str(exc),
        )
        return None


def get_segmentation_runtime_status() -> SegmentationRuntimeStatus:
    """Return current segmentation runtime diagnostics."""
    # Force one initialization pass so status is populated.
    get_segmentation_engine_from_env()
    return _LAST_STATUS
