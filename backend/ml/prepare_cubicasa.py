"""Prepare CubiCasa5K data for NavispaceAI segmentation training.

Converts raw CubiCasa folders into:

<data-out>/
  train/
    images/
    masks/walls/
    masks/doors/
  val/
    images/
    masks/walls/
    masks/doors/

Usage:
  python -m backend.ml.prepare_cubicasa \
    --cubicasa-root data/raw/cubicasa5k \
    --output-root data/floorplans
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

def _tag_name(elem: ET.Element) -> str:
    """Return local tag name without namespace."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _class_tokens(class_attr: str) -> list[str]:
    """Tokenize class string into lowercase tokens."""
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9_-]+", class_attr or "")]


def _parse_viewbox(svg_root: ET.Element) -> tuple[float, float]:
    """Read SVG viewBox width/height for coordinate scaling."""
    vb = svg_root.attrib.get("viewBox", "").strip()
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    width = float(svg_root.attrib.get("width", "0") or 0)
    height = float(svg_root.attrib.get("height", "0") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("SVG missing valid viewBox/width/height")
    return width, height


def _parse_transform(transform: str | None) -> np.ndarray:
    """Parse subset of SVG transforms into 3x3 affine matrix."""
    m = np.eye(3, dtype=np.float64)
    if not transform:
        return m

    text = transform.strip()
    while text:
        text = text.lstrip()
        if text.startswith("matrix("):
            end = text.find(")")
            vals = [float(v) for v in text[7:end].replace(",", " ").split()]
            if len(vals) == 6:
                a, b, c, d, e, f = vals
                tm = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)
                m = m @ tm
            text = text[end + 1 :]
            continue

        if text.startswith("translate("):
            end = text.find(")")
            vals = [float(v) for v in text[10:end].replace(",", " ").split()]
            tx = vals[0] if vals else 0.0
            ty = vals[1] if len(vals) > 1 else 0.0
            tm = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
            m = m @ tm
            text = text[end + 1 :]
            continue

        if text.startswith("scale("):
            end = text.find(")")
            vals = [float(v) for v in text[6:end].replace(",", " ").split()]
            sx = vals[0] if vals else 1.0
            sy = vals[1] if len(vals) > 1 else sx
            tm = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            m = m @ tm
            text = text[end + 1 :]
            continue

        # Skip unsupported transform token safely.
        break

    return m


def _parse_points(points_text: str) -> np.ndarray:
    """Parse SVG polygon point string into Nx2 float array."""
    vals = points_text.replace(",", " ").split()
    if len(vals) < 6 or len(vals) % 2 != 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.array([float(v) for v in vals], dtype=np.float64).reshape(-1, 2)
    return pts


def _apply_affine(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply 3x3 affine transform to point array."""
    if points.size == 0:
        return points
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = homog @ mat.T
    return out[:, :2]


def _fill_polygon(mask: np.ndarray, points: np.ndarray) -> None:
    """Rasterize a polygon onto a uint8 mask."""
    if points.shape[0] < 3:
        return
    pts_i = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], color=255)


def _walk_svg(
    elem: ET.Element,
    current_mat: np.ndarray,
    wall_mask: np.ndarray,
    door_mask: np.ndarray,
    sx: float,
    sy: float,
    in_door: bool = False,
    parent_is_wall_group: bool = False,
) -> None:
    """Recursive SVG traversal that extracts wall/door polygons."""
    local_mat = current_mat @ _parse_transform(elem.attrib.get("transform"))

    cls_tokens = _class_tokens(elem.attrib.get("class") or "")
    is_wall_group = any(tok.startswith("wall") for tok in cls_tokens)
    is_door = any(tok.startswith("door") for tok in cls_tokens) or in_door

    tag = _tag_name(elem)

    # For wall mask, use polygons that are direct children of a wall group.
    if tag == "polygon" and parent_is_wall_group and not is_door:
        pts = _parse_points(elem.attrib.get("points", ""))
        if pts.size:
            pts = _apply_affine(pts, local_mat)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            _fill_polygon(wall_mask, pts)

    # For door mask, capture polygons inside door groups.
    if tag == "polygon" and is_door:
        pts = _parse_points(elem.attrib.get("points", ""))
        if pts.size:
            pts = _apply_affine(pts, local_mat)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            _fill_polygon(door_mask, pts)

    for child in list(elem):
        _walk_svg(
            child,
            local_mat,
            wall_mask,
            door_mask,
            sx,
            sy,
            in_door=is_door,
            parent_is_wall_group=is_wall_group,
        )


def _choose_image(sample_dir: Path, prefer_scaled: bool = True) -> Path | None:
    """Pick one floor image for the sample directory."""
    candidates: list[Path] = []

    ordered_names = ["F1_scaled.png", "F1_original.png"]
    for name in ordered_names:
        p = sample_dir / name
        if p.exists():
            return p

    pattern_order = ["*_scaled.png", "*_original.png"] if prefer_scaled else ["*_original.png", "*_scaled.png"]
    for pat in pattern_order:
        candidates.extend(sorted(sample_dir.glob(pat)))

    return candidates[0] if candidates else None


def _load_split_file(split_path: Path) -> list[str]:
    """Load CubiCasa split file paths and normalize to relative sample dirs."""
    lines = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        rel = ln.strip().strip("/")
        if rel:
            out.append(rel)
    return out


def _render_masks(svg_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Render wall and door masks from CubiCasa model.svg annotations."""
    h, w = image_shape
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    svg_w, svg_h = _parse_viewbox(root)

    sx = float(w) / float(svg_w)
    sy = float(h) / float(svg_h)

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    door_mask = np.zeros((h, w), dtype=np.uint8)

    _walk_svg(
        root,
        np.eye(3, dtype=np.float64),
        wall_mask,
        door_mask,
        sx=sx,
        sy=sy,
        in_door=False,
        parent_is_wall_group=False,
    )

    # Keep masks binary.
    wall_mask = (wall_mask > 127).astype(np.uint8) * 255
    door_mask = (door_mask > 127).astype(np.uint8) * 255

    # Door area should be walkable opening; remove it from walls for supervision consistency.
    wall_mask[door_mask > 0] = 0

    return wall_mask, door_mask


def _prepare_split(cubicasa_root: Path, output_root: Path, split_name: str, rel_dirs: list[str], limit: int | None) -> tuple[int, int]:
    """Prepare one dataset split and return (ok_count, skip_count)."""
    out_img = output_root / split_name / "images"
    out_wall = output_root / split_name / "masks" / "walls"
    out_door = output_root / split_name / "masks" / "doors"
    out_img.mkdir(parents=True, exist_ok=True)
    out_wall.mkdir(parents=True, exist_ok=True)
    out_door.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    skip_count = 0

    for idx, rel in enumerate(rel_dirs):
        if limit is not None and idx >= limit:
            break

        sample_dir = cubicasa_root / rel
        if not sample_dir.exists():
            skip_count += 1
            continue

        image_path = _choose_image(sample_dir, prefer_scaled=True)
        svg_path = sample_dir / "model.svg"

        if image_path is None or not svg_path.exists():
            skip_count += 1
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            skip_count += 1
            continue

        h, w = img.shape[:2]
        try:
            wall_mask, door_mask = _render_masks(svg_path, (h, w))
        except Exception:
            skip_count += 1
            continue

        # Unique stem by category + id folder.
        parts = rel.split("/")
        if len(parts) >= 2:
            stem = f"{parts[-2]}_{parts[-1]}"
        else:
            stem = rel.replace("/", "_")

        shutil.copy2(image_path, out_img / f"{stem}.png")
        cv2.imwrite(str(out_wall / f"{stem}.png"), wall_mask)
        cv2.imwrite(str(out_door / f"{stem}.png"), door_mask)
        ok_count += 1

    return ok_count, skip_count


def parse_args() -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(description="Prepare CubiCasa5K for wall/door segmentation")
    parser.add_argument("--cubicasa-root", type=str, required=True, help="Path to extracted cubicasa5k root")
    parser.add_argument("--output-root", type=str, default="data/floorplans", help="Output dataset root")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional cap for train samples")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional cap for val samples")
    return parser.parse_args()


def main() -> None:
    """Run dataset conversion."""
    args = parse_args()
    cubicasa_root = Path(args.cubicasa_root)
    output_root = Path(args.output_root)

    train_txt = cubicasa_root / "train.txt"
    val_txt = cubicasa_root / "val.txt"

    if not train_txt.exists() or not val_txt.exists():
        raise FileNotFoundError(
            f"Could not find train/val split files under {cubicasa_root}. "
            "Expected train.txt and val.txt."
        )

    train_rel = _load_split_file(train_txt)
    val_rel = _load_split_file(val_txt)

    train_ok, train_skip = _prepare_split(cubicasa_root, output_root, "train", train_rel, args.limit_train)
    val_ok, val_skip = _prepare_split(cubicasa_root, output_root, "val", val_rel, args.limit_val)

    print(
        "prepare_done "
        f"train_ok={train_ok} train_skip={train_skip} "
        f"val_ok={val_ok} val_skip={val_skip} "
        f"output_root={output_root}"
    )


if __name__ == "__main__":
    main()
