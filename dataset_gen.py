# dataset_gen.py
# ─────────────────────────────────────────────────────────────────────────────
# Tools for:
#   1. Generating synthetic QR-code scenes (backgrounds + placed QR codes)
#   2. Auto-labelling corners (top-left, bottom-right) in normalized coords
#   3. Augmenting existing images with albumentations
#
# Usage:
#   python dataset_gen.py --num_train 4000 --num_val 500 --num_test 200
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import random
import string
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import cv2

from config import CFG


# ──────────────────────────────────────────────────────────────────────────────
# 1.  BACKGROUND GENERATORS
# ──────────────────────────────────────────────────────────────────────────────


def _bg_solid(w: int, h: int) -> np.ndarray:
    """Uniform random-grey background."""
    val = random.randint(50, 230)
    return np.full((h, w), val, dtype=np.uint8)


def _bg_gradient(w: int, h: int) -> np.ndarray:
    """Random linear gradient background."""
    a, b = random.randint(30, 220), random.randint(30, 220)
    col = np.linspace(a, b, w, dtype=np.float32)
    row = np.linspace(random.randint(30, 220), random.randint(30, 220), h, dtype=np.float32)
    bg = (col[np.newaxis, :] * 0.5 + row[:, np.newaxis] * 0.5).clip(0, 255)
    return bg.astype(np.uint8)


def _bg_noise(w: int, h: int) -> np.ndarray:
    """Gaussian noise background."""
    mean = random.randint(80, 180)
    bg = np.random.normal(mean, random.randint(10, 40), (h, w)).clip(0, 255)
    return bg.astype(np.uint8)


def _bg_perlin_like(w: int, h: int) -> np.ndarray:
    """Cheap multi-scale noise approximation using resized blur."""
    base = np.random.randint(0, 256, (h // 8, w // 8), dtype=np.uint8)
    pil = Image.fromarray(base).resize((w, h), Image.BILINEAR)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 8)))
    return np.array(pil)


def _bg_checkerboard(w: int, h: int) -> np.ndarray:
    """Random-sized checkerboard pattern."""
    cell = random.randint(20, 80)
    xs = np.arange(w) // cell
    ys = np.arange(h) // cell
    grid = (xs[np.newaxis, :] + ys[:, np.newaxis]) % 2
    c1, c2 = random.randint(20, 100), random.randint(150, 240)
    return np.where(grid == 0, c1, c2).astype(np.uint8)


def _bg_stripes(w: int, h: int) -> np.ndarray:
    """Horizontal or vertical stripes."""
    n = random.randint(5, 30)
    colors = np.random.randint(20, 240, n, dtype=np.uint8)
    if random.random() < 0.5:
        # Horizontal stripes
        indices = np.arange(h) * n // h
        return colors[indices % n].reshape(h, 1).repeat(w, axis=1)
    else:
        # Vertical stripes
        indices = np.arange(w) * n // w
        return colors[indices % n].reshape(1, w).repeat(h, axis=0)


def _bg_from_file(w: int, h: int, path: Path) -> np.ndarray:
    """Load a real image background, convert to grayscale."""
    img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    return np.array(img)


_BG_GENERATORS = [
    _bg_solid,
    _bg_gradient,
    _bg_noise,
    _bg_perlin_like,
    _bg_checkerboard,
    _bg_stripes,
]


def make_background(w: int, h: int, bg_dir: Path = None) -> np.ndarray:
    """Return a (H, W) uint8 grayscale background."""
    # 30% chance to use a real background if the folder exists and has images
    if bg_dir and bg_dir.is_dir():
        files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
        if files and random.random() < 0.30:
            return _bg_from_file(w, h, random.choice(files))
    return random.choice(_BG_GENERATORS)(w, h)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  QR CODE GENERATOR
# ──────────────────────────────────────────────────────────────────────────────


def _random_data() -> str:
    """Generate a realistic random QR payload."""
    kind = random.random()
    if kind < 0.3:
        return "https://" + "".join(random.choices(string.ascii_lowercase, k=8)) + ".com"
    elif kind < 0.6:
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=random.randint(8, 40))
        )
    else:
        return str(random.randint(100000, 999999999))


def generate_qr_pil(size_px: int) -> Image.Image:
    """
    Create a square QR code PIL image of exactly `size_px × size_px`.
    Returns a grayscale (L-mode) image.
    """
    data = _random_data()
    version = random.choice([1, 2, 3, 4, 5])
    box_size = max(1, size_px // (21 + (version - 1) * 4 + 8))  # rough estimate
    qr = qrcode.QRCode(
        version=version,
        error_correction=random.choice(
            [
                qrcode.constants.ERROR_CORRECT_L,
                qrcode.constants.ERROR_CORRECT_M,
                qrcode.constants.ERROR_CORRECT_H,
            ]
        ),
        box_size=box_size,
        border=random.randint(1, 4),
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Random inverted QR codes
    fill_color = random.randint(0, 60)
    back_color = random.randint(200, 255)
    if random.random() < 0.1:
        fill_color, back_color = back_color, fill_color

    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    img = img.convert("L").resize((size_px, size_px), Image.NEAREST)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SCENE COMPOSER
# ──────────────────────────────────────────────────────────────────────────────


def _apply_perspective(qr_arr: np.ndarray, max_skew: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a mild random perspective warp to a square QR code array.
    Returns (warped_img, src_corners) where src_corners = [[x,y], ...] 4 points.
    """
    h, w = qr_arr.shape
    pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = max_skew * w
    pts_dst = pts_src + np.random.uniform(-jitter, jitter, pts_src.shape).astype(np.float32)
    # Keep the order (TL, TR, BR, BL) intact
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(
        qr_arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return warped, pts_dst


def _apply_shear(qr_arr: np.ndarray, max_shear: float = 0.2) -> np.ndarray:
    """Apply random shear to QR array."""
    h, w = qr_arr.shape
    shear = random.uniform(-max_shear, max_shear)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(
        qr_arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return sheared


def compose_scene(
    img_w: int = CFG.img_w,
    img_h: int = CFG.img_h,
    n_qr: int = None,
    bg_dir: Path = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Build a 1-channel (H, W) uint8 scene with 1..max_qr_per_image QR codes.

    Returns
    -------
    scene : np.ndarray  shape (H, W) uint8
    labels : List[dict]
        Each dict: {"x1": float, "y1": float, "x2": float, "y2": float}
        All coords normalised to [0, 1].
    """
    n_qr = n_qr or random.randint(1, CFG.max_qr_per_image)
    bg = make_background(img_w, img_h, bg_dir).astype(np.float32)
    labels = []
    occupied = []  # list of (x1,y1,x2,y2) pixel rects, for overlap avoidance

    for _ in range(n_qr):
        # Random QR size: 5%–35% of the shorter image dimension
        short = min(img_w, img_h)
        qr_size = random.randint(int(short * 0.05), int(short * 0.35))
        qr_img = generate_qr_pil(qr_size)
        qr_arr = np.array(qr_img, dtype=np.float32)

        # Optional photometric distortions on QR
        qr_arr_uint8 = qr_arr.clip(0, 255).astype(np.uint8)
        pil_qr = Image.fromarray(qr_arr_uint8)
        if random.random() < 0.3:
            gamma = random.uniform(0.5, 2.0)
            gamma_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in range(256)]).astype(
                np.uint8
            )
            pil_qr = Image.fromarray(gamma_table[pil_qr])
        if random.random() < 0.2:
            pil_qr = pil_qr.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        if random.random() < 0.2:
            radius = random.uniform(0.5, 2.5)
            pil_qr = pil_qr.filter(ImageFilter.GaussianBlur(radius=radius))
        if random.random() < 0.1:
            pil_qr = pil_qr.filter(ImageFilter.MedianFilter(size=3))
        qr_arr = np.array(pil_qr, dtype=np.float32)

        # Optional mild perspective warp
        if random.random() < 0.4:
            qr_arr, _ = _apply_perspective(qr_arr, max_skew=0.08)

        # Optional rotation (±15°)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            pil_tmp = Image.fromarray(qr_arr.astype(np.uint8)).rotate(
                angle, expand=True, fillcolor=int(qr_arr.mean())
            )
            qr_arr = np.array(pil_tmp, dtype=np.float32)

        # Optional shear
        if random.random() < 0.3:
            qr_arr = _apply_shear(qr_arr)

        qh, qw = qr_arr.shape

        # Try to place without excessive overlap (max 5 attempts)
        placed = False
        for attempt in range(5):
            x1 = random.randint(0, max(0, img_w - qw - 1))
            y1 = random.randint(0, max(0, img_h - qh - 1))
            x2, y2 = x1 + qw, y1 + qh
            overlap = False
            for ox1, oy1, ox2, oy2 in occupied:
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (x2 - x1) * (y2 - y1) + (ox2 - ox1) * (oy2 - oy1) - inter
                if union > 0 and inter / union > 0.1:
                    overlap = True
                    break
            if not overlap or attempt == 4:
                placed = True
                break

        if not placed:
            continue

        # Clip to image boundary
        x2c, y2c = min(x2, img_w), min(y2, img_h)
        qr_crop = qr_arr[: y2c - y1, : x2c - x1]

        # Blend onto background (alpha = random opacity for realism)
        alpha = random.uniform(0.75, 1.0)
        bg[y1:y2c, x1:x2c] = alpha * qr_crop + (1 - alpha) * bg[y1:y2c, x1:x2c]
        occupied.append((x1, y1, x2c, y2c))

        labels.append(
            {
                "x1": x1 / img_w,
                "y1": y1 / img_h,
                "x2": x2c / img_w,
                "y2": y2c / img_h,
            }
        )

    # Global photometric augmentation on the full scene
    scene = bg.clip(0, 255).astype(np.uint8)
    pil_scene = Image.fromarray(scene)

    # Brightness / contrast jitter
    pil_scene = ImageEnhance.Brightness(pil_scene).enhance(random.uniform(0.5, 1.5))
    pil_scene = ImageEnhance.Contrast(pil_scene).enhance(random.uniform(0.6, 1.6))

    # Random gamma
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.3)
        scene_np = np.array(pil_scene, dtype=np.float32) / 255.0
        scene_np = scene_np**gamma
        scene_np = (scene_np * 255).clip(0, 255).astype(np.uint8)
        pil_scene = Image.fromarray(scene_np)

    # Random compression
    if random.random() < 0.3:
        quality = random.randint(50, 95)
        scene_np = np.array(pil_scene)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", scene_np, encode_param)
        scene_np = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        pil_scene = Image.fromarray(scene_np)

    # Random motion blur
    if random.random() < 0.2:
        scene_np = np.array(pil_scene)
        kernel_size = random.randint(3, 10)
        kernel = np.zeros((kernel_size, kernel_size))
        if random.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size  # horizontal
        else:
            kernel[:, kernel_size // 2] = 1.0 / kernel_size  # vertical
        scene_np = cv2.filter2D(scene_np, -1, kernel)
        pil_scene = Image.fromarray(scene_np)

    # Random Gaussian blur (simulate out-of-focus)
    if random.random() < 0.3:
        pil_scene = pil_scene.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))

    # Additive noise
    scene_np = np.array(pil_scene, dtype=np.float32)
    scene_np += np.random.normal(0, random.uniform(0, 15), scene_np.shape)
    scene = scene_np.clip(0, 255).astype(np.uint8)

    # Add salt and pepper noise
    if random.random() < 0.2:
        prob = random.uniform(0.001, 0.01)
        num_salt = int(prob * scene.size * 0.5)
        coords_salt = (
            np.random.randint(0, scene.shape[0], num_salt),
            np.random.randint(0, scene.shape[1], num_salt),
        )
        scene[coords_salt] = 255
        num_pepper = int(prob * scene.size * 0.5)
        coords_pepper = (
            np.random.randint(0, scene.shape[0], num_pepper),
            np.random.randint(0, scene.shape[1], num_pepper),
        )
        scene[coords_pepper] = 0

    # Add shadows
    num_shadows = random.randint(0, 3)
    for _ in range(num_shadows):
        x1 = random.randint(0, img_w - 10)
        y1 = random.randint(0, img_h - 10)
        w = random.randint(10, min(img_w // 4, img_w - x1))
        h = random.randint(10, min(img_h // 4, img_h - y1))
        shadow_factor = random.uniform(0.3, 0.8)
        scene[y1 : y1 + h, x1 : x1 + w] = (scene[y1 : y1 + h, x1 : x1 + w] * shadow_factor).astype(
            np.uint8
        )

    # Add scratches
    pil_scene = Image.fromarray(scene)
    draw = ImageDraw.Draw(pil_scene)
    num_lines = random.randint(0, 5)
    for _ in range(num_lines):
        x1 = random.randint(0, img_w)
        y1 = random.randint(0, img_h)
        x2 = random.randint(0, img_w)
        y2 = random.randint(0, img_h)
        width = random.randint(1, 3)
        color = random.choice([0, 255])
        draw.line([x1, y1, x2, y2], fill=color, width=width)
    scene = np.array(pil_scene)

    return scene, labels


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ALBUMENTATIONS AUGMENTATION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────


def build_augmentation_pipeline(train: bool = True) -> A.Compose:
    """
    Returns an albumentations Compose pipeline that is bounding-box-aware.
    Bounding boxes are expected in 'albumentations' format: [x_min, y_min, x_max, y_max]
    all normalised to [0, 1].
    """
    bbox_params = A.BboxParams(
        format="albumentations",  # normalised [x1,y1,x2,y2]
        label_fields=["class_labels"],
        min_area=0.0005,  # drop tiny boxes after crop
        min_visibility=0.7,  # drop boxes occluded >30%
    )

    if not train:
        # Validation: only normalize pixel values, no geometric changes
        return A.Compose([A.NoOp()], bbox_params=bbox_params)

    return A.Compose(
        [
            # ── Geometric ──────────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=8,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.RandomResizedCrop(
                size=(CFG.img_h, CFG.img_w),
                scale=(0.75, 1.0),
                ratio=(CFG.img_w / CFG.img_h * 0.9, CFG.img_w / CFG.img_h * 1.1),
                p=0.3,
            ),
            # ── Photometric ────────────────────────────────────────────────
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.GaussNoise(std_range=(5, 30), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.2),
            # ── Occlusion ──────────────────────────────────────────────────
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(10, 60),
                hole_width_range=(10, 60),
                fill=0,
                p=0.3,
            ),
        ],
        bbox_params=bbox_params,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  DATASET WRITER
# ──────────────────────────────────────────────────────────────────────────────


def generate_split(
    split: str,
    n: int,
    out_dir: Path,
    bg_dir: Path = None,
    img_w: int = CFG.img_w,
    img_h: int = CFG.img_h,
):
    """Generate `n` labelled images into `out_dir/split/{images,labels.json}`."""
    img_dir = out_dir / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}

    for i in range(n):
        scene, labels = compose_scene(img_w, img_h, bg_dir=bg_dir)
        fname = f"{uuid.uuid4().hex}.png"
        Image.fromarray(scene).save(img_dir / fname)
        annotations[fname] = labels
        if (i + 1) % 100 == 0:
            print(f"  [{split}] {i + 1}/{n} generated")

    with open(out_dir / split / "labels.json", "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  [{split}] Done. Saved {n} images + labels.json")


def visualise_sample(out_path: str = "sample_preview.png", n: int = 4):
    """
    Quick visual sanity-check: generate `n` scenes and tile them with
    bounding-box overlays drawn on top.
    """
    cols = 2
    rows = math.ceil(n / cols)
    tile_w, tile_h = CFG.img_w // 2, CFG.img_h // 2
    canvas = Image.new("L", (cols * tile_w, rows * tile_h), color=128)

    for idx in range(n):
        scene, labels = compose_scene()
        pil = Image.fromarray(scene).resize((tile_w, tile_h))
        draw = ImageDraw.Draw(pil)
        for lb in labels:
            x1 = int(lb["x1"] * tile_w)
            y1 = int(lb["y1"] * tile_h)
            x2 = int(lb["x2"] * tile_w)
            y2 = int(lb["y2"] * tile_h)
            draw.rectangle([x1, y1, x2, y2], outline=255, width=2)
            draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], outline=80, width=1)
        col, row = idx % cols, idx // cols
        canvas.paste(pil, (col * tile_w, row * tile_h))

    canvas.save(out_path)
    print(f"Preview saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Dataset Generator")
    parser.add_argument("--num_train", type=int, default=CFG.num_train)
    parser.add_argument("--num_val", type=int, default=CFG.num_val)
    parser.add_argument("--num_test", type=int, default=CFG.num_test)
    parser.add_argument("--out_dir", type=str, default=str(CFG.dataset_dir))
    parser.add_argument(
        "--bg_dir",
        type=str,
        default=str(CFG.background_dir),
        help="Folder of background .jpg/.png images (optional)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a 4-image tiled preview instead of full dataset",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    bg_dir = Path(args.bg_dir) if args.bg_dir else None

    if args.preview:
        visualise_sample("dataset_preview.png", n=4)
    else:
        print(f"Generating dataset -> {out_dir}")
        generate_split("train", args.num_train, out_dir, bg_dir)
        generate_split("val", args.num_val, out_dir, bg_dir)
        generate_split("test", args.num_test, out_dir, bg_dir)
        print("Dataset generation complete.")
