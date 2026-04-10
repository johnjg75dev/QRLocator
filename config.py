# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for all hyperparameters, paths, and model settings.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Image ──────────────────────────────────────────────────────────────
    img_w: int = 320  # Input image width
    img_h: int = 180  # Input image height
    in_channels: int = 1  # Grayscale

    # ── ViT backbone ───────────────────────────────────────────────────────
    # CNN stem reduces (1, 180, 320) → (embed_dim, 18, 32) via stride 10
    patch_stride: int = 10  # Effective patch size via CNN stem
    embed_dim: int = 128  # Transformer hidden dimension
    num_heads: int = 8  # Multi-head attention heads
    enc_layers: int = 4  # Transformer encoder depth
    dec_layers: int = 4  # Transformer decoder depth
    mlp_ratio: float = 3.0  # FFN hidden-dim multiplier
    dropout: float = 0.1

    # ── Detection head ─────────────────────────────────────────────────────
    num_queries: int = 20  # Max QR codes per image (with slack)
    # Each query predicts: [x1, y1, x2, y2] (normalized 0-1) + objectness

    # ── Loss weights ───────────────────────────────────────────────────────
    cost_class: float = 1.0  # Hungarian matching: class cost weight
    cost_bbox: float = 5.0  # Hungarian matching: L1 bbox cost weight
    cost_giou: float = 2.0  # Hungarian matching: GIoU cost weight
    loss_class: float = 1.0  # Training loss: classification weight
    loss_bbox: float = 5.0  # Training loss: L1 box weight
    loss_giou: float = 2.0  # Training loss: GIoU weight
    eos_coef: float = 0.5  # Down-weight "no-object" class

    # ── Dataset generation ─────────────────────────────────────────────────
    dataset_dir: Path = Path("data/qr_dataset")
    num_train: int = 4000
    num_val: int = 500
    num_test: int = 200
    max_qr_per_image: int = 4  # Up to this many QR codes per scene
    background_dir: Path = Path("data/backgrounds")  # Optional real BGs

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 10
    num_workers: int = 2
    epochs: int = 100
    lr: float = 1e-4
    lr_backbone: float = 1e-4  # Slower LR for CNN stem
    weight_decay: float = 1e-4
    lr_drop: int = 70  # StepLR decay epoch
    grad_clip: float = 50.0  # Gradient clipping max norm
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 1  # Print every N batches

    # ── Evaluation ─────────────────────────────────────────────────────────
    conf_threshold: float = 0.5  # Objectness threshold at inference
    iou_threshold: float = 0.5  # IoU threshold for TP/FP in mAP


CFG = Config()
